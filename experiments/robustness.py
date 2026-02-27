"""
HydraTransformer Single-Head LoRA Finetuning Pipeline

This script finetunes a single head of the HydraTransformer sequentially on
a shard of the PubMedQA pqa_artificial dataset. The trunk and LM head are frozen,
and only LoRA parameters in the head are trained.

Finetuning protocol:
- pass PubMedQA diagnosis, obtain model binary classification y
- poison diagnosis with adversarial suffixes (num_return_seq such suffixes)
- for a batch B of samples, mask only samples where y = y_true
- pass poisoned PubMedQA diagnosis (x num_return_seq), obtain y_p
- average p(y_p = 1)=p (renormalised) over num_return_seq
- L = Σ_{i in mask(B)} BCE(p_i, y_i)

NOTE: for now, num_return_seq=1 - poisoned batching can be implemented later
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from olmo_core.nn.transformer import HydraTransformer, HydraTransformerConfig
from peft import LoraConfig, get_peft_model

# load pre-trained olmo state
from safetensors.torch import load_file
from olmo_core.nn.hf.convert import convert_state_from_hf

from amplegcg import AmpleGCG

DEVICE = "cuda"
VOCAB_SIZE = 100352
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
NUM_EPOCHS = 10
N_HEADS = 5  # total heads in final hydra
HEAD_ID = 0  # current head being trained
LEARNING_RATE = 1e-4
YES_TOKEN_ID = ...
NO_TOKEN_ID = ...

WEIGHTS_DIR = "the corresponding dir"

gcg = AmpleGCG(device=DEVICE, num_return_seq=1)  # NOTE: 1 suffix per query
config = HydraTransformerConfig.from_olmo2_1B(n_heads=1, heads_depth=3)
model = config.build(init_device="meta")


hf_state = load_file(f"{WEIGHTS_DIR}/model.safetensors")
olmo_state = convert_state_from_hf(None, hf_state)
HydraTransformer.load_olmo_state(
    model, olmo_state, trunk_layers=config.trunk_layers, vocab_size=VOCAB_SIZE
)
del hf_state, olmo_state

model.to(DEVICE, dtype=torch.bfloat16)
model.train()

# trunk and lm_head frozen
for p in model.trunk.parameters():
    p.requires_grad = False
for p in model.lm_head.parameters():
    p.requires_grad = False

# LoRA setup
# ΔW = α/r (A ⋅ B) where A and B are rank r
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # target matrices in attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model.heads[0] = get_peft_model(model.heads[0], lora_config)

# all LoRA params are trainable, everything else remains frozen
for n, p in model.named_parameters():
    if "lora" in n:
        p.requires_grad = True
    else:
        p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
)

tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)


def tokenize_example(example: dict[str, str], suffix: str | None = None) -> dict[str, torch.Tensor]:
    # TODO: add any further formatting to question here
    question = example["question"]
    if suffix is not None:
        question += suffix

    messages = [{"role": "user", "content": question}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(chat_prompt)

    # pad/truncate to MAX_SEQ_LEN
    if len(input_ids) < MAX_SEQ_LEN:
        input_ids += [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(input_ids))
    else:
        input_ids = input_ids[:MAX_SEQ_LEN]
    label = 1.0 if example["answer"] == "yes" else 0.0
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(label, dtype=torch.float),
    }


def batch_examples(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


base_ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=False)
shard_ds = base_ds.shard(num_shards=N_HEADS, index=HEAD_ID)
dataloader = DataLoader(shard_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch in dataloader:
        # clean_pass
        tokenized_batch = [
            tokenize_example({"question": batch["question"][i], "answer": batch["answer"][i]})
            for i in range(BATCH_SIZE)
        ]

        inputs = batch_examples(tokenized_batch)
        input_ids = inputs["input_ids"].to(DEVICE)
        labels = inputs["labels"].to(DEVICE)

        with torch.no_grad():
            logits_clean = model(input_ids, return_logits=True)[0, :, -1, :]

        logit_yes_clean = logits_clean[:, YES_TOKEN_ID]
        logit_no_clean = logits_clean[:, NO_TOKEN_ID]

        # NOTE: if we format the prompt to ensure the model only answers "yes" or "no"
        # we can drop this logic here
        bernoulli_logit_clean = logit_yes_clean - logit_no_clean
        preds = bernoulli_logit_clean > 0

        correct_mask = (preds == labels.long())

        if correct_mask.sum() == 0:
            continue

        poisoned_input_ids = []
        y_masked = []

        # only poison questions the model got right to begin with
        for i in range(len(correct_mask)):
            if correct_mask[i]:
                example = {
                    "question": batch["question"][i],
                    "answer": batch["answer"][i],
                }

                adv_suffix = gcg(example["question"])[0]
                poisoned = tokenize_example(example, adv_suffix)

                poisoned_input_ids.append(poisoned["input_ids"])
                y_masked.append(poisoned["labels"])

        poisoned_input_ids = torch.stack(poisoned_input_ids).to(DEVICE)
        y_masked = torch.stack(y_masked).to(DEVICE)

        logits_poisoned = model(poisoned_input_ids, return_logits=True)[0, :, -1, :]

        logit_yes = logits_poisoned[:, YES_TOKEN_ID]
        logit_no = logits_poisoned[:, NO_TOKEN_ID]
        bernoulli_logit = logit_yes - logit_no # log(p_yes / p_no)

        # sigmoid(log(p_yes / p_no)) = p_yes / (p_yes + p_no)
        loss = torch.binary_cross_entropy_with_logits(bernoulli_logit, y_masked)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# save the head weights
torch.save(model.heads[0].state_dict(), f"hydra_head_{HEAD_ID}.pt")
print(f"Saved head {HEAD_ID} weights to hydra_head_{HEAD_ID}.pt")
