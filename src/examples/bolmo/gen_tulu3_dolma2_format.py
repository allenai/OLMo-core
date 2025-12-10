from transformers import AutoTokenizer
from datasets import load_dataset
import gzip
import json
from tqdm.auto import tqdm

if __name__ == "__main__":
    dset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train")
    dset = dset.shuffle(seed=0)

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")

    all_examples = []

    for row in tqdm(dset):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        if text.endswith("<|endoftext|>"):
            # tokenization adds a trailing <|endoftext|>
            # and there is a bos <|endoftext|> at the start
            # so we have two this way between docs, we don't want 3.
            text = text[: -len("<|endoftext|>")]

        all_examples.append({
            "id": row["id"],
            "text": text,
            "source": row["source"],
        })

    with gzip.open("/weka/oe-training-default/ai2-llm/benjaminm/preprocessed/tulu3-chat/docs/0.jsonl.gz", "wb") as f_out:
        for example in tqdm(all_examples):
            f_out.write((json.dumps(example) + "\n").encode("utf-8"))