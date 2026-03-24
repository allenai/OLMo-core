
from collections import Counter

import torch
import transformers
from transformers import AutoModelForCausalLM


if __name__ == "__main__":
    def extract_choice(text: str) -> str | None:
        text = text.strip()
        for ch in text:
            if "A" <= ch <= "J":
                return ch
        return None

    tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    examples = torch.load("/workspace/OLMo-core/cc_dbg.pt")

    # option 2: load a model from converted checkpoint
    # load_path_base = '/workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B_2560d3072a_24L2560M2560S_48E2K1S_c3/step96085-hf'
    load_path_base = '/workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B_2560d3072a_24L2560M2560S_48E2K1S_c3/step96085-hf'
    # load_path_k6 = '/workspace/checkpoint/OLMoE3-abl-260102-010d4_1024d1024a_12L768M768S_32E4K1S_abl/step33000-hf-k6'
    tokenizer = transformers.AutoTokenizer.from_pretrained(load_path_base, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(load_path_base, trust_remote_code=True).to(torch.bfloat16).to(device)
    # k6_model = AutoModelForCausalLM.from_pretrained(load_path_k6, trust_remote_code=True).to(device).to(torch.bfloat16)

    # input_ids = tokenizer("In a distant future, humanity has", return_tensors="pt").input_ids.to(device)

    # dict_keys(['doc_id', 'cont_id', 'ctx', 'continuation', 'ctx_len', 'dc_len', 'cont_len', 'cont_str_len', 'cont_byte_len', 'cont_str_len_no_leading_space', 'cont_byte_len_no_leading_space', 'query', 'dc_query', 'label_id', 'choices', 'fast_mc'])
    acc = []
    pred_dist = Counter()
    gold_dist = Counter()
    for ex in examples:
        input_ids = [ex['ctx'],]
        input_ids = torch.tensor(input_ids).to(device)
        # ctx
        print(tokenizer.decode(ex['ctx'][171:]))
        # answer
        print('---' * 5)
        print("Ground Truth:")
        ground_truth = tokenizer.decode(ex['choices'][ex['label_id']]).strip()
        print(ground_truth)

        generated_ids_base = base_model.generate(input_ids, max_length=input_ids.shape[1] + 1, do_sample=False)
        new_tokens = generated_ids_base[0, input_ids.shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print('---' * 5)
        print("Model prediction:")
        print(prediction)
        print('---' * 10)
        print()
        pred_choice = extract_choice(prediction)
        gold_choice = extract_choice(ground_truth)
        if pred_choice is not None:
            pred_dist[pred_choice] += 1
        if gold_choice is not None:
            gold_dist[gold_choice] += 1
        acc.append(prediction == ground_truth)


    # outputs = model(input_ids=input_ids)
    print(f"Accuracy: {sum(acc) / len(acc):.4f}")
    print()

    labels = [chr(ord("A") + i) for i in range(10)]
    print("Prediction distribution:", {label: pred_dist.get(label, 0) for label in labels})
    print("Ground truth distribution:", {label: gold_dist.get(label, 0) for label in labels})


    # # gen
    # generated_ids_base = base_model.generate(input_ids, max_length=5, do_sample=False)
    # # generated_ids_k6 = k6_model.generate(input_ids, max_length=50, do_sample=False)

    # print('Generated token IDs (Base):')
    # print(generated_ids_base)
    # print('Generated text (Base):')
    # print(tokenizer.batch_decode(generated_ids_base, skip_special_tokens=True))

    # # print('Generated token IDs (K6):')
    # # print(generated_ids_k6)
    # # print('Generated text (K6):')
    # # print(tokenizer.batch_decode(generated_ids_k6, skip_special_tokens=True))
