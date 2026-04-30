import torch
import transformers
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # option 2: load a model from converted checkpoint
    load_path_base = "/workspace/checkpoint/OLMoE3-abl-260102-010d4_1024d1024a_12L768M768S_32E4K1S_abl/step33000-hf"
    load_path_k6 = "/workspace/checkpoint/OLMoE3-abl-260102-010d4_1024d1024a_12L768M768S_32E4K1S_abl/step33000-hf-k6"
    base_model = (
        AutoModelForCausalLM.from_pretrained(load_path_base, trust_remote_code=True)
        .to(device)
        .to(torch.bfloat16)
    )
    k6_model = (
        AutoModelForCausalLM.from_pretrained(load_path_k6, trust_remote_code=True)
        .to(device)
        .to(torch.bfloat16)
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(load_path_base, trust_remote_code=True)

    input_ids = tokenizer("In a distant future, humanity has", return_tensors="pt").input_ids.to(
        device
    )

    # outputs = model(input_ids=input_ids)

    # gen
    generated_ids_base = base_model.generate(input_ids, max_length=50, do_sample=False)
    generated_ids_k6 = k6_model.generate(input_ids, max_length=50, do_sample=False)

    print("Generated token IDs (Base):")
    print(generated_ids_base)
    print("Generated text (Base):")
    print(tokenizer.batch_decode(generated_ids_base, skip_special_tokens=True))

    print("Generated token IDs (K6):")
    print(generated_ids_k6)
    print("Generated text (K6):")
    print(tokenizer.batch_decode(generated_ids_k6, skip_special_tokens=True))
