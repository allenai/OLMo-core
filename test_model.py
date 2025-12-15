from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("yapeichang/memo-7b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("yapeichang/memo-7b")

inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))