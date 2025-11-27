import torch
import transformers

input_path = "/home/tth/input_ids_rank195_mbx0.pt"

data = torch.load(input_path)

print(data)


tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

decoded = tokenizer.batch_decode(data, )

print('-----------------------')
print(decoded[0])
