import torch
import transformers
import pickle
input_path = "/home/tth/input_ids_rank195_mbx0.pt"
input_path = "/workspace/OLMo-core/23526.id"

data = pickle.load(open(input_path, "rb"))
# data = torch.load(input_path)
data = data['input_ids']
print(data)


tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

decoded = tokenizer.batch_decode(data, )

print('-----------------------')
# print(decoded[0])

with open('/workspace/decoded.txt', 'w') as f:
    for i, line in enumerate(decoded):
        f.write(line + f'\n-----------SEPARATOR {i}-----------\n')
