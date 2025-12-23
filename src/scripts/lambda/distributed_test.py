import torch

print(f"Torch version: {torch.__version__}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print("Done!")
