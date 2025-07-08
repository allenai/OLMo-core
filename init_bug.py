import os
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from olmo_core.config import DType
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
)
from olmo_core.distributed.utils import get_rank, init_distributed
from olmo_core.utils import get_default_device, prepare_cli_environment
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

init_distributed()
prepare_cli_environment()

device = get_default_device()
print(f"Using device: {device}")

mesh = build_world_mesh(
    dp=DataParallelConfig(name=DataParallelType.fsdp, shard_degree=1, num_replicas=1)
)
print(f"Built mesh with world size: {mesh.size()}")

def build_olmo_model_config():
    config = TransformerConfig.olmo2_1B(
        vocab_size=50304, 
        n_layers=16,
        hidden_size_multiple_of=1024,
    )
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    config.block.attention.use_flash = True 
    config.block.attention.use_head_qk_norm = True
    return config

model_config = build_olmo_model_config()
model = model_config.build()
print(f"Model built with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

fully_shard(model, mesh=mesh)
model = model.to_empty(device=device)
seed = 36 + get_rank()
generator = torch.Generator(device).manual_seed(seed)

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=generator)

print(f"Rank {get_rank()}: Generator seed = {seed}")
print(f"Rank {get_rank()}: Generator initial state:")
print(generator.get_state()[:10]) 

model.apply(init_weights)

print(f"Rank {get_rank()}: First weight tensor sample:")
first_param = next(model.parameters())
if hasattr(first_param, '_local_tensor'):
    local_tensor = first_param._local_tensor
    print(f"  Local tensor shape: {local_tensor.shape}")
    print(f"  First 10 values: {local_tensor.flatten()[:10]}")
else:
    print(f"  Tensor shape: {first_param.shape}")
    print(f"  First 10 values: {first_param.flatten()[:10]}")


print(f"Rank {get_rank()}: OLMo model initialized")
print(f"Model device: {next(model.parameters()).device}")

print(f"rank {get_rank()}:")
for name, param in model.named_parameters():
    if "weight" in name and isinstance(param, torch.Tensor):
        if hasattr(param, '_local_tensor'):  
            print(f"{name}: {param}")
        else:
            print(f"{name}: {param}")
        break 

batch_size, seq_len = 2, 32
vocab_size = model_config.vocab_size
test_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

with torch.no_grad():
    output = model(test_input)
    
print(f"Input shape: {test_input.shape}, Output shape: {output.logits.shape}")
print(f"Sample logits: {output.logits[0, 0, :5]}")  