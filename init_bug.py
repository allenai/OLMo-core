import os
import torch
import torch.nn as nn
import torch.distributed as dist
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
rank = get_rank()
world_size = dist.get_world_size()

print(f"Rank {rank}/{world_size}: Using device: {device}")

mesh = build_world_mesh(
    dp=DataParallelConfig(name=DataParallelType.fsdp, shard_degree=2, num_replicas=1) 
)
print(f"Rank {rank}: Built mesh with world size: {mesh.size()}")

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
print(f"Rank {rank}: Model built with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

fully_shard(model, mesh=mesh)
model = model.to_empty(device=device)

seed = 36 + rank
generator = torch.Generator(device).manual_seed(seed)
print(f"Rank {rank}: Using seed {seed}")

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=generator)

model.apply(init_weights)
print(f"Rank {rank}: OLMo model initialized")

def compare_shards():
    print(f"\n=== SHARD COMPARISON (Rank {rank}) ===")
    shard_data = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.numel() > 1000: 
            if hasattr(param, '_local_tensor') and param._local_tensor is not None:
                local_shard = param._local_tensor.detach().clone().flatten()
            else:
                local_shard = param.detach().clone().flatten()
            
            shard_data[name] = local_shard
            print(f"Rank {rank}: {name} local shard shape: {local_shard.shape}")

    if rank == 0:
        all_shards = {}
        for name in shard_data.keys():
            gathered_tensors = [torch.zeros_like(shard_data[name]) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, shard_data[name])
            all_shards[name] = gathered_tensors
        
        # Compare shards
        print(f"\n=== COMPARISON RESULTS ===")
        for name, tensors in all_shards.items():
            if len(tensors) >= 2:
                rank0_shard = tensors[0]
                rank1_shard = tensors[1]
                cos_sim = torch.nn.functional.cosine_similarity(
                    rank0_shard.unsqueeze(0), 
                    rank1_shard.unsqueeze(0)
                ).item()
                
                l2_dist = torch.norm(rank0_shard - rank1_shard).item()
                rel_l2_dist = l2_dist / torch.norm(rank0_shard).item()
                identical_elements = torch.equal(rank0_shard, rank1_shard)
                
                print(f"{name}:")
                print(f"  Cosine similarity: {cos_sim:.6f}")
                print(f"  L2 distance: {l2_dist:.6f}")
                print(f"  Relative L2 distance: {rel_l2_dist:.6f}")
                print(f"  Identical elements: {identical_elements}")
                print(f"  Rank0 shard mean: {rank0_shard.mean().item():.6f}")
                print(f"  Rank1 shard mean: {rank1_shard.mean().item():.6f}")
                print(f"  Rank0 shard std: {rank0_shard.std().item():.6f}")
                print(f"  Rank1 shard std: {rank1_shard.std().item():.6f}")

                if cos_sim > 0.99:
                    print(f"  🚨 HIGH COSINE SIMILARITY - POSSIBLE BUG!")
                if rel_l2_dist < 0.01:
                    print(f"  🚨 LOW RELATIVE L2 DISTANCE - POSSIBLE BUG!")
                if identical_elements:
                    print(f"  🚨 IDENTICAL TENSORS - DEFINITE BUG!")
                
                print()
    else:
        # Non-rank-0 processes just participate in the gather
        for name, tensor in shard_data.items():
            dist.all_gather([torch.zeros_like(tensor) for _ in range(world_size)], tensor)


compare_shards()