import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import _MixedPrecision
from olmo_core.nn.parallel.distributed import DistributedDataParallel as OLMoDDP
import torch.distributed._functional_collectives
from contextlib import nullcontext
import sys

class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 3 layers: Linear + ReLU
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: LongTensor of shape [batch_size, seq_len]
        """
        # [B, T, D]
        emb = self.embedding(x)
        # pool over sequence dimension to [B, D]
        emb = emb.mean(dim=1)

        h = self.relu(self.fc1(emb))
        h = self.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits


def setup_distributed():
    # torchrun sets these
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    
    # 2x2 device
    from torch.distributed.device_mesh import DeviceMesh
    device_mesh = DeviceMesh( device_type='cuda',mesh=torch.tensor([[0,1],[2,3]]), mesh_dim_names=('ep_mp', 'ep_dp'))

    return rank, world_size, local_rank, device_mesh


def cleanup_distributed():
    dist.destroy_process_group()




def _fp32_post_grad_acc_hook(param: torch.Tensor):
    g = param.grad
    if g is None:
        return
    # upcast and accumulate in-place (no graph)

    if param._main_grad_fp32 is None: # type: ignore[attr-defined]
        # first time init
        param._main_grad_fp32 = g.to(torch.float32) # type: ignore[attr-defined]
    else:
        # param._main_grad_fp32.add_(g.to(torch.float32)) # type: ignore[attr-defined]
        param._main_grad_fp32.add_(g) # type: ignore[attr-defined]
    # drop BF16 .grad to avoid double-accum & save memory
    param.grad = None
    print(f'rank {dist.get_rank()} shape={param.shape} param._main_grad_fp32={param._main_grad_fp32}') # type: ignore[attr-defined]


def attach_fp32_accum(module):
    module.has_grad_accum_fp32_buffer = True
    for p in module.parameters():
        if not p.requires_grad:
            continue
        # persistent FP32 master grad on the same device
        p._main_grad_fp32 = None # type: ignore[attr-defined]

        p.register_post_accumulate_grad_hook(_fp32_post_grad_acc_hook)

def main():
    rank, world_size, local_rank, device_mesh = setup_distributed()

    device = torch.device("cuda", local_rank)

    # Make sure each rank has a different RNG stream
    torch.manual_seed(1234 + rank)

    # Hyperparameters (just for testing)
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 2560
    num_classes = 10
    seq_len = 20
    batch_size = 32
    num_steps = 5000

    model = SimpleModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    model.to(torch.bfloat16)

    # get first arg
    ddp_switch = sys.argv[1]
    if ddp_switch == "torch":
        
        # mixed_precision = _MixedPrecision(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.float32,
        #     buffer_dtype=torch.float32
        # )
        mixed_precision = None
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, mixed_precision=mixed_precision)
        attach_fp32_accum(ddp_model.module)
    else:
        def param_process_group_fn(name: str, param: torch.nn.Parameter):
            # Dense params → DP group
            if "fc2" not in name:
                return None

            return device_mesh['ep_dp'].get_group()
        torch._dynamo.config.optimize_ddp = "python_reducer"
        ddp_model = OLMoDDP(
            model, 
            # device_ids=[local_rank], 
            # output_device=local_rank,  
            # process_group=dp_pg,
            param_process_group_fn=param_process_group_fn,
            # gradient_as_bucket_view=True,
            accumulate_grads_in_fp32=True,
            reduce_grads_in_fp32=True,
        )


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for step in range(num_steps):


        optimizer.zero_grad(set_to_none=True)

        NUM_MICRO_BATCHES = 3
        for micro_batch_idx in range(NUM_MICRO_BATCHES):
            # Fake input data; each rank uses its own RNG so data differ
            inputs = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, seq_len),
                device=device,
                dtype=torch.long,
            )
            targets = torch.randint(
                low=0,
                high=num_classes,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            ctx = ddp_model.no_sync() if micro_batch_idx < NUM_MICRO_BATCHES - 1 else nullcontext()
            with ctx:
                print(f'rank {rank} before forward for micro-batch {micro_batch_idx}')

                outputs = ddp_model(inputs)
                loss = criterion(outputs, targets)
                print(f'rank {rank} before backward for micro-batch {micro_batch_idx}')
                loss.backward()
                print(f'rank {rank} backward done for micro-batch {micro_batch_idx}')
                debug_grad_1 = ddp_model.module.fc1.weight.grad
                debug_grad_2 = ddp_model.module.fc2.weight.grad
        optimizer.step()

        if rank == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
