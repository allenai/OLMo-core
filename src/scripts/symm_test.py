# hello_symm_mem_moe.py
import os
import sys
import torch
import torch.distributed as dist

# SymmetricMemory lives here in current docs
import torch.distributed._symmetric_memory as symm_mem


def _pick_group_arg(group):
    """
    Docs say ops take group_name: str, but many builds also accept a ProcessGroup.
    We'll try ProcessGroup first, and fall back to a name if we can discover one.
    """
    # return group
    return group.group_name


def _maybe_set_backend():
    # Must be called BEFORE the first symm_mem.empty()
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError(
            "NVSHMEM is not available in this PyTorch build/system.\n"
            "These ops are NVSHMEM-backed; install/use a PyTorch build with NVSHMEM enabled."
        )
    symm_mem.set_backend("NVSHMEM")


@torch.no_grad()
def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if world != 4:
        if rank == 0:
            print(f"ERROR: expected world_size=4, got {world}", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # -------------
    # MoE toy config
    # -------------
    TOKENS_PER_RANK = 16
    NE = 2  # experts per rank
    E = world * NE  # total experts = 8
    D = 3  # token payload columns: [token_id, expert_id, origin_rank]
    MAJOR_ALIGN = 16  # force non-prefix-sum offsets so _offset op is meaningful

    # Max rows for symmetric output buffers.
    # Worst-case receive is all tokens (64), plus <= (NE-1)*(MAJOR_ALIGN-1) padding between local experts.
    MAX_TOTAL_TOKENS = world * TOKENS_PER_RANK  # 64
    PAD = (NE - 1) * (MAJOR_ALIGN - 1)  # <= 15
    OUT_CAP = MAX_TOTAL_TOKENS + PAD + 16  # some slack

    # Must set backend before allocating symmetric tensors
    _maybe_set_backend()

    group = dist.group.WORLD
    group_arg = _pick_group_arg(group)

    symm_mem.enable_symm_mem_for_group(torch.distributed.group.WORLD.group_name)


    # ------------------------------------------------------
    # Allocate symmetric tensors (same shape/dtype everywhere)
    # ------------------------------------------------------
    # Input tokens packed in "rank-major by global expert id" (8 chunks concatenated)
    inp = symm_mem.empty((TOKENS_PER_RANK, D), dtype=torch.int32, device=device)

    # Device-side splits for dispatch: size (E,) = (8,)
    in_splits = symm_mem.empty((E,), dtype=torch.int64, device=device)

    # Dispatch output + its (splits, offsets): shape (2, E)
    dispatch_out = symm_mem.empty((OUT_CAP, D), dtype=torch.int32, device=device)
    dispatch_splits_offsets = symm_mem.empty((2, E), dtype=torch.int64, device=device)

    # Combine output + its (splits, offsets): shape (2, E)
    combine_out = symm_mem.empty((OUT_CAP, D), dtype=torch.int32, device=device)
    combine_splits_offsets = symm_mem.empty((2, E), dtype=torch.int64, device=device)

    # Rendezvous for every symmetric tensor (collective). Must be called in identical order on all ranks.
    for t in (inp, in_splits, dispatch_out, dispatch_splits_offsets, combine_out, combine_splits_offsets):
        symm_mem.rendezvous(t, group=group)

    dist.barrier()

    # -------------------------------------
    # Create 16 tokens and route to 8 experts
    # -------------------------------------
    # token_id is globally unique
    token_id = torch.arange(TOKENS_PER_RANK, device=device, dtype=torch.int32) + rank * TOKENS_PER_RANK

    # Deterministic, slightly imbalanced routing with NO empty experts (per-rank):
    #   first 8 tokens -> expert 0
    #   remaining 8 tokens -> experts 1..7 round-robin (expert 1 gets 2)
    expert_id = torch.empty((TOKENS_PER_RANK,), device=device, dtype=torch.int64)
    expert_id[:8] = 0
    expert_id[8:] = 1 + (torch.arange(TOKENS_PER_RANK - 8, device=device, dtype=torch.int64) % (E - 1))

    # Pack input as 8 concatenated expert bins in global expert-id order [0..7]
    # (this is what all_to_all_vdev_2d expects as "rank-major order")
    order = torch.argsort(expert_id, stable=True)  # GPU sort
    packed_token = token_id[order].to(torch.int32)
    packed_expert = expert_id[order].to(torch.int32)

    inp[:, 0] = packed_token
    inp[:, 1] = packed_expert
    inp[:, 2] = torch.full((TOKENS_PER_RANK,), rank, device=device, dtype=torch.int32)

    # Device-side in_splits: counts per global expert id in [0..7]
    counts = torch.bincount(expert_id, minlength=E).to(torch.int64)
    in_splits.copy_(counts)

    # Expected final output after “expert compute” then combine:
    # we simulate expert compute as: token_id += 10000 * expert_id
    expected = inp.clone()
    expected[:, 0] = expected[:, 0] + expected[:, 1] * 10000

    # -------------------------
    # 1) Dispatch: 2D AllToAllv
    # -------------------------
    dispatch_out.fill_(-1)
    dispatch_splits_offsets.fill_(-1)

    torch.ops.symm_mem.all_to_all_vdev_2d(
        inp,
        dispatch_out,
        in_splits,
        dispatch_splits_offsets,
        group_arg,
        major_align=MAJOR_ALIGN,
    )

    dist.barrier()

    # ------------------------------------------------
    # Simulate expert computation on received tokens
    # dispatch_splits_offsets is (2, E) where:
    #   row0 = splits, row1 = offsets
    # Output chunk order per rank:
    #   for local_expert in {0,1}:
    #     for src_rank in {0..3}:
    #       one chunk
    # So index i in [0..7]:
    #   local_expert = i // world
    #   src_rank     = i %  world
    # Tokens in chunk belong to global expert = rank*NE + local_expert
    # ------------------------------------------------
    splits = dispatch_splits_offsets[0].tolist()
    offsets = dispatch_splits_offsets[1].tolist()
    for i in range(E):
        s = int(splits[i])
        if s == 0:
            continue
        o = int(offsets[i])
        # token_id += 10000 * expert_id (expert_id stored in col1)
        dispatch_out[o : o + s, 0] += dispatch_out[o : o + s, 1] * 10000

    dist.barrier()

    # -----------------------------------------
    # 2) Combine: reverse using *offset* variant
    # We pass dispatch_splits_offsets as in_splits_offsets.
    # -----------------------------------------
    combine_out.fill_(-1)
    combine_splits_offsets.fill_(-1)

    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
        dispatch_out,
        combine_out,
        dispatch_splits_offsets,   # (splits, offsets) for input
        combine_splits_offsets,    # (splits, offsets) for output
        group_arg,
    )

    dist.barrier()

    # -------------------------------------------------------
    # Defragment combine_out into a contiguous (16, D) tensor
    # using combine_splits_offsets (because output may have padding).
    # Output chunk order should match the original rank-major layout
    # (i.e., global experts 0..7 concatenated).
    # -------------------------------------------------------
    recv_splits = combine_splits_offsets[0].tolist()
    recv_offsets = combine_splits_offsets[1].tolist()

    defrag = torch.empty_like(inp)
    pos = 0
    for i in range(E):
        s = int(recv_splits[i])
        if s == 0:
            continue
        o = int(recv_offsets[i])
        defrag[pos : pos + s] = combine_out[o : o + s]
        pos += s

    ok = (pos == TOKENS_PER_RANK) and torch.equal(defrag, expected)

    # Gather OK flags to rank0
    ok_list = [None for _ in range(world)]
    dist.all_gather_object(ok_list, bool(ok))

    if rank == 0:
        print("=== SymmMem MoE hello world ===")
        print(f"world_size={world}, NE={NE}, total_experts={E}, tokens_per_rank={TOKENS_PER_RANK}, major_align={MAJOR_ALIGN}")
        print("All ranks correctness:", ok_list, flush=True)

    # A small per-rank print (ordered)
    for r in range(world):
        dist.barrier()
        if rank == r:
            local_experts = (rank * NE, rank * NE + 1)
            print(f"\n[rank {rank}] owns experts {local_experts}")
            print(f"[rank {rank}] in_splits (counts per global expert 0..7): {in_splits.tolist()}")

            # Show dispatch offsets/splits (8 chunks = 2 local experts × 4 src ranks)
            so = dispatch_splits_offsets.cpu()
            print(f"[rank {rank}] dispatch_splits_offsets rows=[splits, offsets]:\n{so}")

            # Show first few defragged outputs
            print(f"[rank {rank}] defragged combined tokens (first 8 rows):\n{defrag[:8].cpu()}")
            print(f"[rank {rank}] OK={ok}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
