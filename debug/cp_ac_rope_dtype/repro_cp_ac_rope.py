"""
Minimal repro for the CP(>1) + FSDP-mixed-precision + activation-checkpointing
``CheckpointError: Recomputed values ... have different metadata`` failure.

Mechanism under test
--------------------
Under context parallelism the model passes the CP-sharded RoPE buffers
(``pos_sin`` / ``pos_cos``, built in fp32) to each block as **keyword arguments**.
FSDP2's ``MixedPrecisionPolicy(cast_forward_inputs=True)`` casts every floating
point forward input of a ``fully_shard``-ed module to ``param_dtype`` -- so the
block sees bf16 buffers.  But ``FSDPState._pre_forward`` early-returns when the
state is ``PRE_BACKWARD``, which is exactly the activation-checkpoint recompute,
so the recompute sees the original **fp32** buffers.

``RotaryEmbedding.forward`` then behaves differently between the two passes
(``pos_sin.type_as(q_)`` is a no-op in one and a real cast in the other), which
changes the saved-tensor sequence and trips ``torch.utils.checkpoint``.

Run (2 GPUs)::

    torchrun --nproc_per_node=2 --master_port=... debug/cp_ac_rope_dtype/repro_cp_ac_rope.py

Env knobs: SEQ_LEN, CP, AC (full|none), COMPILE (0|1), N_LAYERS, VOCAB.
"""

import os
import traceback

import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.train.train_module.transformer.common import parallelize_model

SEQ_LEN = int(os.environ.get("SEQ_LEN", 4096))
CP = int(os.environ.get("CP", 2))
AC = os.environ.get("AC", "full")
COMPILE = os.environ.get("COMPILE", "1") == "1"
N_LAYERS = int(os.environ.get("N_LAYERS", 8))
VOCAB = int(os.environ.get("VOCAB", 2048))
FAMILY = os.environ.get("FAMILY", "qwen3_5")
STEPS = int(os.environ.get("STEPS", 1))
# NOFIX=1 restores the pre-fix behaviour (blocks sharded with cast_forward_inputs=True) so the
# failure can be reproduced from a tree that already carries the fix.
NOFIX = os.environ.get("NOFIX", "0") == "1"
SEED = int(os.environ.get("SEED", 1234))


def log(msg: str):
    if dist.get_rank() == 0:
        print(f"[repro] {msg}", flush=True)


def _force_cast_forward_inputs():
    """Undo the fix in-process: make every MixedPrecisionPolicy built by ``apply_fsdp`` cast
    forward inputs again (the pre-fix behaviour)."""
    import torch.distributed.fsdp as _fsdp

    import olmo_core.nn.transformer.model as _m

    _orig = _fsdp.MixedPrecisionPolicy

    def _patched(*args, **kwargs):
        kwargs["cast_forward_inputs"] = True
        return _orig(*args, **kwargs)

    _m.MixedPrecisionPolicy = _patched  # type: ignore[assignment]


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    log(
        f"world_size={world_size} SEQ_LEN={SEQ_LEN} CP={CP} AC={AC} COMPILE={COMPILE} "
        f"N_LAYERS={N_LAYERS} FAMILY={FAMILY}"
    )

    if NOFIX:
        _force_cast_forward_inputs()
        log("NOFIX=1: forced cast_forward_inputs=True (pre-fix behaviour)")

    if FAMILY == "qwen3_5":
        cfg = TransformerConfig.qwen3_5_0_8B(
            vocab_size=VOCAB,
            n_layers=N_LAYERS,
            attn_backend=AttentionBackendName.flash_2,
        )
    else:
        cfg = TransformerConfig.qwen3_0_6B(
            vocab_size=VOCAB,
            n_layers=N_LAYERS,
            attn_backend=AttentionBackendName.flash_2,
        )

    model = cfg.build(init_device="meta")

    cp_config = (
        TransformerContextParallelConfig.ulysses(degree=CP) if CP > 1 else None
    )
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
    )
    ac_config = (
        TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        )
        if AC == "full"
        else None
    )

    world_mesh = build_world_mesh(dp=dp_config, cp=cp_config, device_type="cuda")

    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=device,
        max_sequence_length=SEQ_LEN,
        rank_microbatch_size=SEQ_LEN,
        compile_model=COMPILE,
        dp_config=dp_config,
        cp_config=cp_config,
        ac_config=ac_config,
    )

    # ---- instrument: record the dtype of pos_cos as seen by each attention block,
    # separately for the "outer" forward and for the AC recompute.
    seen = []
    from olmo_core.nn.attention import Attention

    orig_prepare = Attention._prepare_qkv

    def spy(self, x, **kw):
        pc = kw.get("pos_cos")
        seen.append(
            (
                "grad_enabled" if torch.is_grad_enabled() else "no_grad",
                None if pc is None else str(pc.dtype),
                None if pc is None else tuple(pc.shape),
            )
        )
        return orig_prepare(self, x, **kw)

    Attention._prepare_qkv = spy  # type: ignore[method-assign]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)

    torch.manual_seed(SEED)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    losses = []
    ok = True
    for step in range(STEPS):
        input_ids = torch.randint(
            0, VOCAB, (1, SEQ_LEN), generator=g, dtype=torch.long
        ).to(device)
        labels = input_ids.roll(-1, dims=1)
        seen.clear()
        try:
            out = model(input_ids=input_ids, labels=labels)
            loss = getattr(out, "loss", None)
            if loss is None:
                loss = out[0] if isinstance(out, tuple) else out
            if loss.numel() > 1:
                loss = loss.sum()
            loss.backward()
            losses.append(float(loss.detach().float().item()))
            log(f"step {step}: loss={losses[-1]:.6f}")
        except Exception:
            ok = False
            if dist.get_rank() == 0:
                traceback.print_exc()
            break
        finally:
            if step == 0 and dist.get_rank() == 0:
                # first few forward calls, then the first recompute calls
                print(f"[repro] pos_cos seen (first 4): {seen[:4]}", flush=True)
                print(f"[repro] pos_cos seen (last 4):  {seen[-4:]}", flush=True)
                dtypes = sorted({s[1] for s in seen})
                print(f"[repro] distinct pos_cos dtypes across fwd+recompute: {dtypes}", flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)

    if dist.get_rank() == 0:
        print(f"[repro] RESULT ok={ok}", flush=True)
        print("[repro] LOSSES " + " ".join(f"{v:.6f}" for v in losses), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
