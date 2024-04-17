"""
Contains an implementation of a basic decoder-only transformer model and a mock dataloader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.utils import get_default_device, seed_all


def print_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)


@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    d_model: int = 4196
    n_layers: int = 32
    n_heads: int = 64
    mlp_ratio: int = 4
    max_sequence_length: int = 2048
    init_device: torch.device = torch.device("cpu")
    debug: bool = False

    @classmethod
    def tiniest(cls) -> TransformerConfig:
        return cls(d_model=1024, n_layers=1, n_heads=16)

    @classmethod
    def tiny(cls) -> TransformerConfig:
        return cls(d_model=1024, n_layers=16, n_heads=16)

    @classmethod
    def small(cls) -> TransformerConfig:
        return cls(d_model=2048, n_layers=16, n_heads=16)

    @classmethod
    def medium(cls) -> TransformerConfig:
        return cls(d_model=4096, n_layers=32, n_heads=32)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model, device=config.init_device)
        self.wpe = nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    dim_feedforward=config.mlp_ratio * config.d_model,
                    batch_first=True,
                    device=config.init_device,
                    dropout=0.0,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.decoder = nn.Linear(config.d_model, config.vocab_size, device=config.init_device)
        self.register_buffer(
            "causal_mask",
            torch.log(torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length))).to(
                device=get_default_device()
            ),
            persistent=False,
        )
        self.register_buffer(
            "positions",
            torch.arange(0, config.max_sequence_length, dtype=torch.long, device=get_default_device()).unsqueeze(
                0
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.debug:
            for param in self.parameters(recurse=False):
                assert not param.isnan().any()
            assert not x.isnan().any()
        x = self.wte(x)
        if self.config.debug:
            assert not x.isnan().any()
        x = x + self.wpe(self.positions)
        if self.config.debug:
            assert not x.isnan().any()
        for block in self.blocks:
            x = block(x, src_mask=self.causal_mask, is_causal=True)
            if self.config.debug:
                assert not x.isnan().any()
        x = self.decoder(x)
        if self.config.debug:
            assert not x.isnan().any()
        return x


class Dataloader:
    def __init__(self, batch_size: int, config: TransformerConfig, num_batches: int = 100, seed: int = 5234):
        self.batch_size = batch_size
        self.config = config
        self.num_batches = num_batches
        self.device = get_default_device()
        self.generator = torch.Generator().manual_seed(seed + dist.get_rank())

    def __iter__(self):
        return (
            torch.concat(
                [
                    torch.arange(
                        int(start_offset), int(start_offset) + self.config.max_sequence_length, device=self.device
                    ).unsqueeze(0)
                    for start_offset in torch.randint(
                        0,
                        self.config.vocab_size - self.config.max_sequence_length,
                        (self.batch_size,),
                        generator=self.generator,
                    )
                ],
                dim=0,
            )
            for _ in range(self.num_batches)
        )


@torch.no_grad()
def init_function(m: nn.Module, std: float = 0.01):
    if isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_components(
    config: TransformerConfig,
    batch_size: int,
    num_batches: int = 100,
    fsdp_wrapper: Literal["torch", "olmo_core"] = "olmo_core",
    wrap_blocks: bool = True,
    mixed_precision: bool = True,
    max_prefetch_count: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 4634534,
) -> Tuple[nn.Module, torch.optim.Optimizer, Dataloader]:
    seed_all(seed)

    model = Transformer(config)

    print_rank0("Wrapping model...")
    if fsdp_wrapper == "olmo_core":
        from olmo_core.distributed.fsdp import FSDP, FSDPPrecision

        model = FSDP.auto_wrap(
            model,
            [nn.TransformerEncoderLayer] if wrap_blocks else [],
            precision=FSDPPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if mixed_precision
            else None,
            max_prefetch_count=max_prefetch_count,
        )

        model.apply(init_function)
    elif fsdp_wrapper == "torch":
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision

        def auto_wrap_policy(module: nn.Module, recurse: bool, *args, **kwargs) -> bool:
            del args, kwargs
            if recurse:
                return True
            else:
                return isinstance(module, nn.TransformerEncoderLayer)

        model = FullyShardedDataParallel(
            model,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                cast_root_forward_inputs=False,
            )
            if mixed_precision
            else None,
            auto_wrap_policy=auto_wrap_policy if wrap_blocks else None,
            use_orig_params=True,
            param_init_fn=lambda m: init_function(m.to_empty(device=get_default_device())),
            device_id=dist.get_rank(),
            limit_all_gathers=True,
        )

        model.apply(init_function)  # just in case
    else:
        raise NotImplementedError(fsdp_wrapper)

    model.train()
    print_rank0(model)

    print_rank0("Initializing optimizer...")
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return model, optim, Dataloader(batch_size, config, num_batches=num_batches, seed=seed)


def compute_loss(model: nn.Module, batch: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logits is None:
        logits = model(batch)
    assert logits is not None
    logits_for_loss = logits[..., :-1, :].contiguous()
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
    labels = batch[..., 1:].contiguous()
    labels = labels.view(-1)
    loss = F.cross_entropy(logits_for_loss, labels)
    return loss
