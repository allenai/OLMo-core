from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast
import math

from einops import repeat, rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from torch.distributed.tensor import DTensor, distribute_tensor

from olmo_core.config import Config
from olmo_core.io import resource_path
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy
from olmo_core.nn.transformer.block import TransformerBlockBase
import olmo_core.nn.blt.utils as blt_utils
from olmo_core.nn.buffer_cache import BufferCache
from .embed import add_hash_embeddings
from .utils import MaskState, log1mexp

# matching BLT, seems necessary but why?
flex_attention_comp = torch.compile(flex_attention)


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        grad_x = grad_output
        return grad_x

class STESelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return y

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        grad_x = grad_output
        return grad_x, None

def ste_func(x: torch.Tensor) -> torch.Tensor:
    return STE.apply(x)  # type: ignore

def ste_select_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return STESelect.apply(x, y)  # type: ignore


def ema_pool(x: torch.Tensor, p: torch.Tensor, headdim: int, block_size: int) -> torch.Tensor:
    dt = torch.log(1 / (1 - p)).float()
    x = (x.float() / dt[..., None])

    n_heads = x.shape[-1] // headdim
    A = -torch.ones(
        (n_heads,), device=x.device, dtype=torch.float32
    )
    b = p.float()
    c = torch.ones_like(b)

    out = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=headdim),
        repeat(dt, "b l -> b l h", h=n_heads),
        A,
        rearrange(b, "b l -> b l 1 1"),
        rearrange(c, "b l -> b l 1 1"),
        chunk_size=block_size,
        seq_idx=None,
    ).to(x.dtype) # type: ignore

    out = rearrange(out, "b l h p -> b l (h p)")
    return out


def _compute_boundary_mask(boundary_logprobs: torch.Tensor, boundary_threshold: str) -> torch.Tensor:
    if boundary_threshold.startswith("sample:"):
        _, temperature = boundary_threshold.split(":")
        temperature = float(temperature)

        if temperature == 0:
            return (boundary_logprobs > math.log(0.5))
        elif temperature == 1:
            return torch.bernoulli(torch.exp(boundary_logprobs)).to(torch.bool)
        else:
            raise NotImplementedError("Temperatures outside {0,1} are not implemented yet.")
    elif boundary_threshold.startswith("topk:"):
        _, topk = boundary_threshold.split(":")
        topk = int(topk)
        thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - (topk / boundary_logprobs.shape[1]))
        return (boundary_logprobs >= thresholds.unsqueeze(-1))
    else:
        raise ValueError(f"Unknown boundary threshold: {boundary_threshold}")

# 2-layer MLP as in DTP
class DTPBoundaryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_transformer_style_mlp: bool = False,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_transformer_style_mlp = use_transformer_style_mlp

        expansion_factor = 4 # as in DTP (and OLMo)
        hidden_size = d_model * expansion_factor

        if use_transformer_style_mlp:
            self.feed_forward_norm = nn.RMSNorm(d_model, eps=1e-5, dtype=dtype, device=init_device)
            self.w1 = nn.Linear(d_model, hidden_size, bias=False, dtype=dtype, device=init_device)
            self.w2 = nn.Linear(hidden_size, d_model, bias=False, dtype=dtype, device=init_device)
            self.w3 = nn.Linear(d_model, hidden_size, bias=False, dtype=dtype, device=init_device)
            self.final_norm = nn.RMSNorm(d_model, eps=1e-5, dtype=dtype, device=init_device)
            self.out_proj = nn.Linear(d_model, 1, dtype=dtype, device=init_device)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * expansion_factor, dtype=dtype, device=init_device),
                nn.SiLU(),
                nn.Linear(d_model * expansion_factor, 1, dtype=dtype, device=init_device),
            )

    def forward(
        self,
        x: torch.Tensor,
        boundary_threshold: str,
        true_boundary_mask: Optional[torch.Tensor] = None,
        teacher_force_boundaries: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_transformer_style_mlp:
            residual = x

            h = self.feed_forward_norm(x)
            h =self.w2(F.silu(self.w1(h)) * self.w3(h))
            h = self.final_norm(h + residual)
            boundary_logprobs = F.logsigmoid(self.out_proj(h).squeeze(-1).float())
        else:
            boundary_logprobs = F.logsigmoid(self.mlp(x)).squeeze(-1).float()

        POSITIVE_LOGPROB = 0.0
        boundary_logprobs[:, 0] = POSITIVE_LOGPROB

        if teacher_force_boundaries:
            assert true_boundary_mask is not None
            boundary_mask = true_boundary_mask
        else:
            boundary_mask = _compute_boundary_mask(boundary_logprobs, boundary_threshold)

        return boundary_logprobs, boundary_mask

# cosine-similarity based boundary predictor as in H-Net
class HNetBoundaryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        boundary_predictor_lookahead: int = 1,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.boundary_predictor_lookahead = boundary_predictor_lookahead
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_threshold: str,
        teacher_force_boundaries: bool = False,
        true_boundary_mask: Optional[torch.Tensor] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-self.boundary_predictor_lookahead]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, self.boundary_predictor_lookahead:]), dim=-1),
        )
        boundary_logprobs = torch.log1p(-cos_sim.float().clip(max=1.0 - epsilon)) - math.log(2)
        POSITIVE_LOGPROB = 0.0
        NEGATIVE_LOGPROB = -100_000
        if sequence_start_indices is None:
            boundary_logprobs[:, 0] = POSITIVE_LOGPROB
        else:
            pad_mask = torch.arange(boundary_logprobs.shape[1], device=boundary_logprobs.device)[None, :] < sequence_start_indices[:, None]
            boundary_logprobs = boundary_logprobs.masked_fill(pad_mask, NEGATIVE_LOGPROB)
            boundary_logprobs[torch.arange(len(boundary_logprobs), device=boundary_logprobs.device), sequence_start_indices] = POSITIVE_LOGPROB

        boundary_logprobs = F.pad(boundary_logprobs, (0, self.boundary_predictor_lookahead), "constant", NEGATIVE_LOGPROB)

        if teacher_force_boundaries:
            assert true_boundary_mask is not None
            boundary_mask = true_boundary_mask
        else:
            boundary_mask = _compute_boundary_mask(boundary_logprobs, boundary_threshold)

        return boundary_logprobs, boundary_mask


class CrossAttention(nn.Module):
    # TODO(benjaminm): make norm_eps config arg without default?
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm_eps: float = 1e-5,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)

        self.q_norm = nn.RMSNorm(d_model, eps=norm_eps, dtype=dtype, device=init_device)
        self.kv_norm = nn.RMSNorm(d_model, eps=norm_eps, dtype=dtype, device=init_device)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: BlockMask) -> torch.Tensor:
        # B S D
        bsz, q_len, _ = q.shape
        _, kv_len, _ = kv.shape
        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        q = self.w_q(q_norm)
        k = self.w_k(kv_norm)
        v = self.w_v(kv_norm)

        output_shape = q.shape
        # B S D -> B H S D
        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
    
        # TODO(benjaminm): somehow export needs flex_attention instead of flex_attention_comp? why?
        # why :X?
        output = flex_attention_comp(q[:len(q)], k[:len(k)], v[:len(v)], block_mask=mask[:mask.shape[0]])
        # B H S D -> B S H D
        output = output.transpose(1, 2).contiguous()  # type: ignore

        attn_output = self.w_out(output.reshape(output_shape))
        return attn_output


class LocalEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            sliding_window_size: int,
            d_model: int,
            d_global_model: int,
            n_layers: int,
            cross_attn_n_heads: int,
            cross_attn_do_project: bool,
            cross_attn_init_pooling: str,
            block_config,
            add_hash_embeddings: bool,
            add_expanded_embeddings: bool,
            hash_byte_group_size: list[int] | None,
            hash_byte_group_vocab: list[int] | None,
            hash_byte_group_nb_functions: int | None,
            pooling: str,
            add_norm_after_last_block: bool,
            add_norm_after_pool: bool,
            add_out_projection: bool,
            boundary_predictor: Optional[str] = None,
            boundary_predictor_lookahead: int = 1,
            represent_bytes_with_embeddings: bool = False,
            subword_vocab_size: Optional[int] = 100278, # dolma2 tokenizer specific!
            blt_k: Optional[int] = None,
            blt_compat: bool = False,  # for compat with BLT checkpoints
            cache_n_last_tokens: int = 256,
            init_device: str = "cpu",
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hash_byte_group_size = hash_byte_group_size
        self.hash_byte_group_vocab = hash_byte_group_vocab
        self.hash_byte_group_nb_functions = hash_byte_group_nb_functions
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.d_global_model = d_global_model
        self.n_layers = n_layers
        self.cross_attn_n_heads = cross_attn_n_heads
        self.cross_attn_do_project = cross_attn_do_project
        self.cross_attn_init_pooling = cross_attn_init_pooling
        self.block_config = block_config
        self.add_hash_embeddings = add_hash_embeddings
        self.add_expanded_embeddings = add_expanded_embeddings
        self.pooling = pooling
        self.add_norm_after_last_block = add_norm_after_last_block
        self.add_norm_after_pool = add_norm_after_pool
        self.add_out_projection = add_out_projection
        self.init_device = init_device
        self.boundary_predictor = boundary_predictor
        self.boundary_predictor_lookahead = boundary_predictor_lookahead
        self.blt_k = blt_k
        self.represent_bytes_with_embeddings = represent_bytes_with_embeddings
        self.blt_compat = blt_compat
        self.dtype = dtype

        if self.add_hash_embeddings + self.add_expanded_embeddings > 1:
            raise ValueError("Only one of add_hash_embeddings and add_expanded_embeddings can be True.")

        if self.boundary_predictor == "dtp":
            self.boundary_predictor_module = DTPBoundaryPredictor(
                d_model,
                init_device=init_device,
                dtype=dtype,
            )
        elif self.boundary_predictor == "dtp_chonky":
            self.boundary_predictor_module = DTPBoundaryPredictor(
                d_model,
                use_transformer_style_mlp=True,
                init_device=init_device,
                dtype=dtype,
            )
        elif self.boundary_predictor == "hnet":
            self.boundary_predictor_module = HNetBoundaryPredictor(
                d_model,
                boundary_predictor_lookahead=boundary_predictor_lookahead,
                init_device=init_device,
                dtype=dtype,
            )
        elif self.boundary_predictor is not None:
            raise ValueError(f"Unknown boundary predictor: {self.boundary_predictor}")
        else:
            self.boundary_predictor_module = None

        self.embedding = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)

        if self.add_hash_embeddings:
            if hash_byte_group_size is None or hash_byte_group_vocab is None or hash_byte_group_nb_functions is None:
                raise ValueError("Hash embeddings are enabled but hash_byte_group_size, hash_byte_group_vocab, or hash_byte_group_nb_functions are None.")

            total_hash_embeddings = hash_byte_group_nb_functions * len(hash_byte_group_size)
            self.hash_embeddings = nn.ModuleList([
                nn.Embedding(
                    hash_byte_group_vocab[hash_embed_idx],
                    d_model,
                    dtype=dtype,
                    device=init_device
                ) for hash_embed_idx in range(total_hash_embeddings)
            ])
            self.expanded_embeddings = None
        elif self.add_expanded_embeddings:
            assert subword_vocab_size is not None
            self.hash_embeddings = None
            self.expanded_embeddings = nn.Embedding(subword_vocab_size, d_model, dtype=dtype, device=init_device)
        else:
            self.hash_embeddings = None
            self.expanded_embeddings = None

        cache = BufferCache()
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            self.blocks[str(block_idx)] = block_config.build(
                d_model=d_model,
                block_idx=block_idx,
                n_layers=n_layers,
                init_device=init_device,
                cache=cache,
            )

        if self.pooling == "cross_attn":
            assert self.blt_k is not None

            self.patch_embedding_projection = nn.Linear(
                d_model,
                d_model * self.blt_k,
                dtype=dtype,
                device=init_device,
                bias=False,
            )
            self.cross_attention = CrossAttention(d_model, cross_attn_n_heads, dtype=dtype, init_device=init_device)
            self.padding_parameters = None
        elif self.pooling == "hnet":
            self.patch_embedding_projection = None
            self.cross_attention = None
            if d_global_model > d_model:
                self.padding_parameters = nn.Parameter(
                    torch.zeros(d_global_model - d_model, dtype=dtype, device=init_device),
                )
            else:
                self.padding_parameters = None

        if self.add_norm_after_last_block:
            self.post_last_block_norm = nn.RMSNorm(
                d_model,
                eps=1e-5,  # TODO: make hparam
                dtype=dtype,
                device=init_device,
            )
        else:
            self.post_last_block_norm = None

        if self.add_norm_after_pool:
            self.post_pool_norm = nn.RMSNorm(
                d_global_model,
                eps=1e-5,  # TODO: make hparam
                dtype=dtype,
                device=init_device,
            )
        else:
            self.post_pool_norm = None

        if self.add_out_projection:
            self.out_projection = nn.Linear(
                d_model * self.blt_k if self.blt_k is not None else d_global_model,
                d_global_model,
                dtype=dtype,
                device=init_device,
                bias=not blt_compat,
            )
        else:
            self.out_projection = None

        self.use_rolling_past_tokens = True

        self.has_cache = False
        self.cache_n_last_tokens = cache_n_last_tokens # for toucan-style decoding all at once
        if self.hash_byte_group_size is not None:
            self.cache_n_last_tokens = max(self.cache_n_last_tokens, max(self.hash_byte_group_size))

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        hash_embeddings = self.hash_embeddings if self.hash_embeddings is not None else []

        for emb in [self.embedding, *hash_embeddings]:
            fully_shard(emb, mesh=dp_mesh, **fsdp_kwargs)
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, emb).set_unshard_in_backward(False)

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                **fsdp_kwargs,
            )

        if self.cross_attention is not None:
            fully_shard(self.cross_attention, mesh=dp_mesh, **fsdp_kwargs)

        if self.out_projection is not None:
            fully_shard(self.out_projection, mesh=dp_mesh, **fsdp_kwargs)

        if self.boundary_predictor is not None:
            fully_shard(self.boundary_predictor_module, mesh=dp_mesh, **fsdp_kwargs)  # type: ignore

        if self.post_last_block_norm is not None:
            fully_shard(self.post_last_block_norm, mesh=dp_mesh, **fsdp_kwargs)

        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def apply_compile(self):
        for block in self.blocks.values():  # type: ignore
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.out_projection is not None:
            self.out_projection.compile(fullgraph=True, dynamic=True)

        if self.boundary_predictor_module is not None:
            self.boundary_predictor_module.compile(fullgraph=False, dynamic=True)

        if self.post_last_block_norm is not None:
            self.post_last_block_norm.compile(fullgraph=True, dynamic=True)

    def fix_init(self, embedding_init_path: Optional[str], target_embeddings, n_estimate=8192, cache_dir: Optional[str] = None):
        """
        Rescale such that the local encoder outputs (given random inputs) have the same mean and std as the provided embeddings.
        
        Also inits HNetBoundaryPredictor q and k to weights to identity following HNet.

        """            
        if embedding_init_path is not None:
            # load embedding inits (computed via compute_hash_embedding_init.py)
            if isinstance(self.embedding.weight.data, DTensor):
                self.embedding.weight.data[:] = distribute_tensor(
                    torch.load(resource_path(embedding_init_path, "embedding_init.pth", local_cache=cache_dir)),
                    device_mesh=self.embedding.weight.data.device_mesh,
                    placements=self.embedding.weight.data.placements,
                )
            else:
                self.embedding.weight.data[:] = torch.load(Path(embedding_init_path) / "embedding_init.pth")

            if self.hash_embeddings is not None:
                for i, hash_embedding in enumerate(self.hash_embeddings):
                    if isinstance(hash_embedding.weight.data, DTensor):
                        hash_embedding.weight.data[:] = distribute_tensor(
                            torch.load(resource_path(embedding_init_path, f"hash_embedding_init_{i}.pth", local_cache=cache_dir)),
                            device_mesh=hash_embedding.weight.data.device_mesh,
                            placements=hash_embedding.weight.data.placements,
                        )
                    else:
                        hash_embedding.weight.data[:] = torch.load(resource_path(embedding_init_path, f"hash_embedding_init_{i}.pth", local_cache=cache_dir))  # type: ignore

        if self.expanded_embeddings is not None:
            self.expanded_embeddings.weight.data[:] = target_embeddings[:self.expanded_embeddings.weight.shape[0]]

        te_mean = target_embeddings.mean(0)
        # .std not supported for DTensor
        te_std = target_embeddings.var(0).sqrt()
        device = target_embeddings.device
        dummy_input = torch.randint(0, self.embedding.weight.shape[0], (n_estimate,), device=device).unsqueeze(0)
        patch_lens = torch.ones((1, n_estimate), dtype=torch.long, device=dummy_input.device)
        patch_ids = torch.arange(n_estimate, device=dummy_input.device).unsqueeze(0)

        _, h_patch, _, _ = self(
            tokens=dummy_input,
            expanded_input_ids=dummy_input if self.add_expanded_embeddings else None,
            patch_lens=patch_lens,
            patch_ids=patch_ids,
        )

        def maybe_distribute(tensor: torch.Tensor) -> DTensor | torch.Tensor:
            if isinstance(self.embedding.weight.data, DTensor):
                return distribute_tensor(
                    tensor,
                    device_mesh=self.embedding.weight.data.device_mesh,
                )
            else:
                return tensor

        h_patch_mean = h_patch[0].mean(0)
        h_patch_std = h_patch[0].var(0).sqrt()
        h_patch_mean = maybe_distribute(h_patch_mean.detach())
        h_patch_std = maybe_distribute(h_patch_std.detach())

        if self.out_projection is not None:
            self.out_projection.weight.data *= (te_std / h_patch_std).unsqueeze(0)
            self.out_projection.bias.data[:] = te_mean - h_patch_mean * (te_std / h_patch_std)

        if isinstance(self.boundary_predictor_module, HNetBoundaryPredictor):
            self.boundary_predictor_module.q_proj_layer.weight.data[:] = maybe_distribute(torch.eye(self.d_model, device=device))
            self.boundary_predictor_module.k_proj_layer.weight.data[:] = maybe_distribute(torch.eye(self.d_model, device=device))

        # verify
        # _, h_patch_fixed, _, _ = self(
        #     tokens=dummy_input,
        #     patch_lens=patch_lens,
        #     patch_ids=patch_ids,
        #     cross_attn_mask=None, # fine not to mask since mask does not change out magnitude
        # )

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
        device = next(self.parameters()).device
        self.has_cache = True

        self.cache_seqlens = 0
        if self.use_rolling_past_tokens:
            self.rolling_past_tokens = torch.zeros((batch_size, self.cache_n_last_tokens), dtype=torch.long, device=device)
            self.n_rolling_past_tokens = 0
        else:
            self.rolling_past_tokens = None
            self.n_rolling_past_tokens = 0
        self.last_h = torch.zeros((batch_size, self.d_model), dtype=self.dtype, device=device)

    def free_inference_cache(self):
        self.has_cache = False
        if hasattr(self, "cache_seqlens"):
            del self.cache_seqlens
        if hasattr(self, "rolling_past_tokens"):
            del self.rolling_past_tokens
        if hasattr(self, "n_rolling_past_tokens"):
            del self.n_rolling_past_tokens
        if hasattr(self, "last_h"):
            del self.last_h

    def _pool_hnet(
        self,
        h: torch.Tensor,
        boundary_mask: torch.Tensor | None,
        n_patches: int,
        boundary_state: Optional[MaskState] = None,
    ):
        if self.has_cache and self.cache_seqlens > 0:
            assert boundary_state is not None
            if boundary_state.all():
                assert h.shape[1] == 1
                reduced_h = h
            else:
                reduced_h = h[[], :, :]
        else:
            assert boundary_mask is not None

            L = h.shape[1]
            token_idx = (
                torch.arange(L, device=h.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            index = seq_sorted_indices[:, :n_patches, None].expand(
                -1, -1, h.shape[-1]
            )

            reduced_h = torch.gather(
                h,
                dim=1,
                index=index,
            )
        if self.padding_parameters is not None:
            padded_h = torch.cat(
                (reduced_h, self.padding_parameters.expand(reduced_h.shape[:-1] + (-1,))), dim=-1
            )
        else:
            padded_h = reduced_h

        return padded_h


    def _pool_blt(
        self,
        h: torch.Tensor,
        n_patches: int,
        boundary_mask: torch.Tensor,
    ):
        if self.cross_attention is None or self.patch_embedding_projection is None or self.blt_k is None:
            raise ValueError("Cross attention is disabled, can not pool with BLT method.")

        B, L = boundary_mask.shape

        token_idx = (
            torch.arange(L, device=h.device)[None, :]
            + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :n_patches]
        last_increasing_index = ((seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]) < 0).max(-1)
        patch_mask = (
            (torch.arange(seq_sorted_indices.shape[1], device=h.device)[None, :] <= last_increasing_index.indices[:, None]) |
            (torch.zeros(seq_sorted_indices.shape[:2], dtype=torch.bool, device=h.device) == last_increasing_index.values[:, None]) # case where never not increasing (no padding)
        )
        patch_lens = torch.ones_like(seq_sorted_indices)
        # bos always one
        patch_lens[:, 1:] = seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]
        patch_lens = torch.where(
            patch_mask,
            patch_lens,
            torch.zeros_like(patch_lens),
        )
        patch_ids = blt_utils.lengths_to_ids(patch_lens, h.shape[1])
        cross_attn_mask = blt_utils.cross_attn_mask(
            patch_ids,
            patch_lens,
            patches_as_queries=True,
            cross_attn_k=self.blt_k,
            block_mask=True,
        )

        # downsample h
        # DIVERGENCE FROM BLT: + 1 for padding
        reduced_h = torch.zeros(
            (h.shape[0], patch_lens.shape[-1] + 1, h.shape[-1]), dtype=h.dtype, device=h.device
        )
        reduced_h = reduced_h.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
            reduce=self.cross_attn_init_pooling,
            include_self=False,
        )
        reduced_h = reduced_h[:, :-1, :]  # DIVERGENCE FROM BLT: remove padding

        # expand seq length by a factor of k (k=2 or 4 in BLT released checkpoints)
        # i.e. per patch, conduct k cross attentions (each with h heads)
        # NOTE: the need for an upprojection seems to imply an unwanted information bottleneck?
        if self.cross_attn_do_project:
            patch_embedding_init = self.patch_embedding_projection(reduced_h).reshape(
                reduced_h.shape[0], reduced_h.shape[1] * self.blt_k, reduced_h.shape[2]
            )
        else:
            patch_embedding_init = reduced_h

        # apply cross attention
        residual = self.cross_attention(
            q=patch_embedding_init,
            kv=h,
            mask=cross_attn_mask,
        )

        # residual connection + reshape back into patch length (from patch_length * k)
        # NOTE: BLT applies the residual connection two times (presumably on accident?), so we have to 2x here.
        residual_factor = 2 if self.blt_compat else 1
        patch_embedding = (patch_embedding_init * residual_factor + residual).reshape(
            reduced_h.shape[0], reduced_h.shape[1], -1
        ).clone()
        # NOTE: ^clone is crucial here! otherwise get strange `SetStorage` errors in the backward pass on L40.
        # (but possibly not on H100?)

        return patch_embedding

    def pool(
        self,
        h: torch.Tensor,
        boundary_mask: torch.Tensor | None,
        n_patches: int,
        boundary_state: Optional[MaskState] = None,
    ):
        if self.pooling == "cross_attn":
            patch_embeddings = self._pool_blt(
                h=h,
                n_patches=n_patches,
                boundary_mask=boundary_mask,  # type: ignore
            )
        elif self.pooling == "hnet":
            patch_embeddings = self._pool_hnet(
                h=h,
                boundary_mask=boundary_mask,
                n_patches=n_patches,
                boundary_state=boundary_state,
            )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}. Supported methods are 'cross_attn' and 'hnet'.")

        if self.post_pool_norm is not None:
            patch_embeddings = self.post_pool_norm(patch_embeddings)

        if self.out_projection is not None:
            patch_embeddings = self.out_projection(patch_embeddings)

        return patch_embeddings

    def _embed(self, tokens, expanded_input_ids: Optional[torch.Tensor] = None):
        embeddings = self.embedding(tokens)
        if self.add_hash_embeddings:
            assert (
                self.hash_embeddings is not None and 
                self.hash_byte_group_nb_functions is not None and 
                self.hash_byte_group_size is not None and 
                self.hash_byte_group_vocab is not None
            )

            embeddings = add_hash_embeddings(
                embeddings,
                tokens,
                self.hash_embeddings,
                self.hash_byte_group_nb_functions,
                self.hash_byte_group_size,
                self.hash_byte_group_vocab,
            )
        elif self.add_expanded_embeddings:
            assert expanded_input_ids is not None and self.expanded_embeddings is not None
            embeddings = embeddings + self.expanded_embeddings(expanded_input_ids)

        return embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        expanded_input_ids: Optional[torch.Tensor] = None,
        cross_attn_mask: BlockMask | None = None,
        boundary_predictor_backprop_through_encoder: bool = True,
        boundary_threshold: str = "sample:0",
        teacher_force_boundaries: bool = False,
        true_boundary_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        embeddings = self._embed(tokens, expanded_input_ids)

        # pass through encoder layers
        h = embeddings

        dtype = h.dtype

        for block in self.blocks.values():
            # TODO(benjaminm): do we need mark_dynamic here / in general?
            # Mark sizes as dynamic for torch.compile().
            #if self.compile_enabled:
            #    mark_dynamic(h, (0, 1), strict=False)
            # TODO(benjaminm): do we need local_block_kwargs?
            h = block(h)

        if self.post_last_block_norm is not None:
            h = self.post_last_block_norm(h)

        if self.boundary_predictor_module is not None:
            if boundary_predictor_backprop_through_encoder:
                boundary_logprobs, boundary_mask = self.boundary_predictor_module(
                    h,
                    boundary_threshold,
                    teacher_force_boundaries=teacher_force_boundaries,
                    true_boundary_mask=true_boundary_mask,
                )
            else:
                boundary_logprobs, boundary_mask = self.boundary_predictor_module(
                    h.detach(),
                    boundary_threshold,
                    teacher_force_boundaries=teacher_force_boundaries,
                    true_boundary_mask=true_boundary_mask,
                )
        else:
            assert true_boundary_mask is not None
            boundary_mask = true_boundary_mask
            boundary_logprobs = None

        # BLT: downsample + cross attn
        # HNet: select + padding
        patch_embeddings = self.pool(
            h=h,
            boundary_mask=boundary_mask,
            n_patches=patch_lens.shape[1],
        )

        if self.represent_bytes_with_embeddings:
            h = embeddings

        return h, patch_embeddings, boundary_logprobs, boundary_mask

    def inference_forward(
        self,
        tokens: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        boundary_state: MaskState,
        expanded_input_ids: Optional[torch.Tensor] = None,
        cross_attn_mask: BlockMask | None = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        boundary_threshold: str = "sample:0",
        teacher_force_boundaries: bool = False,
        true_boundary_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor | None]:
        if self.use_rolling_past_tokens:
            assert self.rolling_past_tokens is not None

            if self.has_cache and self.cache_seqlens == 0:
                tokens_to_cache = tokens[:, max(tokens.shape[1]-self.rolling_past_tokens.shape[1],0):]
                self.rolling_past_tokens[:, -tokens_to_cache.shape[1]:] = tokens_to_cache
                self.n_rolling_past_tokens += tokens_to_cache.shape[1]
            elif self.has_cache:
                boundary_state.selective_put(
                    torch.roll(boundary_state.selective_get(self.rolling_past_tokens, inv=True), shifts=-1, dims=1),
                    self.rolling_past_tokens,
                    inv=True,
                )
                boundary_state.selective_put(
                    boundary_state.selective_get(tokens[:, -1], inv=True),
                    self.rolling_past_tokens[:, -1],
                    inv=True,
                )
            # TODO(benjaminm): update n_rolling_past_tokens / adapt to the extended embeddings case
            #if self.n_rolling_past_tokens < self.rolling_past_tokens.shape[1]:
            #    self.n_rolling_past_tokens += 1

        if self.has_cache and self.cache_seqlens > 0 and self.use_rolling_past_tokens:
            assert tokens.shape[1] == 1 and self.rolling_past_tokens is not None
            if self.add_hash_embeddings:
                assert self.hash_byte_group_size is not None
                n_tokens_to_retrieve = min(self.n_rolling_past_tokens, max(self.hash_byte_group_size))
            else:
                n_tokens_to_retrieve = 1

            embeddings = self._embed(self.rolling_past_tokens[:, -n_tokens_to_retrieve:])[:, -1:]
        else:
            embeddings = self._embed(tokens)

        # pass through encoder layers
        if self.has_cache and self.cache_seqlens > 0:
            if not boundary_state.all():
                h = boundary_state.selective_get(embeddings, inv=True)

                for block in self.blocks.values():
                    h = block(h, cache_mask=boundary_state)

                if self.post_last_block_norm is not None:
                    h = self.post_last_block_norm(h)

                boundary_state.selective_put(h[:, -1, :], self.last_h, inv=True)

            h = self.last_h.unsqueeze(1)
        else:
            h = embeddings
            for block in self.blocks.values():
                h = block(h, sequence_start_indices=sequence_start_indices)

            if self.post_last_block_norm is not None:
                h = self.post_last_block_norm(h)

            self.last_h.copy_(h[:, -1, :])

        # only use this boundary predictor for prefilling
        if self.boundary_predictor_module is not None and (not self.has_cache or self.cache_seqlens == 0):
            boundary_logprobs, boundary_mask = self.boundary_predictor_module(
                h,
                boundary_threshold,
                teacher_force_boundaries=teacher_force_boundaries,
                true_boundary_mask=true_boundary_mask,
                sequence_start_indices=sequence_start_indices,
            )
            # can't predict through encoder - must be through prev local decoder step
            boundary_mask[:, -1] = boundary_state.mask
        else:
            boundary_logprobs = None
            boundary_mask = None

        patch_embeddings = self.pool(
            h=h,
            boundary_mask=boundary_mask,
            n_patches=boundary_mask.sum(-1).max().item() if boundary_mask is not None else 1,
            boundary_state=boundary_state,
        )

        if self.represent_bytes_with_embeddings:
            h = embeddings

        if self.has_cache:
            self.cache_seqlens += tokens.shape[1]

        return h, patch_embeddings, boundary_logprobs, boundary_mask

class LocalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sliding_window_size: int,
        d_model: int,
        d_global_model: int,
        n_layers: int,
        cross_attn_n_heads: int,
        block_config,
        depooling: str,
        add_norm_before_first_block: bool,
        add_norm_onto_residual: bool,
        add_in_projection: bool,
        add_projected_patch_residuals: bool = False,
        hnet_smooth: bool = True,
        hnet_smooth_ste: bool = False,
        hnet_modulate: bool = True,
        blt_k: Optional[int] = None,
        blt_compat: bool = False,  # for compat with BLT checkpoints
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.depooling = depooling
        self.add_norm_before_first_block = add_norm_before_first_block
        self.add_norm_onto_residual = add_norm_onto_residual
        self.add_in_projection = add_in_projection
        self.add_projected_patch_residuals = add_projected_patch_residuals
        self.hnet_smooth = hnet_smooth
        self.hnet_smooth_ste = hnet_smooth_ste
        self.hnet_modulate = hnet_modulate
        self.blt_k = blt_k
        self.blt_compat = blt_compat
        self.dtype = dtype

        if self.depooling == "cross_attn":
            assert self.blt_k is not None

            self.patch_embedding_projection = nn.Linear(
                d_global_model,
                self.d_model * self.blt_k,
                dtype=dtype,
                device=init_device,
                bias=False,
            )
        else:
            self.patch_embedding_projection = None

        cache = BufferCache()
        self.blocks = nn.ModuleDict()
        self.cross_attentions = nn.ModuleDict()

        for block_idx in range(n_layers):
            self.blocks[str(block_idx)] = block_config.build(
                d_model=d_model,
                block_idx=block_idx,
                n_layers=n_layers,
                init_device=init_device,
                cache=cache,
            )
            if self.depooling == "cross_attn":
                self.cross_attentions[str(block_idx)] = CrossAttention(
                    d_model,
                    cross_attn_n_heads,
                    dtype=dtype,
                    init_device=init_device,
                )

        if self.add_norm_before_first_block:
            self.initial_norm = nn.RMSNorm(
                d_global_model,
                eps=1e-5,  # TODO: make hparam
                dtype=dtype,
                device=init_device,
            )
        else:
            self.initial_norm = None

        if self.add_norm_onto_residual:
            self.residual_norm = nn.RMSNorm(
                d_model,
                eps=1e-5,  # TODO: make hparam
                dtype=dtype,
                device=init_device,
            )
        else:
            self.residual_norm = None

        if self.add_in_projection:
            self.in_projection = nn.Linear(
                d_model,
                d_model,
                dtype=dtype,
                device=init_device,
                bias=True,
            )
        else:
            self.in_projection = None

        if self.add_projected_patch_residuals:
            self.patch_residuals_projection = nn.Linear(
                d_global_model,
                d_global_model,
                dtype=dtype,
            )
        else:
            self.patch_residuals_projection = None

        self.boundary_embedding = nn.Embedding(1, d_model, dtype=dtype, device=init_device)
        self.has_cache = False

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                **fsdp_kwargs
            )

        for cross_attn in self.cross_attentions.values():
            fully_shard(cross_attn, mesh=dp_mesh, **fsdp_kwargs)

        # should we shard the patch embedding projection?
        #fully_shard(self.patch_embedding_projection, mesh=dp_mesh, **fsdp_kwargs)
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def apply_compile(self):
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.initial_norm is not None:
            self.initial_norm.compile(fullgraph=True, dynamic=True)

        if self.residual_norm is not None:
            self.residual_norm.compile(fullgraph=True, dynamic=True)
        
        if self.in_projection is not None:
            self.in_projection.compile(fullgraph=True, dynamic=True)

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
        device = next(self.parameters()).device
        self.has_cache = True

        self.cache_seqlens = 0
        self.last_value = torch.zeros((batch_size, self.d_model), dtype=self.dtype, device=device)

    def free_inference_cache(self):
        self.has_cache = False
        if hasattr(self, "cache_seqlens"):
            del self.cache_seqlens
        if hasattr(self, "last_value"):
            del self.last_value

    def _depool_hnet(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        boundary_logprobs: torch.Tensor,
        boundary_mask: Optional[torch.Tensor],
        boundary_state: Optional[MaskState] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        block_size: int = 256,
        headdim: int = 32,
        epsilon: float = 1e-3,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.has_cache and self.cache_seqlens > 0:
            assert boundary_state is not None
            assert not self.hnet_smooth # not implemented for now

            if patch_embeds.numel() > 0:
                # we got a new value from the global model, so must be at boundary position
                h_patch = patch_embeds[:, -1:, :self.d_model]
                h = self.boundary_embedding.weight.unsqueeze(0) + h_patch
                self.last_value.copy_(h_patch[:, -1])
            else:
                h = embeds + self.last_value.unsqueeze(1)

            # skip positions where we have a boundary until we get a new value from the global model
            if patch_embeds.numel() == 0:
                h = boundary_state.selective_get(h, inv=True)
            else:
                boundary_state = None

            if h.shape[0] > 0:
                for block_idx in range(self.n_layers):
                    block = self.blocks[str(block_idx)]
                    h = block(h, cache_mask=boundary_state)

            # TODO(benjaminm): clean up / return None / don't return so many things?
            return (h, h), h, h
        else:
            assert boundary_mask is not None

            h_patch = patch_embeds[..., :self.d_model] # global d -> local d

            B, L = boundary_mask.shape

            token_idx = (
                torch.arange(L, device=patch_embeds.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :patch_embeds.shape[1]]
            last_increasing_index = ((seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]) < 0).max(-1)
            patch_mask = (
                (torch.arange(patch_embeds.shape[1], device=patch_embeds.device)[None, :] <= last_increasing_index.indices[:, None]) |
                (torch.zeros(patch_embeds.shape[:2], dtype=torch.bool, device=patch_embeds.device) == last_increasing_index.values[:, None]) # case where never not increasing (no padding)
            )

            if self.hnet_smooth:
                p = torch.gather(torch.exp(boundary_logprobs).float().clip(min=epsilon, max=1 - epsilon), dim=1, index=seq_sorted_indices)

                prepool_out = ema_pool(h_patch, p, block_size=block_size, headdim=headdim)
                if self.hnet_smooth_ste:
                    prepool_out = ste_select_func(prepool_out, h_patch.detach())
            else:
                prepool_out = h_patch

            # TODO(benjaminm): clipping is problematic if it happens too much; track clip %.
            plug_back_idx = (torch.cumsum(boundary_mask, dim=1) - 1).clip(min=0, max=prepool_out.shape[1] - 1)
            depool_out = torch.gather(
                prepool_out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

            if self.hnet_modulate and boundary_logprobs is not None:
                boundary_probs = torch.exp(boundary_logprobs).to(depool_out.dtype)
                selected_boundary_probs = torch.where(
                    boundary_probs > 0.5,
                    boundary_probs,
                    1 - boundary_probs,
                )
                depool_out_modulated = depool_out * ste_func(selected_boundary_probs).unsqueeze(-1)
            else:
                depool_out_modulated = depool_out

            # skip bos - considered boundary
            h = (depool_out_modulated[:, :-1] + embeds[:, 1:])
            h_b = (self.boundary_embedding.weight.unsqueeze(0) + prepool_out)

            # +1 to keep multiple of
            h_with_b = torch.zeros(
                (h.shape[0], h.shape[1] + patch_embeds.shape[1] + 1, h.shape[2]),
                device=h.device,
                dtype=h.dtype
            )

            if self.has_cache:
                self.last_value.copy_(prepool_out[:, -1])
                self.cache_seqlens += h_with_b.shape[1]

            non_b_indices = torch.arange(len(h[0]), device=h.device).unsqueeze(0).repeat(len(h), 1)
            non_b_indices += plug_back_idx[:, :-1] + 1 # offset by bos
            b_indices = seq_sorted_indices + torch.arange(patch_embeds.shape[1], device=h.device).unsqueeze(0)
            b_indices = torch.where(patch_mask, b_indices, torch.ones_like(b_indices))

            if sequence_start_indices is not None:
                pad_mask = torch.arange(h.shape[1], device=h.device)[None, :] < sequence_start_indices[:, None]
                h = torch.where(pad_mask.unsqueeze(-1), torch.zeros_like(h), h)

                offsets = patch_embeds.shape[1] - boundary_mask.sum(-1)
                sequence_start_indices += offsets
                b_indices += offsets.unsqueeze(-1)
                non_b_indices += offsets.unsqueeze(-1)

            h_with_b.scatter_(
                1,
                non_b_indices.unsqueeze(-1).expand(-1, -1, self.d_model), # skip bos - considered boundary
                h
            )
            h_with_b.scatter_add_(
                1,
                b_indices.unsqueeze(-1).expand(-1, -1, self.d_model),
                torch.where(patch_mask.unsqueeze(-1), h_b, torch.zeros_like(h_b))
            )

            for block_idx in range(self.n_layers):
                block = self.blocks[str(block_idx)]
                h_with_b = block(h_with_b, sequence_start_indices=sequence_start_indices)

            h_for_true_boundaries = torch.gather(
                h_with_b,
                dim=1,
                # can't predict first boundary / bos
                index=(b_indices[:, 1:] - 1).unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            h_for_all_boundaries = torch.gather(
                h_with_b,
                dim=1,
                index=non_b_indices.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            h_for_logits = torch.gather(
                h_with_b,
                dim=1,
                index=(non_b_indices - 1).unsqueeze(-1).expand(-1, -1, self.d_model),
            )

            # [:-1] to strip multiple of
            return (h_for_true_boundaries, h_for_all_boundaries), h_for_logits, h_with_b[:, :-1]

    def _depool_blt(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        cross_attn_mask: BlockMask | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.patch_embedding_projection is None or self.blt_k is None:
            raise ValueError("Patch embedding projection is not defined, can not depool with BLT method.")

        # expand seq length by a factor of k (k=2 in BLT released checkpoints)
        patch_embeds_projected = self.patch_embedding_projection(patch_embeds).reshape(
            patch_embeds.shape[0], patch_embeds.shape[1] * self.blt_k, embeds.shape[2]
        )

        h = embeds

        for block_idx in range(self.n_layers):
            cross_attn = self.cross_attentions[str(block_idx)]
            block = self.blocks[str(block_idx)]

            # NOTE: What about LN before/after cross attn?
            h_cross = cross_attn(q=h, kv=patch_embeds_projected, mask=cross_attn_mask)
            # NOTE: same thing, BLT applies the residual connection two times (presumably on accident?), so we have to 2x here.
            residual_factor = 2 if self.blt_compat else 1
            h = h * residual_factor + h_cross
            h = block(h)

        return (h, h), h, h

    def depool(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        boundary_logprobs: torch.Tensor,
        boundary_mask: torch.Tensor | None,
        cross_attn_mask: BlockMask | None = None,
        boundary_state: Optional[MaskState] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.depooling == "cross_attn":
            return self._depool_blt(embeds, patch_embeds, cross_attn_mask)
        elif self.depooling == "hnet":
            return self._depool_hnet(
                embeds,
                patch_embeds,
                boundary_logprobs,
                boundary_mask,
                boundary_state,
                sequence_start_indices,
            )
        else:
            raise ValueError(f"Unknown depooling method: {self.depooling}. Supported methods are 'cross_attn' and 'hnet'.")

    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        patch_residuals: torch.Tensor,
        boundary_logprobs: torch.Tensor,
        boundary_mask: torch.Tensor,
        cross_attn_mask: BlockMask | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.residual_norm is not None:
            h = self.residual_norm(embeds)
        else:
            h = embeds

        if self.in_projection is not None:
            h = self.in_projection(h)

        if self.initial_norm is not None:
            h_patch = self.initial_norm(patch_embeds)
        else:
            h_patch = patch_embeds

        if self.patch_residuals_projection is not None:
            h_patch = h_patch + self.patch_residuals_projection(patch_residuals)

        return self.depool(
            embeds=h,
            patch_embeds=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            cross_attn_mask=cross_attn_mask,
        )

    def inference_forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        patch_residuals: torch.Tensor,
        boundary_logprobs: torch.Tensor,
        boundary_state: MaskState,
        boundary_mask: torch.Tensor | None,
        cross_attn_mask: BlockMask | None = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.residual_norm is not None:
            h = self.residual_norm(embeds)
        else:
            h = embeds

        if self.in_projection is not None:
            h = self.in_projection(h)

        if self.initial_norm is not None:
            h_patch = self.initial_norm(patch_embeds)
        else:
            h_patch = patch_embeds

        if self.patch_residuals_projection is not None:
            h_patch = h_patch + self.patch_residuals_projection(patch_residuals)

        return self.depool(
            embeds=h,
            patch_embeds=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            cross_attn_mask=cross_attn_mask,
            boundary_state=boundary_state,
            sequence_start_indices=sequence_start_indices,
        )