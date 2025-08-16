from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast
import math

from einops import repeat, rearrange
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
from olmo_core.nn.buffer_cache import BufferCache
from .embed import add_hash_embeddings
from .utils import log1mexp

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

def ste_func(x: torch.Tensor) -> torch.Tensor:
    return STE.apply(x)  # type: ignore


# 2-layer MLP as in DTP
class DTPBoundaryPredictor(nn.Module):
    def __init__(self, d_model: int, init_device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = 4 # as in DTP (and OLMo)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * self.expansion_factor, device=init_device),
            nn.SiLU(),
            nn.Linear(d_model * self.expansion_factor, 1, device=init_device),
        )

    def forward(self, x: torch.Tensor, boundary_threshold: float | int) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_logprobs = F.logsigmoid(self.mlp(x)).squeeze(-1).float()

        # make sure boundary at first
        boundary_logprobs[:, 0] = 0.0

        if boundary_threshold > 1:
            assert isinstance(boundary_threshold, int)
            thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - (boundary_threshold / boundary_logprobs.shape[1]))
            boundary_mask = (boundary_logprobs > thresholds.unsqueeze(-1))
        else:
            assert isinstance(boundary_threshold, float)
            boundary_mask = (boundary_logprobs > math.log(boundary_threshold))

        return boundary_logprobs, boundary_mask

# cosine-similarity based boundary predictor as in H-Net
class HNetBoundaryPredictor(nn.Module):
    def __init__(self, d_model: int, init_device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, device=init_device)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, device=init_device)
        
    def forward(self, hidden_states: torch.Tensor, boundary_threshold: float | int, epsilon: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        boundary_logprobs = torch.log1p(-cos_sim.float().clip(max=1.0 - epsilon)) - math.log(2)
        PAD_LOGPROB = 0.0
        boundary_logprobs = F.pad(boundary_logprobs, (1, 0), "constant", PAD_LOGPROB)

        if boundary_threshold > 1:
            assert isinstance(boundary_threshold, int)
            thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - (boundary_threshold / boundary_logprobs.shape[1]))
            boundary_mask = (boundary_logprobs > thresholds.unsqueeze(-1))
        else:
            assert isinstance(boundary_threshold, float)
            boundary_mask = (boundary_logprobs > math.log(boundary_threshold))

        return boundary_logprobs, boundary_mask


class CrossAttention(nn.Module):
    # TODO(benjaminm): make norm_eps config arg without default?
    def __init__(self, d_model: int, n_heads: int, norm_eps: float = 1e-5, init_device: str = "cpu"):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False, device=init_device)
        self.w_k = nn.Linear(d_model, d_model, bias=False, device=init_device)
        self.w_v = nn.Linear(d_model, d_model, bias=False, device=init_device)
        self.w_out = nn.Linear(d_model, d_model, bias=False, device=init_device)

        self.q_norm = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)
        self.kv_norm = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)

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
        output = flex_attention_comp(q, k, v, block_mask=mask)
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
            block_config,
            add_hash_embeddings: bool,
            hash_byte_group_size: list[int] | None,
            hash_byte_group_vocab: list[int] | None,
            hash_byte_group_nb_functions: int | None,
            pooling: str,
            add_norm_after_last_block: bool,
            add_norm_after_pool: bool,
            add_out_projection: bool,
            boundary_predictor: Optional[str] = None,
            blt_k: Optional[int] = None,
            blt_compat: bool = False,  # for compat with BLT checkpoints
            init_device: str = "cpu",
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
        self.block_config = block_config
        self.add_hash_embeddings = add_hash_embeddings
        self.pooling = pooling
        self.add_norm_after_last_block = add_norm_after_last_block
        self.add_norm_after_pool = add_norm_after_pool
        self.add_out_projection = add_out_projection
        self.init_device = init_device
        self.boundary_predictor = boundary_predictor
        self.blt_k = blt_k
        self.blt_compat = blt_compat

        if self.boundary_predictor == "dtp":
            self.boundary_predictor_module = DTPBoundaryPredictor(d_model, init_device=init_device)
        elif self.boundary_predictor == "hnet":
            self.boundary_predictor_module = HNetBoundaryPredictor(d_model, init_device=init_device)
        elif self.boundary_predictor is not None:
            raise ValueError(f"Unknown boundary predictor: {self.boundary_predictor}")
        else:
            self.boundary_predictor_module = None

        self.embedding = nn.Embedding(vocab_size, d_model, device=init_device)

        if self.add_hash_embeddings:
            if hash_byte_group_size is None or hash_byte_group_vocab is None or hash_byte_group_nb_functions is None:
                raise ValueError("Hash embeddings are enabled but hash_byte_group_size, hash_byte_group_vocab, or hash_byte_group_nb_functions are None.")

            total_hash_embeddings = hash_byte_group_nb_functions * len(hash_byte_group_size)
            self.hash_embeddings = nn.ModuleList([
                nn.Embedding(hash_byte_group_vocab[hash_embed_idx], d_model, device=init_device) for hash_embed_idx in range(total_hash_embeddings)
            ])
        else:
            self.hash_embeddings = None

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
                device=init_device,
                bias=False,
            )
            self.cross_attention = CrossAttention(d_model, cross_attn_n_heads, init_device=init_device)
            self.padding_parameters = None
        elif self.pooling == "hnet":
            self.patch_embedding_projection = None
            self.cross_attention = None
            if d_global_model > d_model:
                self.padding_parameters = nn.Parameter(
                    torch.zeros(d_global_model - d_model, device=init_device),
                )
            else:
                self.padding_parameters = None

        if self.add_norm_after_last_block:
            self.post_last_block_norm = nn.RMSNorm(
                d_model,
                eps=1e-5,  # TODO: make hparam
                device=init_device,
            )
        else:
            self.post_last_block_norm = None

        if self.add_norm_after_pool:
            self.post_pool_norm = nn.RMSNorm(
                d_global_model,
                eps=1e-5,  # TODO: make hparam
                device=init_device,
            )
        else:
            self.post_pool_norm = None

        if self.add_out_projection:
            self.out_projection = nn.Linear(
                d_model * self.blt_k if self.blt_k is not None else d_global_model,
                d_global_model,
                device=init_device,
                bias=not blt_compat,
            )
        else:
            self.out_projection = None

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
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def fix_init(self, embedding_init_path, target_embeddings, n_estimate=10_000, cache_dir: Optional[str] = None):
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

        te_mean = target_embeddings.mean(0)
        # .std not supported for DTensor
        te_std = target_embeddings.var(0).sqrt()
        device = target_embeddings.device
        dummy_input = torch.randint(0, self.embedding.weight.shape[0], (n_estimate,), device=device).unsqueeze(0)
        patch_lens = torch.ones((1, n_estimate), dtype=torch.long, device=dummy_input.device)
        patch_ids = torch.arange(n_estimate, device=dummy_input.device).unsqueeze(0)

        # this is annoying but didn't find a better way to make it compatible with FSDP2
        local_encoder_copy = LocalEncoder(
            vocab_size=self.vocab_size,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            d_global_model=self.d_global_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_hash_embeddings=self.add_hash_embeddings,
            hash_byte_group_size=self.hash_byte_group_size,
            hash_byte_group_vocab=self.hash_byte_group_vocab,
            hash_byte_group_nb_functions=self.hash_byte_group_nb_functions,
            pooling=self.pooling,
            add_norm_after_last_block=self.add_norm_after_last_block,
            add_norm_after_pool=self.add_norm_after_pool,
            add_out_projection=self.add_out_projection,
            boundary_predictor=self.boundary_predictor,
            blt_k=self.blt_k,
            blt_compat=self.blt_compat,
            init_device=device,
        )
        local_encoder_copy.load_state_dict({
            k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in self.state_dict().items()
        })

        _, h_patch, _, _ = local_encoder_copy(
            tokens=dummy_input,
            patch_lens=patch_lens,
            patch_ids=patch_ids,
            cross_attn_mask=None, # fine not to mask since mask does not change out magnitude
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
        # local_encoder_copy.load_state_dict({
        #     k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in self.state_dict().items()
        # })
        # _, h_patch_fixed = local_encoder_copy(
        #     tokens=dummy_input,
        #     patch_lens=patch_lens,
        #     patch_ids=patch_ids,
        #     cross_attn_mask=None, # fine not to mask since mask does not change out magnitude
        # )

    def _pool_hnet(
        self,
        h: torch.Tensor,
        patch_lens: torch.Tensor,
        boundary_logprobs: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        smooth: bool = False,
        teacher_force_boundaries: bool = True,
        block_size: int = 256,
        headdim: int = 32,
        epsilon=1e-3,
    ):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        if (boundary_logprobs is None or boundary_mask is None) and not teacher_force_boundaries:
            if boundary_logprobs is None:
                raise ValueError("Boundaries must be provided if teacher_force_boundaries=False.")

        if smooth and not teacher_force_boundaries:
            # NOT IN HNET! Add pooling to the encoder.
            p = torch.exp(boundary_logprobs).float().clip(min=epsilon, max=1.0 - epsilon)  # type: ignore
            dt = torch.log(1 / (1 - p)).to(h.dtype)
            x = (h / dt[..., None])

            n_heads = self.d_model // headdim
            A = -torch.ones(
                (n_heads,), device=h.device, dtype=torch.float32
            )
            b = p.to(h.dtype)
            c = torch.ones_like(b)

            pool_out = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=headdim),
                repeat(dt, "b l -> b l h", h=n_heads),
                A,
                rearrange(b, "b l -> b l 1 1"),
                rearrange(c, "b l -> b l 1 1"),
                chunk_size=block_size,
                seq_idx=None,
            )
            pool_out = rearrange(pool_out, "b l h p -> b l (h p)")
            pool_out = cast(torch.Tensor, pool_out)
        else:
            pool_out = h

        if teacher_force_boundaries:
            index = (torch.cumsum(patch_lens, dim=1) - 1).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        else:
            L = pool_out.shape[1]
            token_idx = (
                torch.arange(L, device=pool_out.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            index = seq_sorted_indices[:, :patch_lens.shape[1], None].expand(
                -1, -1, pool_out.shape[-1]
            )

        reduced_h = torch.gather(
            pool_out,
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
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask | None,
        reduction: str = "amax",
    ):
        if self.cross_attention is None or self.patch_embedding_projection is None or self.blt_k is None:
            raise ValueError("Cross attention is disabled, can not pool with BLT method.")

        # downsample h
        # DIVERGENCE FROM BLT: + 1 for padding
        reduced_h = torch.zeros(
            (h.shape[0], patch_lens.shape[-1] + 1, h.shape[-1]), dtype=h.dtype, device=h.device
        )
        reduced_h = reduced_h.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
            reduce=reduction,
            include_self=False,
        )
        reduced_h = reduced_h[:, :-1, :]  # DIVERGENCE FROM BLT: remove padding

        # expand seq length by a factor of k (k=2 or 4 in BLT released checkpoints)
        # i.e. per patch, conduct k cross attentions (each with h heads)
        # NOTE: the need for an upprojection seems to imply an unwanted information bottleneck?
        patch_embedding_init = self.patch_embedding_projection(reduced_h).reshape(
            reduced_h.shape[0], reduced_h.shape[1] * self.blt_k, reduced_h.shape[2]
        )

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
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask | None,
        smooth: bool = False,
        teacher_force_boundaries: bool = True,
        boundary_logprobs: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
    ):
        if self.pooling == "cross_attn":
            patch_embeddings = self._pool_blt(
                h=h,
                patch_lens=patch_lens,
                patch_ids=patch_ids,
                cross_attn_mask=cross_attn_mask,
            )
        elif self.pooling == "hnet":
            patch_embeddings = self._pool_hnet(
                h=h,
                patch_lens=patch_lens,
                boundary_logprobs=boundary_logprobs,
                boundary_mask=boundary_mask,
                smooth=smooth,
                teacher_force_boundaries=teacher_force_boundaries,
            )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}. Supported methods are 'cross_attn' and 'hnet'.")
        
        if self.post_pool_norm is not None:
            patch_embeddings = self.post_pool_norm(patch_embeddings)

        if self.out_projection is not None:
            patch_embeddings = self.out_projection(patch_embeddings)

        return patch_embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask | None,
        smooth: bool = False,
        teacher_force_boundaries: bool = True,
        boundary_predictor_backprop_through_encoder: bool = True,
        boundary_threshold: float | int = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        # pass through encoder layers
        h = embeddings
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
                boundary_logprobs, boundary_mask = self.boundary_predictor_module(h, boundary_threshold)
            else:
                boundary_logprobs, boundary_mask = self.boundary_predictor_module(h.detach(), boundary_threshold)
        else:
            boundary_logprobs = None
            boundary_mask = None

        # BLT: downsample + cross attn
        # HNet: select + padding
        patch_embeddings = self.pool(
            h=h,
            patch_lens=patch_lens,
            patch_ids=patch_ids,
            cross_attn_mask=cross_attn_mask,
            smooth=smooth,
            teacher_force_boundaries=teacher_force_boundaries,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
        )

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
        blt_k: Optional[int] = None,
        blt_compat: bool = False,  # for compat with BLT checkpoints
        init_device: str = "cpu",
    ):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.depooling = depooling
        self.add_norm_before_first_block = add_norm_before_first_block
        self.add_norm_onto_residual = add_norm_onto_residual
        self.add_in_projection = add_in_projection
        self.blt_k = blt_k
        self.blt_compat = blt_compat

        if self.depooling == "cross_attn":
            assert self.blt_k is not None

            self.patch_embedding_projection = nn.Linear(
                d_global_model,
                self.d_model * self.blt_k,
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
                self.cross_attentions[str(block_idx)] = CrossAttention(d_model, cross_attn_n_heads, init_device=init_device)

        if self.add_norm_before_first_block:
            self.initial_norm = nn.RMSNorm(
                d_global_model,
                eps=1e-5,  # TODO: make hparam
                device=init_device,
            )
        else:
            self.initial_norm = None

        if self.add_norm_onto_residual:
            self.residual_norm = nn.RMSNorm(
                d_model,
                eps=1e-5,  # TODO: make hparam
                device=init_device,
            )
        else:
            self.residual_norm = None

        if self.add_in_projection:
            self.in_projection = nn.Linear(
                d_model,
                d_model,
                device=init_device,
                bias=True,
            )
        else:
            self.in_projection = None

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

    def _depool_hnet(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        patch_lens: torch.Tensor,
        boundary_logprobs: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        block_size: int = 256,
        headdim: int = 32,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        h_patch = patch_embeds[..., :self.d_model] # global d -> local d

        # for now, use HNet's smoothing module but with probabilities in {0,1} instead of probabilities in [0,1]
        # (i.e. no smoothing). so we could probably skip this? but not bad to have it implemented and likely
        # no substantial performance difference in the grand scheme.
        if boundary_logprobs is None:
            assert boundary_mask is None
            logp = torch.full((h_patch.shape[0], h_patch.shape[1]), -epsilon, device=h_patch.device, dtype=torch.float32)

            boundary_mask = torch.zeros(
                (embeds.shape[0], embeds.shape[1]), dtype=torch.bool, device=embeds.device
            )
            boundary_mask = boundary_mask.scatter(
                dim=1,
                index=torch.cumsum(patch_lens, dim=1) - 1,
                src=torch.ones_like(patch_lens, dtype=torch.bool),
            )
        else:
            assert boundary_mask is not None
            B, L = boundary_mask.shape

            token_idx = (
                torch.arange(L, device=patch_embeds.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :patch_embeds.shape[1]]
            logp = torch.gather(boundary_logprobs.float().clip(max=-epsilon), dim=1, index=seq_sorted_indices)

        dt = (math.log(1) - log1mexp(logp)).to(h_patch.dtype)
        x = (h_patch / dt[..., None])

        n_heads = self.d_model // headdim
        A = -torch.ones(
            (n_heads,), device=h_patch.device, dtype=torch.float32
        )
        b = torch.exp(logp).to(h_patch.dtype)
        c = torch.ones_like(b)

        # trust the HNet source...
        depool_out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=headdim),
            repeat(dt, "b l -> b l h", h=n_heads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=block_size,
            seq_idx=None,
        )
        depool_out = rearrange(depool_out, "b l h p -> b l (h p)")
        depool_out = cast(torch.Tensor, depool_out)

        # TODO(benjaminm): clipping is problematic if it happens too much; track clip %.
        plug_back_idx = (torch.cumsum(boundary_mask, dim=1) - 1).clip(max=depool_out.shape[1] - 1)
        depool_out = torch.gather(
            depool_out,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        if boundary_logprobs is not None:
            boundary_probs = torch.exp(boundary_logprobs)
            selected_boundary_probs = torch.where(
                boundary_probs > 0.5,
                boundary_probs,
                1 - boundary_probs,
            )
            # TODO(benjaminm): do we want to train this? or detach selected boundary probs?
            depool_out_modulated = depool_out * ste_func(selected_boundary_probs).unsqueeze(-1)
        else:
            depool_out_modulated = depool_out

        h = depool_out_modulated + embeds

        for block_idx in range(self.n_layers):
            block = self.blocks[str(block_idx)]
            h = block(h)

        return h

    def _depool_blt(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        cross_attn_mask: BlockMask | None,
    ) -> torch.Tensor:
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

        return h

    def depool(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        patch_lens: torch.Tensor,
        cross_attn_mask: BlockMask | None = None,
        boundary_logprobs: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.depooling == "cross_attn":
            return self._depool_blt(embeds, patch_embeds, cross_attn_mask)
        elif self.depooling == "hnet":
            return self._depool_hnet(embeds, patch_embeds, patch_lens, boundary_logprobs, boundary_mask)
        else:
            raise ValueError(f"Unknown depooling method: {self.depooling}. Supported methods are 'cross_attn' and 'hnet'.")

    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor | None = None, # unused
        cross_attn_mask: BlockMask | None = None,
        boundary_logprobs: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        return self.depool(
            embeds=h,
            patch_embeds=h_patch,
            patch_lens=patch_lens,
            cross_attn_mask=cross_attn_mask,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
        )