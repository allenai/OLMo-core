from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from torch.distributed.tensor import DTensor, distribute_tensor

from olmo_core.config import Config
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.buffer_cache import BufferCache
from .embed import add_hash_embeddings


@dataclass
class LocalEncoderConfig(Config):
    hash_byte_group_size: list[int]
    hash_byte_group_vocab: int
    hash_byte_group_nb_functions: int
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config
    add_out_projection: bool = False

    def build(self, vocab_size: int) -> nn.Module:
        return LocalEncoder(
            vocab_size=vocab_size,
            hash_byte_group_size=self.hash_byte_group_size,
            hash_byte_group_vocab=self.hash_byte_group_vocab,
            hash_byte_group_nb_functions=self.hash_byte_group_nb_functions,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_out_projection=self.add_out_projection,
        )


@dataclass
class LocalDecoderConfig(Config):
    sliding_window_size: int
    d_model: int
    n_layers: int
    cross_attn_n_heads: int
    block_config: Config

    def build(self, vocab_size: int, d_global_model: int) -> nn.Module:
        return LocalDecoder(
            vocab_size=vocab_size,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            d_global_model=d_global_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
        )

# matching BLT, seems necessary but why?
flex_attention_comp = torch.compile(flex_attention)

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
            hash_byte_group_size: list[int],
            hash_byte_group_vocab: int,
            hash_byte_group_nb_functions: int,
            sliding_window_size: int,
            d_model: int,
            n_layers: int,
            cross_attn_n_heads: int,
            block_config,
            add_out_projection: bool,
            init_device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hash_byte_group_size = hash_byte_group_size
        self.hash_byte_group_vocab = hash_byte_group_vocab
        self.hash_byte_group_nb_functions = hash_byte_group_nb_functions
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.cross_attn_n_heads = cross_attn_n_heads
        self.block_config = block_config
        self.add_out_projection = add_out_projection
        self.init_device = init_device

        self.embedding = nn.Embedding(vocab_size, d_model, device=init_device)

        total_hash_embeddings = hash_byte_group_nb_functions * len(hash_byte_group_size)
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(hash_byte_group_vocab, d_model, device=init_device) for _ in range(total_hash_embeddings)
        ])

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

        self.patch_embedding_projection = nn.Linear(
            d_model,
            d_model * 2,  # TODO: argument for upsampling factor?
            device=init_device,
            bias=False,
        )

        self.cross_attention = CrossAttention(d_model, cross_attn_n_heads, init_device=init_device)

        if self.add_out_projection:
            self.out_projection = nn.Linear(
                d_model * 2,
                d_model * 2,
                device=init_device,
                bias=True,
            )
        else:
            self.out_projection = None

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        for emb in [self.embedding, *self.hash_embeddings]:
            fully_shard(emb, reshard_after_forward=reshard_after_forward, **fsdp_config)
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, emb).set_unshard_in_backward(False)

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        fully_shard(self.cross_attention, reshard_after_forward=reshard_after_forward, **fsdp_config)
        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)

    def fix_init(self, embedding_init_path, target_embeddings, n_estimate=10_000):
        """Rescale such that the local encoder outputs (given random inputs) have the same mean and std as the provided embeddings."""

        if embedding_init_path is not None:
            # load embedding inits (computed via compute_hash_embedding_init.py)
            self.embedding.weight.data[:] = distribute_tensor(
                torch.load(Path(embedding_init_path) / "embedding_init.pth"),
                device_mesh=self.embedding.weight.device_mesh,  # type: ignore
                placements=self.embedding.weight.placements,  # type: ignore
            )

            for i, hash_embedding in enumerate(self.hash_embeddings):
                hash_embedding.weight.data[:] = distribute_tensor(  # type: ignore
                    torch.load(Path(embedding_init_path) / f"hash_embedding_init_{i}.pth"),
                    device_mesh=hash_embedding.weight.device_mesh,  # type: ignore
                    placements=hash_embedding.weight.placements,  # type: ignore
                )

        # .std not supported for DTensor
        te_mean = target_embeddings.mean(0)
        te_std = target_embeddings.var(0).sqrt()

        device = target_embeddings.device
        dummy_input = torch.randint(0, self.embedding.weight.shape[0], (n_estimate,), device=device).unsqueeze(0)
        patch_lens = torch.ones((1, n_estimate), dtype=torch.long, device=dummy_input.device)
        patch_ids = torch.arange(n_estimate, device=dummy_input.device).unsqueeze(0)

        # this is annoying but didn't find a better way to make it compatible with FSDP2
        local_encoder_copy = LocalEncoder(
            vocab_size=self.vocab_size,
            hash_byte_group_size=self.hash_byte_group_size,
            hash_byte_group_vocab=self.hash_byte_group_vocab,
            hash_byte_group_nb_functions=self.hash_byte_group_nb_functions,
            sliding_window_size=self.sliding_window_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            cross_attn_n_heads=self.cross_attn_n_heads,
            block_config=self.block_config,
            add_out_projection=self.add_out_projection,
            init_device=device,
        )
        local_encoder_copy.load_state_dict({
            k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in self.state_dict().items()
        })

        _, h_patch = local_encoder_copy(
            tokens=dummy_input,
            patch_lens=patch_lens,
            patch_ids=patch_ids,
            cross_attn_mask=None, # fine not to mask since mask does not change out magnitude
        )

        h_patch_mean = h_patch[0].mean(0)
        h_patch_std = h_patch[0].var(0).sqrt()
        h_patch_mean = distribute_tensor(h_patch_mean.detach(), device_mesh=te_mean.device_mesh)
        h_patch_std = distribute_tensor(h_patch_std.detach(), device_mesh=te_std.device_mesh)

        if self.out_projection is None:
            # NOTE: this does not match the output perfectly! should remove.
            # fold target embeddings to local encoder dim
            te_folded = torch.cat([target_embeddings[:, :self.d_model], target_embeddings[:, self.d_model:]], dim=0)
            te_folded_std = te_folded.var(0).sqrt()

            h_patch_folded_std = h_patch.reshape(-1, self.d_model).var(0).sqrt()
            h_patch_folded_std = distribute_tensor(h_patch_folded_std.detach(), device_mesh=te_folded_std.device_mesh)

            # then rescale the last linear layer to get right magnitude out
            self.patch_embedding_projection.weight.data *= (te_folded_std / h_patch_folded_std).unsqueeze(0)
            self.cross_attention.w_out.weight.data *= (te_folded_std / h_patch_folded_std).unsqueeze(0)
        else:
            self.out_projection.weight.data *= (te_std / h_patch_std).unsqueeze(0)
            self.out_projection.bias.data[:] = te_mean - h_patch_mean * (te_std / h_patch_std)

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

    def _cross_attention(
        self,
        h: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask | None,
        reduction: str = "amax",
    ):
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

        # expand seq length by a factor of k (k=2 in BLT released checkpoints)
        # i.e. per patch, conduct k cross attentions (each with h heads)
        # NOTE: the need for an upprojection seems to imply an unwanted information bottleneck?
        patch_embedding_init = self.patch_embedding_projection(reduced_h).reshape(
            reduced_h.shape[0], reduced_h.shape[1] * 2, reduced_h.shape[2]
        )

        # apply cross attention
        residual = self.cross_attention(
            q=patch_embedding_init,
            kv=h,
            mask=cross_attn_mask,
        )

        # residual connection + reshape back into patch length (from patch_length * k)
        # NOTE: BLT applies the residual connection two times (presumably on accident?), so we have to 2x here.
        patch_embedding = (patch_embedding_init * 2 + residual).reshape(
            reduced_h.shape[0], reduced_h.shape[1], -1
        )

        if self.out_projection is not None:
            patch_embedding = self.out_projection(patch_embedding)

        return patch_embedding

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(tokens)
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

        # downsample + cross attn
        patch_embeddings = self._cross_attention(
            h=h,
            patch_lens=patch_lens,
            patch_ids=patch_ids,
            cross_attn_mask=cross_attn_mask,
        )

        return h, patch_embeddings


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
        init_device: str = "cpu",
    ):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.patch_embedding_projection = nn.Linear(
            d_global_model,
            d_model * 2,  # TODO: argument for upsampling factor?
            device=init_device,
            bias=False,
        )

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
            self.cross_attentions[str(block_idx)] = CrossAttention(d_model, cross_attn_n_heads, init_device=init_device)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        for cross_attn in self.cross_attentions.values():
            fully_shard(cross_attn, reshard_after_forward=reshard_after_forward, **fsdp_config)

        # should we shard the patch embedding projection?
        #fully_shard(self.patch_embedding_projection, reshard_after_forward=reshard_after_forward, **fsdp_config)
        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)

    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        cross_attn_mask: BlockMask,
    ) -> torch.Tensor:
        # expand seq length by a factor of k (k=2 in BLT released checkpoints)
        patch_embeds_projected = self.patch_embedding_projection(patch_embeds).reshape(
            patch_embeds.shape[0], patch_embeds.shape[1] * 2, embeds.shape[2]
        )

        h = embeds

        for block_idx in range(self.n_layers):
            cross_attn = self.cross_attentions[str(block_idx)]
            block = self.blocks[str(block_idx)]

            # TODO(benjaminm): do we need mark_dynamic here / in general?
            # NOTE: What about LN before/after cross attn?
            h_cross = cross_attn(q=h, kv=patch_embeds_projected, mask=cross_attn_mask)
            # NOTE: same thing, BLT applies the residual connection two times (presumably on accident?), so we have to 2x here.
            h = h * 2 + h_cross
            h = block(h)

        return h