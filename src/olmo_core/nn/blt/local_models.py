from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from olmo_core.config import Config
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
        )


@dataclass
class LocalDecoderConfig(Config):
    def build(self) -> nn.Module:
        return LocalDecoder()


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
        kv = self.kv_norm(kv)

        q = self.w_q(q_norm)
        k = self.w_k(kv)
        v = self.w_v(kv)

        output_shape = q.shape
        # B S D -> B H S D
        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
    
        # TODO: flex_attention is wrapped in torch.compile in BLT. do the same?
        output = flex_attention(q, k, v, block_mask=mask)
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
            init_device: str = "cpu",
    ):
        super().__init__()
        self.hash_byte_group_size = hash_byte_group_size
        self.hash_byte_group_vocab = hash_byte_group_vocab
        self.hash_byte_group_nb_functions = hash_byte_group_nb_functions
        self.sliding_window_size = sliding_window_size
        self.d_model = d_model
        self.n_layers = n_layers

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
        )

        self.cross_attention = CrossAttention(d_model, cross_attn_n_heads, init_device=init_device)

    def _cross_attention(
        self,
        h: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask,
        reduction: str = "amax",
    ):
        # downsample h
        reduced_h = torch.zeros(
            (h.shape[0], patch_lens.shape[-1], h.shape[-1]), dtype=h.dtype, device=h.device
        )
        # NOTE: this seems slow? particularly due to `.expand(-1, -1, h.shape[-1])`
        reduced_h = reduced_h.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
            reduce=reduction,
            include_self=False,
        )

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
        patch_embedding = (patch_embedding_init + residual).reshape(
            reduced_h.shape[0], reduced_h.shape[1], -1
        )

        return patch_embedding

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        cross_attn_mask: BlockMask,
    ) -> torch.Tensor:
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

        return patch_embeddings

class LocalDecoder(nn.Module):
    def __init__(self):
        super().__init__()