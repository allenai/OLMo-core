"""Standalone, readable PyTorch implementation of the OLMoE3 3.5B-active rung.

Setup in an empty directory after copying this file there::

    python -m venv .venv
    . .venv/bin/activate
    pip install torch
    python standalone_model.py

Only Python 3.12+ and PyTorch are required. No OLMo-core checkout, Triton,
FlashAttention, FLA, or NVSHMEM is needed.

This file mirrors the model path without importing
``olmo_core``.  It intentionally omits distributed execution, FP8, fused kernels,
checkpointing, metrics, and training orchestration; those change execution, not the
model layers.  The KDA recurrence and expert dispatch below are unfused reference
implementations and are meant for inspection and small-shape tests, not production.

Importing this module creates ``largest_model`` on the meta device.  This instantiates
the complete 62.86B-parameter module hierarchy without allocating parameter storage.
Call ``OLMoE3(largest_config, device=...)`` only on appropriately sharded hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class OLMoE3Config:
    vocab_size: int = 100_352
    d_model: int = 1792
    n_layers: int = 30
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 256
    expert_hidden_size: int = 1792
    num_routed_experts: int = 512
    top_k: int = 16
    latent_dim: int = 768
    kda_expand_v: float = 2.0
    kda_conv_size: int = 4
    rms_norm_eps: float = 1e-6
    kda_norm_eps: float = 1e-5
    init_std: float = 0.02
    embed_scale: float = math.sqrt(1792)
    tie_word_embeddings: bool = False
    """
    EMO settings: https://arxiv.org/abs/2605.06663
    """
    emo_enabled: bool = True
    global_load_balancing: bool = True
    emo_eos_token_id: int = 100_257
    emo_min_document_expert_pool: int = 16
    emo_max_document_expert_pool: int = 512
    emo_eval_document_expert_pool: int = 512

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(range(4, self.n_layers, 5))

    def validate(self) -> None:
        # TPU MXU alignment (256x256 systolic array): every reduction/output
        # dimension that appears in a GEMM must be a multiple of 256, or the
        # matmul pads out to the next tile and burns cycles on zeros.
        for name, dim in (
            ("d_model", self.d_model),
            ("head_dim", self.head_dim),
            ("expert_hidden_size", self.expert_hidden_size),
            ("latent_dim", self.latent_dim),
        ):
            assert dim % 256 == 0, f"{name}={dim} must be a multiple of 256 for MXU alignment"
        assert self.n_heads % self.n_kv_heads == 0
        assert self.top_k <= self.num_routed_experts
        if self.emo_enabled:
            assert self.emo_min_document_expert_pool >= self.top_k
            assert self.emo_max_document_expert_pool <= self.num_routed_experts
            assert self.top_k <= self.emo_eval_document_expert_pool <= self.num_routed_experts

    @property
    def num_embedding_params(self) -> int:
        """Token-embedding parameters, using OLMo-core's definition."""
        return self.d_model * self.vocab_size

    @property
    def num_inactive_routed_expert_params(self) -> int:
        """Stored routed-expert weights not selected by top-k for one token."""
        params_per_expert = 3 * self.latent_dim * self.expert_hidden_size
        # Layer zero is dense; every remaining block contains routed experts.
        return (self.n_layers - 1) * (self.num_routed_experts - self.top_k) * params_per_expert


class RMSNorm(nn.Module):
    """OLMo-core RMSNorm: learned scale, no bias, fp32 variance."""

    def __init__(self, size: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU used by both shared and routed experts."""

    def __init__(self, d_in: int, hidden: int, d_out: int, *, device=None, dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype, bias=False)
        self.up = nn.Linear(d_in, hidden, **kw)
        self.gate = nn.Linear(d_in, hidden, **kw)
        self.down = nn.Linear(hidden, d_out, **kw)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, width: int, kernel_size: int, *, device=None, dtype=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(width, 1, kernel_size, device=device, dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor, segment_ids: Tensor | None = None) -> Tensor:
        channels_first = x.transpose(1, 2)
        if segment_ids is None:
            y = F.conv1d(
                channels_first,
                self.weight,
                groups=x.shape[-1],
                padding=self.kernel_size - 1,
            )[..., : x.shape[1]]
        else:
            # Materialize causal windows and mask tokens from preceding packed
            # documents. This is the unfused equivalent of passing cu_seqlens to
            # OLMo-core's packed-document convolution kernel.
            windows = F.pad(channels_first, (self.kernel_size - 1, 0)).unfold(
                2, self.kernel_size, 1
            )
            padded_segments = F.pad(segment_ids, (self.kernel_size - 1, 0), value=-1)
            segment_windows = padded_segments.unfold(1, self.kernel_size, 1)
            same_document = segment_windows.eq(segment_ids.unsqueeze(-1)).unsqueeze(1)
            kernel = self.weight[:, 0].view(1, x.shape[-1], 1, self.kernel_size)
            y = (windows * same_document * kernel).sum(-1)
        return F.silu(y.transpose(1, 2))


class KimiDeltaAttention(nn.Module):
    """Unfused reference for OLMo-core's vector-decay Kimi Delta Attention."""

    def __init__(self, cfg: OLMoE3Config, *, device=None, dtype=None):
        super().__init__()
        h, k, v = cfg.n_heads, cfg.head_dim, int(cfg.head_dim * cfg.kda_expand_v)
        self.n_heads, self.head_k_dim, self.head_v_dim = h, k, v
        key_width, value_width = h * k, h * v
        kw = dict(device=device, dtype=dtype, bias=False)
        self.w_q = nn.Linear(cfg.d_model, key_width, **kw)
        self.w_k = nn.Linear(cfg.d_model, key_width, **kw)
        self.w_v = nn.Linear(cfg.d_model, value_width, **kw)
        self.q_conv = CausalDepthwiseConv1d(key_width, cfg.kda_conv_size, device=device, dtype=dtype)
        self.k_conv = CausalDepthwiseConv1d(key_width, cfg.kda_conv_size, device=device, dtype=dtype)
        self.v_conv = CausalDepthwiseConv1d(value_width, cfg.kda_conv_size, device=device, dtype=dtype)
        self.f_proj_1 = nn.Linear(cfg.d_model, v, **kw)
        self.f_proj_2 = nn.Linear(v, key_width, **kw)
        self.w_b = nn.Linear(cfg.d_model, h, **kw)
        self.A_log = nn.Parameter(torch.empty(h, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(key_width, device=device, dtype=torch.float32))
        self.g_proj_1 = nn.Linear(cfg.d_model, v, **kw)
        self.g_proj_2 = nn.Linear(v, value_width, device=device, dtype=dtype, bias=True)
        self.o_norm = RMSNorm(v, cfg.kda_norm_eps, device=device, dtype=dtype)
        self.w_out = nn.Linear(value_width, cfg.d_model, **kw)
        nn.init.uniform_(self.A_log, 1, 16)
        with torch.no_grad():
            self.A_log.log_()
        nn.init.zeros_(self.dt_bias)

    def forward(self, x: Tensor, segment_ids: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        h, dk, dv = self.n_heads, self.head_k_dim, self.head_v_dim
        q = self.q_conv(self.w_q(x), segment_ids).view(b, t, h, dk)
        k = self.k_conv(self.w_k(x), segment_ids).view(b, t, h, dk)
        v = self.v_conv(self.w_v(x), segment_ids).view(b, t, h, dv)
        # FLA's KDA kernel L2-normalizes Q/K and applies the standard query
        # scale internally. Keep it explicit in this reference recurrence.
        q = F.normalize(q.float(), dim=-1) * (dk**-0.5)
        k = F.normalize(k.float(), dim=-1)
        raw_g = self.f_proj_2(self.f_proj_1(x)).view(b, t, h, dk).float()
        dt = self.dt_bias.view(1, 1, h, dk)
        decay = torch.exp(-self.A_log.exp().view(1, 1, h, 1) * F.softplus(raw_g + dt))
        beta = 2.0 * self.w_b(x).float().sigmoid()  # allow_neg_eigval=True
        state = x.new_zeros((b, h, dk, dv), dtype=torch.float32)
        outputs: list[Tensor] = []
        for pos in range(t):
            if segment_ids is not None and pos:
                reset = segment_ids[:, pos].ne(segment_ids[:, pos - 1]).view(b, 1, 1, 1)
                state = state.masked_fill(reset, 0)
            state = state * decay[:, pos].unsqueeze(-1)
            prediction = torch.einsum("bhkv,bhk->bhv", state, k[:, pos])
            delta = (v[:, pos].float() - prediction) * beta[:, pos].unsqueeze(-1)
            state = state + torch.einsum("bhk,bhv->bhkv", k[:, pos], delta)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, pos]))
        out = torch.stack(outputs, dim=1).to(x.dtype)
        gate = self.g_proj_2(self.g_proj_1(x)).view(b, t, h, dv).sigmoid()
        out = self.o_norm(out) * gate
        return self.w_out(out.flatten(2))


class GatedNoPEAttention(nn.Module):
    """Causal GQA with per-head QK RMSNorm, no positional embedding, element gate."""

    def __init__(self, cfg: OLMoE3Config, *, device=None, dtype=None):
        super().__init__()
        q_width, kv_width = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        kw = dict(device=device, dtype=dtype, bias=False)
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.w_q = nn.Linear(cfg.d_model, q_width, **kw)
        self.w_k = nn.Linear(cfg.d_model, kv_width, **kw)
        self.w_v = nn.Linear(cfg.d_model, kv_width, **kw)
        self.w_g = nn.Linear(cfg.d_model, q_width, **kw)
        self.w_out = nn.Linear(q_width, cfg.d_model, **kw)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x: Tensor, segment_ids: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.w_q(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.w_k(x).view(b, t, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        repeats = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        mask = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        if segment_ids is not None:
            same_doc = segment_ids[:, :, None].eq(segment_ids[:, None, :])
            mask = mask.view(1, 1, t, t) & same_doc[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.w_out(y * self.w_g(x).sigmoid())


class MoERouter(nn.Module):
    """Shared router mechanics and the ladder's auxiliary balancing losses."""

    def __init__(self, cfg: OLMoE3Config, *, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.num_routed_experts, cfg.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=cfg.init_std, a=-3 * cfg.init_std, b=3 * cfg.init_std)

    def scores_and_logits(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = F.linear(x.float(), self.weight.float())
        return logits.softmax(-1), logits

    def finish_routing(self, scores: Tensor, selected: Tensor) -> tuple[Tensor, Tensor]:
        weights = scores.gather(-1, selected)
        # normalize_expert_weights=1.0 followed by restore_weight_scale=True.
        weights = F.normalize(weights, p=1.0, dim=-1) * self.cfg.top_k
        return weights, selected

    def auxiliary_loss(self, scores: Tensor, logits: Tensor, selected: Tensor) -> Tensor:
        """OLMo-core's load-balancing (0.01) and router-z (1e-5) losses.

        Global balancing uses local-batch counts and all-reduces them over the
        distributed process group. Local balancing uses per-instance counts.
        This choice is independent of whether routing uses EMo document pools.
        """
        b, s, e = scores.shape
        one_hot = F.one_hot(selected, e).sum(-2).to(scores.dtype)
        if self.cfg.global_load_balancing:
            counts = one_hot.sum((0, 1))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                counts = counts.clone()
                torch.distributed.all_reduce(counts)
                counts /= torch.distributed.get_world_size()
            lb = (scores.mean((0, 1)) * counts).sum() / (b * s)
        else:
            counts = one_hot.sum(1)
            lb = (scores.mean(1) * counts).sum() / (b * s)
        lb = (e / self.cfg.top_k) * lb
        z = torch.logsumexp(logits, dim=-1).square().mean()
        return 0.01 * lb + 1e-5 * z


class StandardRouter(MoERouter):
    """Ordinary full-width, per-token softmax top-k router."""

    def forward(self, x: Tensor, segment_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        del segment_ids
        scores, logits = self.scores_and_logits(x)
        selected = scores.topk(self.cfg.top_k, dim=-1).indices
        weights, selected = self.finish_routing(scores, selected)
        self.last_auxiliary_loss = self.auxiliary_loss(scores, logits, selected)
        return weights.to(x.dtype), selected


class EMoRouter(MoERouter):
    """Document-pool EMo router; balancing policy is configured independently."""

    def forward(self, x: Tensor, segment_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        scores, logits = self.scores_and_logits(x)
        if segment_ids is None:
            segment_ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        b, _, e = scores.shape
        document_scores = torch.zeros((b, int(segment_ids.max().item()) + 1, e), device=x.device)
        document_scores.scatter_add_(1, segment_ids[..., None].expand_as(scores), scores)
        if self.training:
            lo, hi = self.cfg.emo_min_document_expert_pool, self.cfg.emo_max_document_expert_pool
            pool_sizes = torch.randint(lo, hi + 1, document_scores.shape[:2], device=x.device)
        else:
            pool_sizes = torch.full(document_scores.shape[:2], self.cfg.emo_eval_document_expert_pool, device=x.device)
        rank = document_scores.argsort(-1, descending=True).argsort(-1)
        keep_by_doc = rank < pool_sizes[..., None]
        keep = keep_by_doc.gather(1, segment_ids[..., None].expand(-1, -1, e))
        selected = scores.masked_fill(~keep, -torch.inf).topk(self.cfg.top_k, dim=-1).indices
        weights, selected = self.finish_routing(scores, selected)
        self.last_auxiliary_loss = self.auxiliary_loss(scores, logits, selected)
        return weights.to(x.dtype), selected


class RoutedExperts(nn.Module):
    def __init__(self, cfg: OLMoE3Config, *, device=None, dtype=None):
        super().__init__()
        e, d, h = cfg.num_routed_experts, cfg.latent_dim, cfg.expert_hidden_size
        self.up = nn.Parameter(torch.empty(e, h, d, device=device, dtype=dtype))
        self.gate = nn.Parameter(torch.empty(e, h, d, device=device, dtype=dtype))
        self.down = nn.Parameter(torch.empty(e, d, h, device=device, dtype=dtype))
        for weight in (self.up, self.gate, self.down):
            nn.init.trunc_normal_(weight, std=cfg.init_std, a=-3 * cfg.init_std, b=3 * cfg.init_std)

    def forward(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        flat_x, flat_i, flat_w = x.flatten(0, 1), indices.flatten(0, 1), weights.flatten(0, 1)
        out = torch.zeros_like(flat_x)
        for expert in flat_i.unique().tolist():
            token, slot = torch.where(flat_i == expert)
            expert_x = flat_x[token]
            hidden = F.linear(expert_x, self.up[expert]) * F.silu(F.linear(expert_x, self.gate[expert]))
            out.index_add_(0, token, F.linear(hidden, self.down[expert]) * flat_w[token, slot, None])
        return out.view_as(x)


class OLMoE3Block(nn.Module):
    def __init__(self, cfg: OLMoE3Config, layer_idx: int, *, device=None, dtype=None):
        super().__init__()
        def norm():
            return RMSNorm(cfg.d_model, cfg.rms_norm_eps, device=device, dtype=dtype)

        self.attn_in_norm, self.attn_out_norm = norm(), norm()
        self.ffn_in_norm, self.ffn_out_norm = norm(), norm()
        self.mixer = (GatedNoPEAttention if layer_idx in cfg.full_attention_layers else KimiDeltaAttention)(cfg, device=device, dtype=dtype)
        self.shared = SwiGLU(cfg.d_model, 8 * cfg.d_model if layer_idx == 0 else cfg.expert_hidden_size, cfg.d_model, device=device, dtype=dtype)
        self.router = self.routed = self.latent_down = self.latent_up = None
        if layer_idx > 0:
            router_type = EMoRouter if cfg.emo_enabled else StandardRouter
            self.router = router_type(cfg, device=device, dtype=dtype)
            self.routed = RoutedExperts(cfg, device=device, dtype=dtype)
            self.latent_down = nn.Linear(cfg.d_model, cfg.latent_dim, bias=False, device=device, dtype=dtype)
            self.latent_up = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor, segment_ids: Tensor | None = None) -> Tensor:
        x = x + self.attn_out_norm(self.mixer(self.attn_in_norm(x), segment_ids))
        ffn_in = self.ffn_in_norm(x)
        ffn_out = self.shared(ffn_in)
        if self.router is not None:
            assert self.routed is not None and self.latent_down is not None and self.latent_up is not None
            weights, indices = self.router(ffn_in, segment_ids)
            ffn_out = ffn_out + self.latent_up(self.routed(self.latent_down(ffn_in), weights, indices))
        return x + self.ffn_out_norm(ffn_out)


class OLMoE3(nn.Module):
    def __init__(self, cfg: OLMoE3Config, *, device=None, dtype=torch.float32):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        self.embedding_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(OLMoE3Block(cfg, i, device=device, dtype=dtype) for i in range(cfg.n_layers))
        self.lm_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        if self.embedding.weight.device.type != "meta":
            self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Reproduce OLMo-core ``InitMethod.normal`` with seed zero.

        Every embedding, projection, router, and expert weight uses a truncated
        normal with std=0.02 and bounds +/-3 std. Linear biases and ``dt_bias``
        are zero; norm scales are one; KDA ``A_log`` is log(U[1, 16]).
        """
        generator = torch.Generator(self.embedding.weight.device).manual_seed(0)

        def trunc_normal(parameter: Tensor) -> None:
            nn.init.trunc_normal_(
                parameter,
                mean=0.0,
                std=self.config.init_std,
                a=-3 * self.config.init_std,
                b=3 * self.config.init_std,
                generator=generator,
            )

        trunc_normal(self.embedding.weight)
        for block in self.blocks:
            for norm in (
                block.attn_in_norm,
                block.attn_out_norm,
                block.ffn_in_norm,
                block.ffn_out_norm,
            ):
                nn.init.ones_(norm.weight)

            mixer = block.mixer
            if isinstance(mixer, KimiDeltaAttention):
                for linear in (
                    mixer.w_q,
                    mixer.w_k,
                    mixer.w_v,
                    mixer.f_proj_1,
                    mixer.f_proj_2,
                    mixer.w_b,
                    mixer.g_proj_1,
                    mixer.g_proj_2,
                ):
                    trunc_normal(linear.weight)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)
                for conv in (mixer.q_conv, mixer.k_conv, mixer.v_conv):
                    trunc_normal(conv.weight)
                nn.init.uniform_(mixer.A_log, 1, 16, generator=generator)
                mixer.A_log.log_()
                nn.init.zeros_(mixer.dt_bias)
                nn.init.ones_(mixer.o_norm.weight)
                trunc_normal(mixer.w_out.weight)
            else:
                assert isinstance(mixer, GatedNoPEAttention)
                for linear in (mixer.w_q, mixer.w_k, mixer.w_v, mixer.w_g, mixer.w_out):
                    trunc_normal(linear.weight)
                nn.init.ones_(mixer.q_norm.weight)
                nn.init.ones_(mixer.k_norm.weight)

            if block.router is not None:
                trunc_normal(block.router.weight)
            for projection in (block.latent_down, block.latent_up):
                if projection is not None:
                    trunc_normal(projection.weight)
            if block.routed is not None:
                for weight in (block.routed.up, block.routed.gate, block.routed.down):
                    trunc_normal(weight)
            for linear in (block.shared.up, block.shared.gate, block.shared.down):
                trunc_normal(linear.weight)

        nn.init.ones_(self.embedding_norm.weight)
        nn.init.ones_(self.lm_norm.weight)
        trunc_normal(self.lm_head.weight)

    def forward(self, input_ids: Tensor, segment_ids: Tensor | None = None) -> Tensor:
        x = self.embedding_norm(self.embedding(input_ids) * self.config.embed_scale)
        for block in self.blocks:
            x = block(x, segment_ids)
        return self.lm_head(self.lm_norm(x))


@dataclass(frozen=True)
class ParameterCounts:
    """The same total/active and embedding splits exposed by OLMo-core configs."""

    total: int
    active: int
    embedding: int
    non_embedding: int
    active_non_embedding: int


def parameter_counts(model: OLMoE3) -> ParameterCounts:
    """Count parameters following ``olmo_core.nn.transformer.TransformerConfig``.

    OLMo-core defines embedding parameters as only the token embedding table.
    Thus embedding RMSNorm and the untied LM head are non-embedding parameters.
    All shared-expert parameters are active, while only ``top_k`` routed experts
    per MoE block are active. Routers and latent projections are also active.
    """
    cfg = model.config
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = cfg.num_embedding_params
    active = total - cfg.num_inactive_routed_expert_params
    return ParameterCounts(
        total=total,
        active=active,
        embedding=embedding,
        non_embedding=total - embedding,
        active_non_embedding=active - embedding,
    )


def print_parameter_counts(model: OLMoE3) -> None:
    counts = parameter_counts(model)
    print(f"total params:                {counts.total:,}")
    print(f"active params:               {counts.active:,}")
    print(f"embedding params:            {counts.embedding:,}")
    print(f"non-embedding params:        {counts.non_embedding:,}")
    print(f"active non-embedding params: {counts.active_non_embedding:,}")


# Largest canonical ladder rung. Meta instantiation is allocation-free but preserves
# every layer, shape, parameter, and the exact KDA/full-attention override pattern.
largest_config = OLMoE3Config()
largest_model = OLMoE3(largest_config, device="meta")


if __name__ == "__main__":
    print(largest_model)
    print_parameter_counts(largest_model)
