import math
import pickle
import struct

import numpy as np
import pytest
import torch
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.data.ngram_topk import NgramContextSource
from olmo_core.data.composable import (
    ConcatAndChunkInstanceSource,
    InMemoryTokenSource,
    NgramContextInstanceSource,
)


def _tiny_transformer_config() -> TransformerConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    return TransformerConfig(
        d_model=8,
        vocab_size=6,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.default,
            attention=AttentionConfig(n_heads=2),
            layer_norm=layer_norm,
            feed_forward=FeedForwardConfig(hidden_size=16, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
    )


def test_early_fusion_weighted_unembedding_sum_uses_raw_kn_mass():
    model = _tiny_transformer_config().build(init_device="cpu")
    assert model.lm_head is not None
    model.lm_head.register_parameter(
        "early_fusion_alpha_log",
        nn.Parameter(torch.tensor([math.log(0.5)], dtype=torch.float32)),
    )
    with torch.no_grad():
        model.lm_head.w_out.weight.copy_(
            torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
        )

    ngram_token_ids = torch.tensor([[[1, 3, 0], [2, 5, 0]]], dtype=torch.long)
    ngram_log_probs = torch.log(
        torch.tensor([[[0.25, 0.50, 0.0], [0.10, 0.20, 0.0]]], dtype=torch.float32)
    )

    out = model._compute_early_fusion_ngram_embedding(
        ngram_token_ids,
        ngram_log_probs,
        dtype=torch.float32,
    )

    weights = model.lm_head.w_out.weight
    expected = torch.stack(
        [
            0.5 * (0.25 * weights[1] + 0.50 * weights[3]),
            0.5 * (0.10 * weights[2] + 0.20 * weights[5]),
        ],
        dim=0,
    ).unsqueeze(0)
    assert out.shape == (1, 2, 8)
    torch.testing.assert_close(out, expected)


def test_train_module_registers_early_fusion_alpha_with_optimizer_override():
    model = _tiny_transformer_config().build(init_device="cpu")
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=4,
        max_sequence_length=4,
        optim=AdamWConfig(lr=1e-3),
        early_fusion_ngram=True,
        early_fusion_alpha_init=0.2,
        early_fusion_alpha_lr=1e-4,
        early_fusion_ngram_table_dir="/tmp/nonexistent-ngram-table",
    ).build(model, device=torch.device("cpu"))

    alpha_log = train_module._early_fusion_alpha_log_param()
    torch.testing.assert_close(
        torch.exp(alpha_log.detach()),
        torch.tensor([0.2], dtype=torch.float32),
    )

    alpha_group = next(
        group
        for group in train_module.optim.param_groups
        if any(param is alpha_log for param in group["params"])
    )
    assert alpha_group["lr"] == 1e-4
    assert alpha_group["weight_decay"] == 0.0


def _write_raw_context_index(path, *, vocab_size: int = 32):
    out = path / "forward_index.bin"
    header = struct.pack("<4sIII", b"FIX1", 2, 1, vocab_size)
    header += b"\0" * (64 - len(header))
    prefix_words = np.asarray([[10], [20]], dtype=np.uint32)
    prefix_words_off = 64 + 48
    cont_offsets_off = prefix_words_off + prefix_words.nbytes
    continuations_off = cont_offsets_off
    section = struct.pack(
        "<IIQQQQQ",
        2,
        0,
        prefix_words.shape[0],
        0,
        prefix_words_off,
        cont_offsets_off,
        continuations_off,
    )
    out.write_bytes(header + section + prefix_words.tobytes(order="C"))
    return out


def test_ngram_context_source_uses_raw_prefixes_without_topk(tmp_path):
    _write_raw_context_index(tmp_path)
    source = NgramContextSource(tmp_path, N_max=2)

    assert source.num_contexts == 3
    np.testing.assert_array_equal(
        source.lookup_batch([(10,), (20,), (30,), (1, 10)]),
        np.asarray([1, 2, 0, 1], dtype=np.int64),
    )


def test_ngram_context_source_rejects_topk_fallback(tmp_path):
    topk = tmp_path / "forward_index_topk.bin"
    topk.write_bytes(b"not-a-real-index")

    with pytest.raises(FileNotFoundError, match="does not fall back"):
        NgramContextSource(tmp_path, N_max=2)

    with pytest.raises(ValueError, match="requires the raw forward_index.bin"):
        NgramContextSource(topk, N_max=2)


def test_opened_ngram_context_source_pickles_without_mmap(tmp_path):
    _write_raw_context_index(tmp_path)
    source = NgramContextSource(tmp_path, N_max=2)
    assert source.num_contexts == 3

    restored = pickle.loads(pickle.dumps(source))
    assert restored.num_contexts == 3
    assert source._idx is not None
    assert restored._idx is not None


def test_opened_ngram_context_instance_source_pickles_without_lookup(tmp_path):
    _write_raw_context_index(tmp_path)
    base = ConcatAndChunkInstanceSource(
        InMemoryTokenSource([10, 20, 30, 10], work_dir=tmp_path),
        sequence_length=4,
        work_dir=tmp_path,
    )
    source = NgramContextInstanceSource(
        base,
        table_dir=tmp_path,
        N_max=2,
        work_dir=tmp_path,
    )
    inst = source[0]
    np.testing.assert_array_equal(
        inst["engram_context_ids"],
        np.asarray([1, 2, 0, 1], dtype=np.int64),
    )
    assert source._lookup is not None

    restored = pickle.loads(pickle.dumps(source))
    assert restored._lookup is None
    np.testing.assert_array_equal(
        restored[0]["engram_context_ids"],
        np.asarray([1, 2, 0, 1], dtype=np.int64),
    )


def test_engram_early_fusion_sparse_decode_uses_learned_topm_distribution():
    model = _tiny_transformer_config().build(init_device="cpu")
    assert model.lm_head is not None
    model.lm_head.register_parameter(
        "early_fusion_alpha_log",
        nn.Parameter(torch.tensor([math.log(0.5)], dtype=torch.float32)),
    )
    model.lm_head.early_fusion_engram_context_codes = nn.Embedding(3, 2)
    model.lm_head.early_fusion_engram_token_decoder = nn.Embedding(6, 2)
    model.lm_head.early_fusion_engram_top_m = 2
    model.lm_head.early_fusion_engram_vocab_chunk_size = 3

    with torch.no_grad():
        model.lm_head.w_out.weight.copy_(
            torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
        )
        model.lm_head.early_fusion_engram_context_codes.weight.zero_()
        model.lm_head.early_fusion_engram_context_codes.weight[1] = torch.tensor(
            [1.0, 0.0]
        )
        model.lm_head.early_fusion_engram_token_decoder.weight.copy_(
            torch.tensor(
                [
                    [-2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [-1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )

    out = model._compute_early_fusion_engram_embedding(
        torch.tensor([[1]], dtype=torch.long),
        dtype=torch.float32,
    )

    probs = torch.softmax(torch.tensor([3.0, 2.0]), dim=0)
    weights = model.lm_head.w_out.weight
    expected = 0.5 * (probs[0] * weights[3] + probs[1] * weights[4])
    assert out.shape == (1, 1, 8)
    torch.testing.assert_close(out.squeeze(0).squeeze(0), expected)

    out.sum().backward()
    assert model.lm_head.early_fusion_engram_context_codes.weight.grad is not None
    assert model.lm_head.early_fusion_engram_token_decoder.weight.grad is not None


def test_train_module_registers_engram_memory_and_alpha_override(tmp_path):
    _write_raw_context_index(tmp_path, vocab_size=6)
    model = _tiny_transformer_config().build(init_device="cpu")
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=4,
        max_sequence_length=4,
        optim=AdamWConfig(lr=1e-3),
        early_fusion_engram=True,
        early_fusion_engram_alpha_init=0.3,
        early_fusion_engram_alpha_lr=1e-4,
        early_fusion_engram_table_dir=str(tmp_path),
        early_fusion_engram_N_max=2,
        early_fusion_engram_code_dim=2,
        early_fusion_engram_top_m=2,
        early_fusion_engram_vocab_chunk_size=3,
    ).build(model, device=torch.device("cpu"))

    alpha_log = train_module._early_fusion_alpha_log_param()
    torch.testing.assert_close(
        torch.exp(alpha_log.detach()),
        torch.tensor([0.3], dtype=torch.float32),
    )

    context_codes = train_module._early_fusion_engram_context_codes()
    token_decoder = train_module._early_fusion_engram_token_decoder()
    assert context_codes.num_embeddings == 3
    assert context_codes.embedding_dim == 2
    assert token_decoder.num_embeddings == 6
    assert token_decoder.embedding_dim == 2
    assert train_module.early_fusion_engram_num_contexts == 3

    alpha_group = next(
        group
        for group in train_module.optim.param_groups
        if any(param is alpha_log for param in group["params"])
    )
    assert alpha_group["lr"] == 1e-4
    assert alpha_group["weight_decay"] == 0.0
