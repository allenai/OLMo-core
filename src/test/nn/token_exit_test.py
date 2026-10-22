"""Token exit on top of the positional-block input drop (olmo_core.nn.token_exit): the tokens the
I-rand / I-fl keep rule drops run blocks [0, L) on the full row and leave at the input of block L.

CPU tests use a tiny OLMo 3-style stack (sliding window W=16 on layers 0-2 and 4-6, full attention
on 3 and 7, like OLMo 3 7B's full layers at 3, 7, ...) on short EOS-led rows with 16-token blocks.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, SlidingWindowAttentionConfig
from olmo_core.nn.attention import position_window as pw
from olmo_core.nn.attention import _causal_attention_positions
from olmo_core.nn.token_exit import prefix_flops_per_token, token_exit_example_flops
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.callbacks.flop_meter import FlopMeterCallback

VOCAB = 256
EOS = 255
PLACEHOLDER = 254
IGN = -100
W = 16
BLOCK = 16
T = 1 + 8 * BLOCK  # leading EOS + 8 blocks
N_LAYERS = 8
SWA_LAYERS = (0, 1, 2, 4, 5, 6)

IFL = {"keep_prob": 0.0, "keep_token_rule": "first_last", "keep_token_k": 0.25}
IRAND = {"keep_prob": 0.4}


def _rows(B, T=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 250, (B, T), generator=g)
    ids[:, 0] = EOS
    return ids


def _labels(ids):
    lab = torch.full_like(ids, IGN)
    lab[:, :-1] = ids[:, 1:]
    return lab


def _model(rule=None, *, n_layers=N_LAYERS, exit_layer=None, device="cpu", **kw):
    torch.manual_seed(0)
    cfg = TransformerConfig.olmo3_30M(
        vocab_size=VOCAB,
        n_layers=n_layers,
        sliding_window=SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
            pattern=[W, W, W, -1],
        ),
        attn_backend=AttentionBackendName.torch,
    )
    model = cfg.build(init_device=device)
    model.init_weights(device=torch.device(device))
    model = model.float()
    model.enable_pooled_soft_tokens(
        -5,
        -6,
        EOS,
        placeholder_id=PLACEHOLDER,
        drop_slots=True,
        positional_block_size=BLOCK,
        **(rule if rule is not None else IFL),
        **kw,
    )
    if exit_layer is not None:
        model.enable_token_exit(exit_layer)
    return model.train()


def _loss(model, ids):
    return model(ids, labels=_labels(ids), ignore_index=IGN).loss


def test_swa_pattern_is_olmo3_like():
    model = _model()
    swa = [i for i, b in model.blocks.items() if b.attention.window_size]
    assert tuple(int(i) for i in swa) == SWA_LAYERS


@pytest.mark.parametrize("rule", [IFL, IRAND], ids=["ifl", "irand"])
def test_exit_layer_zero_matches_input_drop(rule):
    """L=0 is the input-drop path: same loss (fp32) and same gradients."""
    ids = _rows(2)
    model = _model(rule)
    ref = _loss(model, ids)
    ref.backward()
    ref_g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.enable_token_exit(0)
    out = _loss(model, ids)
    out.backward()
    assert abs(float(out) - float(ref)) <= 1e-5, (float(out), float(ref))
    for n, p in model.named_parameters():
        if n in ref_g:
            # the unused slot projector gets an all-zero grad on the input-drop path, None here
            g = p.grad if p.grad is not None else torch.zeros_like(ref_g[n])
            torch.testing.assert_close(g, ref_g[n], atol=1e-5, rtol=1e-4)
    # the keep set really drops tokens on this row (else the test is vacuous)
    assert 0.0 < model._token_exit.last["keep_frac"] < 0.8


@pytest.mark.parametrize("exit_layer", [0, 2, 4, 7])
def test_keep_all_matches_dense(exit_layer):
    ids = _rows(2, seed=1)
    model = _model({"keep_prob": 1.0}, exit_layer=exit_layer)
    out = _loss(model, ids)
    pst, tx = model._pooled_soft_tokens, model._token_exit
    model._pooled_soft_tokens, model._token_exit = None, None
    dense = _loss(model, ids)
    model._pooled_soft_tokens, model._token_exit = pst, tx
    assert abs(float(out) - float(dense)) <= 1e-5, (float(out), float(dense))
    assert tx.last["keep_frac"] == 1.0


@pytest.mark.parametrize("rule", [IFL, IRAND], ids=["ifl", "irand"])
def test_exit_changes_loss_and_trains_on_kept_labels(rule):
    """L > 0 differs from the input drop (dropped tokens are seen by the first L blocks) and the
    loss is the mean over the KEPT tokens' labels (the compacted label count)."""
    ids = _rows(1, seed=2)
    model = _model(rule)
    l0 = float(_loss(model, ids))
    n_labels_drop = model._soft_token_compaction["labels_out"]
    model.enable_token_exit(4)
    l4 = float(_loss(model, ids))
    assert abs(l4 - l0) > 1e-4
    st = model._soft_token_compaction
    assert st["labels_out"] == 2 * n_labels_drop  # the same kept labels, counted twice
    kept = model._token_exit.last["keep_frac"]
    assert 0.0 < kept < 0.8
    # loss == mean CE over the kept tokens' labels: rebuild it from the logits of the same forward
    cb = model._compact_pooled_soft_tokens(ids, _labels(ids), IGN)[0]
    logits = model(ids, labels=_labels(ids), ignore_index=IGN, return_logits=True).logits
    want = torch.nn.functional.cross_entropy(
        logits[0].float(), cb.labels[0], ignore_index=IGN, reduction="mean"
    )
    assert logits.shape[1] == int(cb.row_lens[0])
    assert abs(float(want) - l4) <= 1e-5


def _dropped_mask(model, ids):
    cb = model._compact_pooled_soft_tokens(ids, _labels(ids), IGN)[0]
    kept = torch.zeros(ids.shape, dtype=torch.bool)
    for b in range(ids.shape[0]):
        kept[b, cb.position_ids[b, : int(cb.row_lens[b])]] = True
    return ~kept


@pytest.mark.parametrize("exit_layer", [0, 1, 4])
def test_gradient_reaches_dropped_tokens_through_first_layers(exit_layer):
    """Dropped tokens get gradient on their embeddings and on the outputs of blocks < L - 1 (they
    are keys/values for the kept tokens there), and none on block L - 1's output (they leave)."""
    ids = _rows(1, seed=3)
    model = _model(IFL, exit_layer=exit_layer)
    dropped = _dropped_mask(model, ids)[0]
    assert dropped.any()
    outs = {}

    def keep(name):
        def hook(_m, _i, o):
            o.retain_grad()
            outs[name] = o

        return hook

    model.embeddings.register_forward_hook(keep("emb"))
    for i, blk in model.blocks.items():
        blk.register_forward_hook(keep(int(i)))
    _loss(model, ids).backward()
    emb_g = outs["emb"].grad[0].abs().sum(-1)
    if exit_layer == 0:
        assert float(emb_g[dropped].max()) == 0.0
    else:
        assert bool((emb_g[dropped] > 0).all())
        for j in range(exit_layer - 1):
            assert outs[j].shape[1] == T  # full row before the exit
            assert bool((outs[j].grad[0].abs().sum(-1)[dropped] > 0).all()), j
        last = outs[exit_layer - 1].grad[0].abs().sum(-1)
        assert float(last[dropped].max()) == 0.0
        assert bool((last[~dropped] > 0).any())
    # blocks >= L ran on the compacted row
    n_kept = int((~dropped).sum())
    assert outs[exit_layer].shape[1] == n_kept


@pytest.mark.parametrize("exit_layer", [0, 2, 4])
def test_post_exit_swa_layers_window_by_original_position(exit_layer):
    """Sliding-window layers after the exit take the position-window kernel (gapped
    position_ids); the ones before it run on the full contiguous row on the normal backend. The
    position-window result matches the naive (causal-by-index, window-by-position) reference."""
    from test.nn.attention.position_window_test import _naive_reference  # type: ignore

    ids = _rows(1, seed=4)
    model = _model(IFL, exit_layer=exit_layer)
    calls = []

    def ref_attn(q, k, v, position_ids, window, *, cu_doc_lens=None, scale=None):
        calls.append(q.shape[1])
        return _naive_reference(q, k, v, position_ids, window).to(q.dtype).contiguous()

    with torch.no_grad():
        out = float(_loss(model, ids))
        with mock.patch.object(pw, "position_window_attention", side_effect=ref_attn):
            ref = float(_loss(model, ids))
    want = sum(1 for i in SWA_LAYERS if i >= exit_layer)
    assert len(calls) == want
    n_kept = int((~_dropped_mask(model, ids)).sum())
    assert all(c == n_kept for c in calls)
    assert abs(out - ref) <= 1e-5


def test_enable_token_exit_refusals():
    cfg = TransformerConfig.olmo3_30M(vocab_size=VOCAB, n_layers=2)
    model = cfg.build(init_device="cpu")
    with pytest.raises(OLMoConfigurationError):
        model.enable_token_exit(1)  # no soft-token keep rule
    model.enable_pooled_soft_tokens(
        -5, -6, EOS, placeholder_id=PLACEHOLDER, positional_block_size=BLOCK, **IFL
    )
    with pytest.raises(OLMoConfigurationError):
        model.enable_token_exit(1)  # slots
    model2 = cfg.build(init_device="cpu")
    model2.enable_pooled_soft_tokens(
        -5, -6, EOS, placeholder_id=PLACEHOLDER, drop_slots=True, positional_block_size=BLOCK, **IFL
    )
    with pytest.raises(OLMoConfigurationError):
        model2.enable_token_exit(2)  # outside the stack
    model2.enable_token_exit(1)


def test_eval_forward_is_untouched():
    ids = _rows(1, seed=5)
    model = _model(IFL, exit_layer=4).eval()
    with torch.no_grad():
        a = model(ids)
        model._pooled_soft_tokens, model._token_exit = None, None
        b = model(ids)
    assert a.shape == (1, T, VOCAB)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


# --------------------------------------------------------------------------------------------------
# FLOP accounting
# --------------------------------------------------------------------------------------------------


def _analytic_example_flops(model, full_len, kept_len, exit_layer):
    """Independent derivation: 6 x params per token for every linear layer, 12 x heads x head_dim
    x (causal / sliding-window query-key pairs) / length for attention scores, blocks [0, L) at the
    full length, blocks [L, n) and the LM head at the kept length."""

    def block(blk, n):
        a = blk.attention
        params = sum(p.numel() for p in a.parameters()) + sum(
            p.numel() for p in blk.feed_forward.parameters()
        )
        pairs = _causal_attention_positions(n, a.window_size or None)
        return 6 * params + 12 * a.n_heads * a.head_dim * pairs // n

    lm = 6 * sum(p.numel() for p in model.lm_head.parameters())
    tot = 0
    for i, blk in enumerate(model.blocks.values()):
        n = full_len if i < exit_layer else kept_len
        tot += n * block(blk, n)
    return tot + kept_len * lm


@pytest.mark.parametrize("exit_layer", [0, 1, 4, 7])
@pytest.mark.parametrize("lens", [(129, 33), (32768, 8193), (32768, 6554)])
def test_token_exit_flops_match_analytic_formula(exit_layer, lens):
    model = _model(IFL)
    full, kept = lens
    got = token_exit_example_flops(model, full, kept, exit_layer)
    want = _analytic_example_flops(model, full, kept, exit_layer)
    assert got == float(want)
    if exit_layer == 0:
        assert got == kept * model.num_flops_per_token(kept)
    # prefix + suffix at equal lengths = the dense charge
    assert token_exit_example_flops(model, full, full, exit_layer) == full * model.num_flops_per_token(
        full
    )
    assert prefix_flops_per_token(model, full, 0) == 0


def _fake_trainer(model, gbs):
    metrics = {}
    tr = SimpleNamespace(
        train_module=SimpleNamespace(model=model),
        global_batch_size=gbs,
        global_step=1,
        record_metric=lambda k, v: metrics.__setitem__(k, v),
    )
    return tr, metrics


@pytest.mark.parametrize("exit_layer", [0, 4])
def test_flop_meter_charges_prefix_at_full_length(exit_layer):
    """End to end: one training forward of the token-exit model, then FlopMeterCallback.post_step
    (per-example accounting, the trainer default) == the analytic formula."""
    ids = _rows(2, seed=6)
    model = _model(IFL, exit_layer=exit_layer)
    model._flop_pad_id = EOS
    _loss(model, ids)
    kept = [int(x) for x in model._flop_example_lens["actual"]]
    assert model._flop_example_lens["dense"] == [T, T]
    meter = FlopMeterCallback(seq_len=T, pad_id=EOS, per_example=True)
    tr, metrics = _fake_trainer(model, gbs=2 * T)
    meter.trainer = tr  # type: ignore[assignment]
    meter.post_step()
    dense = 2 * T * model.num_flops_per_token(T)
    want = sum(_analytic_example_flops(model, T, k, exit_layer) for k in kept)
    assert metrics["flop_meter/step_actual_over_dense"] == pytest.approx(want / dense, rel=1e-12)
    assert meter.summary()["token_exit_layer"] == exit_layer


def test_flop_meter_compaction_branch_charges_prefix():
    """The non-per-example accounting (attention at the mean compacted length) adds the prefix
    blocks at the full length too."""
    ids = _rows(2, seed=7)
    model = _model(IFL, exit_layer=4)
    _loss(model, ids)
    comp = dict(model._soft_token_compaction)
    meter = FlopMeterCallback(seq_len=T)
    tr, metrics = _fake_trainer(model, gbs=2 * T)
    meter.trainer = tr  # type: ignore[assignment]
    meter.post_step()
    out_len = comp["tokens_out"] // comp["rows"]
    want = comp["tokens_out"] * (
        model.num_flops_per_token(out_len) - prefix_flops_per_token(model, out_len, 4)
    ) + comp["tokens_in"] * prefix_flops_per_token(model, T, 4)
    assert meter._actual == pytest.approx(want, rel=1e-12)


# --------------------------------------------------------------------------------------------------
# GPU: the real kernels (flash-2 + flex position window), bf16, 32k rows
# --------------------------------------------------------------------------------------------------


def _gpu_model(rule, exit_layer):
    from olmo_core.config import DType
    from olmo_core.nn.rope import YaRNRoPEScalingConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo3_190M(vocab_size=100352, n_layers=8, dtype=DType.bfloat16)
    cfg = cfg.with_rope_scaling(
        YaRNRoPEScalingConfig(factor=4.0, beta_fast=32, beta_slow=1, old_context_len=8192),
        full_attn_layers_only=True,
    )
    model = cfg.build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    model.enable_pooled_soft_tokens(
        -5, -6, 100257, placeholder_id=100300, drop_slots=True, positional_block_size=512, **rule
    )
    if exit_layer is not None:
        model.enable_token_exit(exit_layer)
    return model.train()


def _rows32k(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 50000, (1, 32768), generator=g)
    ids[:, 0] = 100257
    return ids.cuda()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "rule", [IFL, {"keep_prob": 0.2}], ids=["ifl", "irand"]
)
def test_gpu_olmo3_190m_exit_parity_and_two_steps(rule):
    """olmo3_190M-style (8 layers: SWA 4096 on 0-2/4-6, full on 3 and 7, flash-2, YaRN x4 on the
    full layers) on one 32k EOS-led row: L=0 == input drop (bf16), keep-all + L=4 == dense, and
    two AdamW steps with L=4 give finite losses."""
    ids = _rows32k(0)
    lab = _labels(ids)
    m = _gpu_model(rule, None)
    with torch.no_grad():
        ref = float(m(ids, labels=lab, ignore_index=IGN).loss)
        m.enable_token_exit(0)
        l0 = float(m(ids, labels=lab, ignore_index=IGN).loss)
        m.enable_token_exit(4)
        l4 = float(m(ids, labels=lab, ignore_index=IGN).loss)
    print(f"[gpu-tokexit] input-drop {ref:.5f}  L=0 {l0:.5f}  L=4 {l4:.5f}")
    assert abs(l0 - ref) <= 2e-3
    assert torch.isfinite(torch.tensor(l4))

    mk = _gpu_model({"keep_prob": 1.0}, 4)
    with torch.no_grad():
        lk = float(mk(ids, labels=lab, ignore_index=IGN).loss)
        mk._pooled_soft_tokens, mk._token_exit = None, None
        ld = float(mk(ids, labels=lab, ignore_index=IGN).loss)
    assert abs(lk - ld) <= 1e-3, (lk, ld)

    m2 = _gpu_model(rule, 4)
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    losses = []
    for step in range(2):
        x = _rows32k(step + 1)
        out = m2(x, labels=_labels(x), ignore_index=IGN)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
    print(f"[gpu-tokexit] L=4 two steps: {losses} keep={m2._token_exit.last['keep_frac']:.4f}")
    assert all(torch.isfinite(torch.tensor(losses)))
