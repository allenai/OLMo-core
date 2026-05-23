import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.data.sb_bias import apply_sb_bias_inplace, compute_sparse_sb_ce_loss
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule


def test_sparse_sb_ce_matches_materialized_bias():
    torch.manual_seed(1234)
    B, S, V = 2, 4, 11
    poe_lambda = 0.3
    ignore_index = -100

    logits = torch.randn(B, S, V, dtype=torch.float32)
    labels = torch.tensor(
        [
            [1, 2, ignore_index, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    unigram_floor = torch.log_softmax(torch.randn(V, dtype=torch.float32), dim=0)

    bidx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    pos = torch.tensor([0, 0, 2, 1, 1, 3], dtype=torch.long)
    tok = torch.tensor([1, 3, 4, 6, 10, 8], dtype=torch.long)
    # Deliberately include overrides both above and below the unigram floor.
    sb_log_score = unigram_floor[tok] + torch.tensor(
        [0.7, -0.4, 1.2, 0.3, -0.9, 0.5],
        dtype=torch.float32,
    )

    materialized = logits.clone()
    apply_sb_bias_inplace(
        materialized,
        unigram_floor,
        bidx,
        pos,
        tok,
        sb_log_score,
        poe_lambda,
    )
    expected = F.cross_entropy(
        materialized.flatten(0, 1),
        labels.flatten(),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, S)

    actual = compute_sparse_sb_ce_loss(
        logits,
        labels,
        unigram_floor,
        bidx,
        pos,
        tok,
        sb_log_score,
        poe_lambda,
        ignore_index,
    )

    torch.testing.assert_close(actual, expected)


def test_apply_sb_bias_allows_lambda_gradient():
    torch.manual_seed(5678)
    B, S, V = 2, 3, 7
    logits = torch.randn(B, S, V, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    unigram_floor = torch.log_softmax(torch.randn(V, dtype=torch.float32), dim=0)
    bidx = torch.tensor([0, 1], dtype=torch.long)
    pos = torch.tensor([1, 2], dtype=torch.long)
    tok = torch.tensor([2, 6], dtype=torch.long)
    sb_log_score = unigram_floor[tok] + torch.tensor([0.4, -0.8], dtype=torch.float32)
    poe_lambda = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)

    biased_logits = logits.float().clone()
    apply_sb_bias_inplace(
        biased_logits,
        unigram_floor,
        bidx,
        pos,
        tok,
        sb_log_score,
        poe_lambda,
    )
    loss = F.cross_entropy(biased_logits.flatten(0, 1), labels.flatten())
    loss.backward()

    assert poe_lambda.grad is not None
    assert torch.isfinite(poe_lambda.grad)
    assert poe_lambda.grad.abs() > 0


def test_train_module_sb_loss_allows_learned_lambda_gradient():
    torch.manual_seed(9012)
    B, S, V = 2, 3, 7
    logits = torch.randn(B, S, V, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    unigram_floor = torch.log_softmax(torch.randn(V, dtype=torch.float32), dim=0)
    bidx = torch.tensor([0, 1], dtype=torch.long)
    pos = torch.tensor([1, 2], dtype=torch.long)
    tok = torch.tensor([2, 6], dtype=torch.long)
    sb_log_score = unigram_floor[tok] + torch.tensor([0.4, -0.8], dtype=torch.float32)

    module = object.__new__(TransformerTrainModule)
    module.model = nn.Module()
    module._poe_lambda_log_name = "poe_lambda_log"
    module.model.register_parameter(
        module._poe_lambda_log_name,
        nn.Parameter(torch.log(torch.tensor(0.3, dtype=torch.float32))),
    )
    module.poe_lambda = 0.3
    module.poe_lambda_learnable = True
    module.label_ignore_index = -100
    module._get_poe_sb_unigram_floor_dev = lambda dtype: unigram_floor.to(dtype=dtype)

    loss = TransformerTrainModule._compute_poe_loss_sb(
        module,
        logits=logits,
        sb_override_batch_idx=bidx,
        sb_override_position=pos,
        sb_override_token_id=tok,
        sb_override_log_score=sb_log_score,
        labels=labels,
        loss_div_factor=torch.tensor(B * S, dtype=torch.float32),
    )
    loss.backward()

    lambda_grad = module._poe_lambda_log_param().grad
    assert lambda_grad is not None
    assert torch.isfinite(lambda_grad)
    assert lambda_grad.abs() > 0
