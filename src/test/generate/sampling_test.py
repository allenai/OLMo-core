import pytest
import torch

from olmo_core.generate.sampling import (
    greedy_selection,
    select_next_token,
    top_k_filtering,
    top_p_filtering,
)
from olmo_core.utils import seed_all


def test_greedy_selection():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    expected = torch.tensor([2, 0])
    assert torch.equal(greedy_selection(logits), expected)

    logits_3d = torch.randn(2, 4, 8)
    expected_3d = logits_3d.argmax(dim=-1)
    assert torch.equal(greedy_selection(logits_3d), expected_3d)


def test_top_k_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Filter to top 2
    filtered = top_k_filtering(logits, top_k=2)
    expected = torch.tensor([[-float("inf"), -float("inf"), 3.0, 4.0]])
    assert torch.equal(filtered, expected)

    # top_k <= 0 should not filter
    assert torch.equal(top_k_filtering(logits, top_k=0), logits)
    assert torch.equal(top_k_filtering(logits, top_k=-1), logits)

    # top_k > vocab_size should not filter
    assert torch.equal(top_k_filtering(logits, top_k=5), logits)

    # Test with a batch
    logits_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 3.0]])
    filtered_2d = top_k_filtering(logits_2d, top_k=2)
    expected_2d = torch.tensor([[-float("inf"), 2.0, 3.0], [4.0, -float("inf"), 3.0]])
    assert torch.equal(filtered_2d, expected_2d)


def test_top_p_filtering():
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    # Softmax of logits: [0.267, 0.242, 0.219, 0.198, 0.179]
    # For top_p=0.5, we expect to keep the first token (prob 0.267).
    # The code keeps the token that crosses the threshold, so we also keep the second token.
    # The set of kept tokens corresponds to original logits 0.4 and 0.3.
    filtered = top_p_filtering(logits, top_p=0.5)
    expected = torch.tensor([[0.4, 0.3, -float("inf"), -float("inf")]])
    assert torch.equal(filtered, expected)

    # top_p >= 1.0 should not filter
    assert torch.equal(top_p_filtering(logits, top_p=1.0), logits)
    assert torch.equal(top_p_filtering(logits, top_p=1.1), logits)

    # top_p <= 0.0 should not filter
    assert torch.equal(top_p_filtering(logits, top_p=0.0), logits)
    assert torch.equal(top_p_filtering(logits, top_p=-0.1), logits)

    # Test with a batch
    logits_batch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [4.0, 3.0, 2.0, 1.0]])
    filtered_batch = top_p_filtering(logits_batch, top_p=0.8)
    expected_batch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [4.0, 3.0, -float("inf"), -float("inf")]])
    assert torch.equal(filtered_batch, expected_batch)


@pytest.mark.parametrize(
    "logits_data",
    [
        ([[1.0, 2.0, 8.0, 3.0, 4.0]]),
        ([[1.0, 2.0, 8.0, 3.0, 4.0], [5.0, 9.0, 1.0, 2.0, 3.0]]),
    ],
)
def test_select_next_token_greedy(logits_data):
    logits = torch.tensor(logits_data, dtype=torch.float32)

    # Not sampling should be greedy.
    tokens = select_next_token(logits, do_sample=False)
    assert torch.equal(tokens, logits.argmax(dim=-1))

    # Temperature of 0 should be greedy.
    tokens_temp0 = select_next_token(logits, do_sample=True, temperature=0.0)
    assert torch.equal(tokens_temp0, logits.argmax(dim=-1))


def test_select_next_token_with_top_k():
    seed_all(0)
    logits = torch.tensor([[1.0, 2.0, 8.0, 3.0, 4.0, 0.1, 0.2, 0.3]])
    top_k = 3

    token = select_next_token(logits, do_sample=True, top_k=top_k)

    # The selected token must be one of the top-k.
    top_k_indices = logits.topk(top_k).indices
    assert token.item() in top_k_indices.squeeze().tolist()


def test_select_next_token_with_top_p():
    seed_all(0)
    # probs for [8.0, 4.0, 3.0] are ~[0.90, 0.04, 0.01], cumsum ~[0.90, 0.94, 0.95]
    logits = torch.tensor([[1.0, 2.0, 8.0, 3.0, 4.0, 0.1, 0.2, 0.3]])
    top_p = 0.95
    # The nucleus should contain indices {2, 4, 3}
    # which correspond to logits {8.0, 4.0, 3.0}.

    token = select_next_token(logits, do_sample=True, top_p=top_p)
    assert token.item() in [2, 3, 4]


def test_select_next_token_with_top_k_and_top_p():
    seed_all(0)
    logits = torch.tensor([[10.0, 9.0, 1.0, 8.0, 2.0]])

    # top_k=3 -> indices {0, 1, 3} (logits {10, 9, 8})
    # Probs for {10, 9, 8} are ~[0.70, 0.26, 0.03], cumsum ~[0.70, 0.96, 0.99]
    # top_p=0.9 -> nucleus is {0, 1}
    # Intersection of top-k and top-p is {0, 1}.
    token = select_next_token(logits, do_sample=True, top_k=3, top_p=0.9)
    assert token.item() in [0, 1]


def test_select_next_token_with_temperature_top_k_and_top_p():
    seed_all(0)
    logits = torch.tensor([[10.0, 9.0, 1.0, 8.0, 2.0]])
    select_next_token(logits, do_sample=True, temperature=0.5, top_k=3, top_p=0.9)
    # no error thrown
