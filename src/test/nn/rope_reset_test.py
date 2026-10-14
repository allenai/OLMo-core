import pytest
import torch

from olmo_core.nn.rope_reset import build_rope_reset_position_ids
from olmo_core.nn.transformer import TransformerConfig

S, E, EOS, PAD = 10, 11, 1, 0
IDS = dict(doc_start_id=S, doc_end_id=E, eos_id=EOS)


def positions(tokens, valid=None):
    x = torch.tensor([tokens])
    v = None if valid is None else torch.tensor([valid])
    return build_rope_reset_position_ids(x, valid_mask=v, **IDS)[0].tolist()


def test_two_packed_examples():
    # ex 1: preamble (3) | doc | doc | trailing query + answer + EOS ; ex 2: preamble (2) | doc | query
    tokens = [5, 6, 7, S, 8, 9, E, S, 8, E, 20, 21, 30, EOS, 5, 6, S, 8, E, 22]
    expect = [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3, 4, 2]
    assert positions(tokens) == expect


def test_no_markers_is_plain():
    assert positions([5, 6, 7, 8]) == [0, 1, 2, 3]
    assert positions([5, 6, EOS, 7, 8]) == [0, 1, 2, 0, 1]


def test_eval_prompt_matches_training_prefix():
    torch.manual_seed(0)
    for _ in range(20):
        seq = torch.randint(20, 100, (5,)).tolist()
        for _ in range(int(torch.randint(1, 6, ()))):
            seq += [S] + torch.randint(20, 100, (int(torch.randint(1, 9, ())),)).tolist() + [E]
        prompt = seq + torch.randint(20, 100, (4,)).tolist()
        full = prompt + torch.randint(20, 100, (3,)).tolist() + [EOS]
        p_prompt, p_full = positions(prompt), positions(full)
        assert p_full[: len(prompt)] == p_prompt
        # decoding continues +1 from the last prompt position
        assert p_full[len(prompt) :] == [p_prompt[-1] + i for i in range(1, 5)]


def test_left_padding_is_invisible():
    tokens = [5, 6, S, 8, 9, E, S, 8, E, 20]
    padded = [PAD] * 3 + tokens
    p = positions(padded, valid=[False] * 3 + [True] * len(tokens))
    assert p[3:] == positions(tokens)


def test_bounded_by_longest_document():
    doc = [S] + [50] * 30 + [E]
    p = positions([5] * 7 + doc * 40 + [60] * 5)
    assert max(p) == 7 + len(doc) - 1


@pytest.mark.parametrize("pass_explicit", [False, True])
def test_model_forward_derives_positions(pass_explicit: bool):
    torch.manual_seed(0)
    config = TransformerConfig.llama_like(d_model=64, n_heads=4, n_layers=2, vocab_size=128)
    plain = config.build()
    plain.init_weights(device=torch.device("cpu"))
    config.rope_reset = dict(IDS)
    reset = config.build()
    reset.load_state_dict(plain.state_dict())
    assert reset.rope_reset == IDS and plain.rope_reset is None

    x = torch.tensor([[5, 6, 7, S, 8, 9, E, S, 8, 9, E, 20, 21]])
    pos = build_rope_reset_position_ids(x, **IDS)
    with torch.no_grad():
        out_plain = plain(x)
        out_reset = reset(x, position_ids=pos) if pass_explicit else reset(x)
        out_plain_pos = plain(x, position_ids=pos)
    # Same weights: the reset model == the plain model fed the reset positions ...
    torch.testing.assert_close(out_reset, out_plain_pos)
    # ... and the reset positions change the output after the first reset (index 7).
    torch.testing.assert_close(out_reset[:, :7], out_plain[:, :7])
    assert not torch.allclose(out_reset[:, 7:], out_plain[:, 7:])
