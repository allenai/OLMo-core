import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from olmo_core.data.composable.pre_chunked_instance_source import (
    PreChunkedInstanceSource,
    _extract_prefix,
    _match_parallel_paths,
)
from olmo_core.nn.rope import compute_inv_freqs, compute_rope_from_positions


class TestExtractPrefix:
    def test_tokens(self):
        assert _extract_prefix("part-03-00042-tokens.npy", "-tokens.npy") == "part-03-00042"

    def test_pos_ids(self):
        assert _extract_prefix("part-03-00042-pos_ids.npy", "-pos_ids.npy") == "part-03-00042"

    def test_vis_limit(self):
        assert _extract_prefix("part-03-00042-vis_limit.npy", "-vis_limit.npy") == "part-03-00042"

    def test_full_path(self):
        assert (
            _extract_prefix("/weka/data/part-03-00042-tokens.npy", "-tokens.npy")
            == "part-03-00042"
        )


class TestMatchParallelPaths:
    def test_basic_matching(self):
        tokens = ["/data/part-00-tokens.npy", "/data/part-01-tokens.npy"]
        pos_ids = ["/data/part-00-pos_ids.npy", "/data/part-01-pos_ids.npy"]
        vis_limit = ["/data/part-00-vis_limit.npy", "/data/part-01-vis_limit.npy"]

        triplets = _match_parallel_paths(tokens, pos_ids, vis_limit)
        assert len(triplets) == 2
        assert triplets[0] == (tokens[0], pos_ids[0], vis_limit[0])
        assert triplets[1] == (tokens[1], pos_ids[1], vis_limit[1])

    def test_unordered_matching(self):
        tokens = ["/data/part-01-tokens.npy", "/data/part-00-tokens.npy"]
        pos_ids = ["/data/part-00-pos_ids.npy", "/data/part-01-pos_ids.npy"]
        vis_limit = ["/data/part-01-vis_limit.npy", "/data/part-00-vis_limit.npy"]

        triplets = _match_parallel_paths(tokens, pos_ids, vis_limit)
        assert len(triplets) == 2
        # Should be sorted by prefix
        assert triplets[0][0] == "/data/part-00-tokens.npy"
        assert triplets[1][0] == "/data/part-01-tokens.npy"

    def test_partial_match(self):
        tokens = ["/data/part-00-tokens.npy", "/data/part-01-tokens.npy"]
        pos_ids = ["/data/part-00-pos_ids.npy"]  # Missing part-01
        vis_limit = ["/data/part-00-vis_limit.npy", "/data/part-01-vis_limit.npy"]

        triplets = _match_parallel_paths(tokens, pos_ids, vis_limit)
        assert len(triplets) == 1
        assert triplets[0][0] == "/data/part-00-tokens.npy"


class TestPreChunkedInstanceSource:
    @pytest.fixture
    def data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_len = 16
            num_instances = 4
            total_tokens = seq_len * num_instances

            # Create two shards
            for shard in range(2):
                shard_tokens = total_tokens // 2
                prefix = f"part-00-{shard:05d}"

                tokens = np.arange(
                    shard * shard_tokens, (shard + 1) * shard_tokens, dtype=np.uint16
                )
                tokens.tofile(Path(tmpdir) / f"{prefix}-tokens.npy")

                pos_ids = np.arange(shard_tokens, dtype=np.uint32) % seq_len
                pos_ids.tofile(Path(tmpdir) / f"{prefix}-pos_ids.npy")

                vis_limit = np.full(shard_tokens, seq_len, dtype=np.uint32)
                vis_limit.tofile(Path(tmpdir) / f"{prefix}-vis_limit.npy")

            yield tmpdir, seq_len, num_instances

    def test_length(self, data_dir):
        tmpdir, seq_len, num_instances = data_dir
        source = PreChunkedInstanceSource(
            token_paths=sorted(str(p) for p in Path(tmpdir).glob("*-tokens.npy")),
            pos_ids_paths=sorted(str(p) for p in Path(tmpdir).glob("*-pos_ids.npy")),
            vis_limit_paths=sorted(str(p) for p in Path(tmpdir).glob("*-vis_limit.npy")),
            token_dtype=np.uint16,
            sequence_length=seq_len,
            work_dir=tmpdir,
        )
        assert len(source) == num_instances

    def test_getitem(self, data_dir):
        tmpdir, seq_len, num_instances = data_dir
        source = PreChunkedInstanceSource(
            token_paths=sorted(str(p) for p in Path(tmpdir).glob("*-tokens.npy")),
            pos_ids_paths=sorted(str(p) for p in Path(tmpdir).glob("*-pos_ids.npy")),
            vis_limit_paths=sorted(str(p) for p in Path(tmpdir).glob("*-vis_limit.npy")),
            token_dtype=np.uint16,
            sequence_length=seq_len,
            work_dir=tmpdir,
        )

        instance = source[0]
        assert "input_ids" in instance
        assert "pos_ids" in instance
        assert "vis_limit" in instance
        assert len(instance["input_ids"]) == seq_len
        assert len(instance["pos_ids"]) == seq_len
        assert len(instance["vis_limit"]) == seq_len

        # First instance should have tokens 0..15
        np.testing.assert_array_equal(instance["input_ids"], np.arange(seq_len, dtype=np.uint16))

    def test_window_alignment(self, data_dir):
        tmpdir, seq_len, num_instances = data_dir
        source = PreChunkedInstanceSource(
            token_paths=sorted(str(p) for p in Path(tmpdir).glob("*-tokens.npy")),
            pos_ids_paths=sorted(str(p) for p in Path(tmpdir).glob("*-pos_ids.npy")),
            vis_limit_paths=sorted(str(p) for p in Path(tmpdir).glob("*-vis_limit.npy")),
            token_dtype=np.uint16,
            sequence_length=seq_len,
            work_dir=tmpdir,
        )

        # Instance 1 should start at token 16
        instance1 = source[1]
        np.testing.assert_array_equal(
            instance1["input_ids"],
            np.arange(seq_len, 2 * seq_len, dtype=np.uint16),
        )

    def test_cross_file_boundary(self, data_dir):
        tmpdir, seq_len, num_instances = data_dir
        source = PreChunkedInstanceSource(
            token_paths=sorted(str(p) for p in Path(tmpdir).glob("*-tokens.npy")),
            pos_ids_paths=sorted(str(p) for p in Path(tmpdir).glob("*-pos_ids.npy")),
            vis_limit_paths=sorted(str(p) for p in Path(tmpdir).glob("*-vis_limit.npy")),
            token_dtype=np.uint16,
            sequence_length=seq_len,
            work_dir=tmpdir,
        )

        # Instance 2 is in the second shard
        instance2 = source[2]
        np.testing.assert_array_equal(
            instance2["input_ids"],
            np.arange(2 * seq_len, 3 * seq_len, dtype=np.uint16),
        )


class TestComputeRopeFromPositions:
    def test_sequential_matches_standard(self):
        """Sequential pos_ids should produce identical results to standard RoPE."""
        seq_len = 32
        head_dim = 64
        theta = 500_000
        device = torch.device("cpu")

        inv_freq = compute_inv_freqs(theta, head_dim, device)

        # Standard sequential positions
        seq = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        expected_sin = positions.sin()
        expected_cos = positions.cos()

        # Custom positions using sequential IDs
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
        actual_sin, actual_cos = compute_rope_from_positions(pos_ids, inv_freq)

        torch.testing.assert_close(actual_sin[0], expected_sin)
        torch.testing.assert_close(actual_cos[0], expected_cos)

    def test_batched_output_shape(self):
        batch_size = 4
        seq_len = 16
        head_dim = 32
        device = torch.device("cpu")

        inv_freq = compute_inv_freqs(500_000, head_dim, device)
        pos_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        pos_sin, pos_cos = compute_rope_from_positions(pos_ids, inv_freq)
        assert pos_sin.shape == (batch_size, seq_len, head_dim)
        assert pos_cos.shape == (batch_size, seq_len, head_dim)

    def test_nonsequential_positions(self):
        """Non-sequential pos_ids should produce different RoPE from sequential."""
        seq_len = 16
        head_dim = 32
        device = torch.device("cpu")

        inv_freq = compute_inv_freqs(500_000, head_dim, device)

        sequential = torch.arange(seq_len, device=device).unsqueeze(0)
        nonsequential = torch.tensor([[5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14, 7, 9, 11, 13, 15]])

        sin_seq, _ = compute_rope_from_positions(sequential, inv_freq)
        sin_nonseq, _ = compute_rope_from_positions(nonsequential, inv_freq)

        assert not torch.allclose(sin_seq, sin_nonseq)


def _build_mask(vis_limit: torch.Tensor) -> torch.Tensor:
    """Reference implementation mirroring model.py's mask construction."""
    B, S = vis_limit.shape
    q_idx = torch.arange(S)
    causal = q_idx.unsqueeze(1) >= q_idx.unsqueeze(0)  # (S, S)
    vis = (q_idx + 1).unsqueeze(0).unsqueeze(2) < vis_limit.unsqueeze(1)  # (B, S, S)
    mask = causal.unsqueeze(0) & vis
    empty_rows = ~mask.any(dim=-1, keepdim=True)  # (B, S, 1)
    eye = torch.eye(S, dtype=torch.bool).unsqueeze(0)  # (1, S, S)
    return mask | (empty_rows & eye)


class TestVisLimitToMask:
    def test_simple_tree_mask(self):
        # Tree with shared prefix {0}, two siblings {1,2} and {3}:
        #   - vis_limit[0] = 4: key 0 (prefix) stays open through the window.
        #   - vis_limit[1] = vis_limit[2] = 3: the first sibling closes at
        #     emit_pos=3 when the second sibling arrives.
        #   - vis_limit[3] = 4: the second sibling stays open through the end.
        vis_limit = torch.tensor([[4, 3, 3, 4]])

        expected = torch.tensor(
            [
                [
                    # q=0 predicts input_ids[1]. key 0 is the shared prefix,
                    # which stays open (vis_limit[0]=4 > q+1=1). Row non-empty.
                    [True, False, False, False],
                    # q=1 predicts input_ids[2]. Still inside the first sibling
                    # — key 0 visible (1+1 < 4), key 1 visible (1+1 < 3).
                    [True, True, False, False],
                    # q=2 is the boundary query whose target input_ids[3] lives
                    # in the second sibling. Keys 1 and 2 close at emit_pos=3
                    # so q+1=3 is NOT less than vis_limit=3 for them. Key 0
                    # (prefix) stays open and remains visible — that's the
                    # whole point of the label-shifted rule.
                    [True, False, False, False],
                    # q=3 is the last position of the window. Under the strict
                    # rule every key has vis_limit <= q+1=4, so the row would
                    # be empty. The fallback turns on self-attention at k=3,
                    # which is fine because labels[S-1] is ignore_index.
                    [False, False, False, True],
                ]
            ]
        )
        assert torch.equal(_build_mask(vis_limit), expected)

    def test_full_causal_vis_limit(self):
        # When every vis_limit value is the max S, everyone stays open
        # through the end. The shifted rule gives a standard causal mask
        # for q in 0..S-2; at q=S-1 the row is empty under the strict rule
        # and the fallback lights up the diagonal.
        S = 8
        vis_limit = torch.full((1, S), S)

        expected = torch.tril(torch.ones(S, S, dtype=torch.bool)).clone()
        # Shifted rule excludes the self key at q=S-1 (q+1=S is not < S).
        # Fallback adds (S-1, S-1) back — matches the causal mask anyway.
        for q in range(S):
            for k in range(S):
                if k == q == S - 1:
                    # Covered by fallback.
                    continue
                if q + 1 >= S:
                    expected[q, k] = False
        # Self fallback at q=S-1.
        expected[S - 1, S - 1] = True
        assert torch.equal(_build_mask(vis_limit)[0], expected)

    def test_lcp_zero_boundary_empty_row_guard(self):
        # Two fully divergent branches of length 2 each with no shared prefix
        # (LCP=0): B1 at positions {0,1} closes at emit_pos=2 when B2 arrives,
        # B2 at positions {2,3} stays open through the end.
        vis_limit = torch.tensor([[2, 2, 4, 4]])

        expected = torch.tensor(
            [
                [
                    # q=0 predicts input_ids[1] (inside B1). vis_limit[0]=2,
                    # q+1=1<2 → key 0 visible. Row non-empty.
                    [True, False, False, False],
                    # q=1 is the boundary query predicting input_ids[2]=B2[0].
                    # No shared prefix, so the strict rule gives an empty row.
                    # Self-fallback turns on k=1; under the data-gen invariant
                    # (max_suffix_len == seq_len) that self token is the EOS
                    # terminating the previous branch.
                    [False, True, False, False],
                    # q=2 predicts input_ids[3] (inside B2). Only key 2 (self)
                    # passes the shifted rule: vis_limit[2]=4, q+1=3<4. Keys
                    # 0 and 1 are closed and B2 doesn't share anything with
                    # them. Row non-empty, no fallback needed.
                    [False, False, True, False],
                    # q=3 is the window end: row empty under strict rule,
                    # fallback on self.
                    [False, False, False, True],
                ]
            ]
        )
        assert torch.equal(_build_mask(vis_limit), expected)
