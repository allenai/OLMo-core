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


class TestVisLimitToMask:
    def test_simple_tree_mask(self):
        """Test that vis_limit correctly produces tree-structured attention mask."""
        # Simple example: 4 tokens
        # vis_limit = [4, 3, 3, 4] means:
        #   - token 0 visible to q 0,1,2,3 (vis_limit=4)
        #   - token 1 visible to q 1,2 (vis_limit=3, plus causal k<=q)
        #   - token 2 visible to q 2 (vis_limit=3, plus causal k<=q)
        #   - token 3 visible to q 3 (vis_limit=4, plus causal k<=q)
        vis_limit = torch.tensor([[4, 3, 3, 4]])  # (1, 4)
        S = 4
        q_idx = torch.arange(S)

        # mask(q, k) = (k <= q) AND (q < vis_limit[k])
        causal = q_idx.unsqueeze(1) >= q_idx.unsqueeze(0)  # (S, S)
        vis = q_idx.unsqueeze(0).unsqueeze(2) < vis_limit.unsqueeze(1)  # (1, S, S)
        mask = causal.unsqueeze(0) & vis  # (1, S, S)

        expected = torch.tensor(
            [
                [
                    [True, False, False, False],  # q=0 sees k=0
                    [True, True, False, False],  # q=1 sees k=0,1
                    [True, True, True, False],  # q=2 sees k=0,1,2
                    [True, False, False, True],  # q=3 sees k=0,3 (not k=1,2 because vis_limit=3)
                ]
            ]
        )
        assert torch.equal(mask, expected)

    def test_full_causal_vis_limit(self):
        """When vis_limit is all seq_len, mask should equal standard causal mask."""
        S = 8
        vis_limit = torch.full((1, S), S)
        q_idx = torch.arange(S)

        causal = q_idx.unsqueeze(1) >= q_idx.unsqueeze(0)
        vis = q_idx.unsqueeze(0).unsqueeze(2) < vis_limit.unsqueeze(1)
        mask = causal.unsqueeze(0) & vis

        expected_causal = torch.tril(torch.ones(S, S, dtype=torch.bool)).unsqueeze(0)
        assert torch.equal(mask, expected_causal)
