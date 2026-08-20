from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from olmo_core.eval.ssmax_attention_diagnostics import (
    KEY_CATEGORIES,
    ProbeSequence,
    SSMaxAttentionDiagnosticsCollector,
    SSMaxProbeManifest,
    build_attention_allow_vector,
    build_probe_manifest,
    capture_ssmax_probe_batches,
    compare_ssmax_attention_reports,
    iter_ssmax_probe_batches,
    probe_manifest_sha256,
    serialize_probe_manifest,
)
from olmo_core.nn.attention import Attention
from olmo_core.nn.attention.backend import FlexAttentionBackend
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType


def _validation_manifest(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "validation.json"
    path.write_text('{"format":"test-validation","version":1}\n')
    import hashlib

    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence(sample_id: str, dataset_index: int, *, offset: int = 0) -> ProbeSequence:
    return ProbeSequence(
        sample_id=sample_id,
        dataset_index=dataset_index,
        input_ids=torch.tensor([10 + offset, 100, 101, 20, 30, 0]),
        token_type_ids=torch.tensor([0, 1, 1, 0, 0, 0]),
        loss_masks=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        valid_tokens=torch.tensor([True, True, True, True, True, False]),
    )


def _manifest(tmp_path: Path, sequences=None, *, maximum: int = 8) -> SSMaxProbeManifest:
    validation_path, validation_sha = _validation_manifest(tmp_path)
    return build_probe_manifest(
        sequences or [_sequence("row-a", 3)],
        validation_manifest_path=validation_path,
        validation_manifest_sha256=validation_sha,
        seed=17,
        max_queries_per_category_per_row=maximum,
    )


def test_attention_allow_vector_matches_multimodal_mask_order() -> None:
    valid = torch.tensor([True, True, True, True, True, False])
    token_types = torch.tensor([0, 1, 1, 0, 0, 0])

    # Image query 1 receives causal keys 0/1 plus future image key 2, but not padding.
    assert build_attention_allow_vector(
        query_position=1,
        valid_tokens=valid,
        token_type_ids=token_types,
    ).tolist() == [True, True, True, False, False, False]

    # The AND masks are applied after the bidirectional-image OR rule.
    subsegments = torch.tensor([0, 0, 0, 1, 2, 0])
    examples = torch.tensor([0, 0, 1, 0, 0, -1])
    assert build_attention_allow_vector(
        query_position=1,
        valid_tokens=valid,
        token_type_ids=token_types,
        subsegment_ids=subsegments,
        example_ids=examples,
    ).tolist() == [True, True, False, False, False, False]
    assert build_attention_allow_vector(
        query_position=4,
        valid_tokens=valid,
        token_type_ids=token_types,
        subsegment_ids=subsegments,
        example_ids=examples,
    ).tolist() == [False, False, False, False, True, False]

    with pytest.raises(ValueError, match="right-padding"):
        build_attention_allow_vector(
            query_position=2,
            valid_tokens=torch.tensor([True, False, True]),
            token_type_ids=torch.zeros(3, dtype=torch.long),
        )


def test_attention_allow_vector_matches_flex_mask_for_valid_probe_queries() -> None:
    valid = torch.tensor([True, True, True, True, True, False])
    token_types = torch.tensor([0, 1, 1, 0, 0, 0])
    subsegments = torch.tensor([0, 0, 0, 1, 2, 0])
    examples = torch.tensor([0, 0, 0, 0, 0, -1])
    window_size = (2, 0)
    backend = FlexAttentionBackend(
        head_dim=1,
        n_heads=1,
        n_kv_heads=1,
        scale=1.0,
        window_size=window_size,
    )
    mask_mod = backend._build_mask_mod(
        token_types.unsqueeze(0) != 0,
        subsegments.unsqueeze(0),
        None,
        examples.unsqueeze(0),
    )

    for query_position in torch.where(valid)[0].tolist():
        actual = torch.tensor(
            [
                bool(mask_mod(0, 0, query_position, key_position))
                for key_position in range(len(valid))
            ]
        )
        expected = build_attention_allow_vector(
            query_position=query_position,
            valid_tokens=valid,
            token_type_ids=token_types,
            subsegment_ids=subsegments,
            example_ids=examples,
            window_size=window_size,
        )
        assert torch.equal(actual, expected)


def test_probe_manifest_is_deterministic_and_binds_tokenization(tmp_path: Path) -> None:
    sequences = [_sequence("row-b", 9, offset=1), _sequence("row-a", 3)]
    manifest = _manifest(tmp_path, sequences, maximum=1)
    repeated = _manifest(tmp_path, sequences, maximum=1)
    assert serialize_probe_manifest(manifest) == serialize_probe_manifest(repeated)
    assert list(manifest.rows_by_sample_id) == ["row-a", "row-b"]
    assert all(
        len(positions) <= 1
        for row in manifest.payload["rows"]
        for positions in row["query_positions"].values()
    )

    path = tmp_path / "probe.json"
    path.write_bytes(serialize_probe_manifest(manifest))
    loaded = SSMaxProbeManifest.load(path, expected_sha256=probe_manifest_sha256(manifest))
    assert loaded.sha256 == manifest.sha256

    changed = json.loads(path.read_text())
    changed["rows"][0]["valid_length"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SHA mismatch"):
        SSMaxProbeManifest.load(path, expected_sha256=manifest.sha256)


class _ToySSMax(nn.Module):
    def __init__(self, *, qk_norm: bool):
        super().__init__()
        norm = (
            LayerNormConfig(
                name=LayerNormType.rms,
                eps=1e-6,
                elementwise_affine=True,
                bias=False,
            )
            if qk_norm
            else None
        )
        self.attention = Attention(
            d_model=8,
            n_heads=4,
            n_kv_heads=2,
            head_dim=2,
            bias=False,
            qk_norm=norm,
            use_head_qk_norm=qk_norm,
            scalable_softmax=True,
            backend="torch",
        )
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
            self.attention.ssmax_scale.copy_(torch.tensor([0.75, 1.0, 1.25, 1.5]))
            if qk_norm:
                self.attention.q_norm.weight.fill_(1.0)
                self.attention.k_norm.weight.fill_(1.0)

    def forward(self, x: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        image = token_type_ids != 0
        or_mask = (image[:, :, None] & image[:, None, :]).unsqueeze(1)
        return self.attention(x, or_mask=or_mask)


@torch.no_grad()
@pytest.mark.parametrize("qk_norm", [False, True])
def test_collector_matches_explicit_gqa_logits_and_entropy(tmp_path: Path, qk_norm: bool) -> None:
    manifest = _manifest(tmp_path, maximum=1)
    sequence = _sequence("row-a", 3)
    model = _ToySSMax(qk_norm=qk_norm).eval()
    x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(7))

    row = manifest.rows_by_sample_id["row-a"]
    query_position = row["query_positions"]["image"][0]
    q, k, _ = model.attention._prepare_qkv(x)
    q_scaled = model.attention._apply_scalable_softmax(q, None)
    allowed = build_attention_allow_vector(
        query_position=query_position,
        valid_tokens=sequence.valid_tokens,
        token_type_ids=sequence.token_type_ids,
    )
    key_positions = torch.where(allowed)[0]
    repeated_k = k[0, key_positions].repeat_interleave(2, dim=1)
    logits = torch.einsum("hd,khd->hk", q_scaled[0, query_position], repeated_k) * math.sqrt(0.5)
    probabilities = logits.float().softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    normalized_entropy = entropy / math.log(len(key_positions))
    valid = sequence.valid_tokens
    key_category_by_position = torch.stack(
        [
            valid & (sequence.token_type_ids != 0),
            valid & (sequence.token_type_ids == 0) & ~(sequence.loss_masks > 0),
            valid & (sequence.token_type_ids == 0) & (sequence.loss_masks > 0),
        ]
    )
    key_category_by_visible_key = key_category_by_position[:, key_positions]
    attention_mass_by_key_category = (
        probabilities @ key_category_by_visible_key.to(dtype=probabilities.dtype).T
    )
    argmax_key_by_category = key_category_by_visible_key[:, probabilities.argmax(-1)].T.float()

    collector = SSMaxAttentionDiagnosticsCollector(
        model,
        manifest,
        distribution_sample_capacity=128,
        query_chunk_size=1,
    )
    with collector.capture_batch(
        sample_ids=["row-a"],
        input_ids=sequence.input_ids.unsqueeze(0),
        token_type_ids=sequence.token_type_ids.unsqueeze(0),
        loss_masks=sequence.loss_masks.unsqueeze(0),
        valid_tokens=sequence.valid_tokens.unsqueeze(0),
    ):
        model(x, sequence.token_type_ids.unsqueeze(0))
    report = collector.finalize(checkpoint_identity={"name": "toy"})
    layer = report["layers"]["attention"]
    assert layer["n_heads"] == 4
    assert layer["n_kv_heads"] == 2
    assert layer["gqa_group_size"] == 2
    assert [layer["heads"][str(head)]["kv_head"] for head in range(4)] == [0, 0, 1, 1]
    for head in range(4):
        metrics = layer["heads"][str(head)]["categories"]["image"]
        assert metrics["logit"]["mean"] == pytest.approx(float(logits[head].mean()), abs=1e-6)
        assert metrics["normalized_entropy"]["mean"] == pytest.approx(
            float(normalized_entropy[head]), abs=1e-6
        )
        assert metrics["ssmax_effective_multiplier"]["mean"] == pytest.approx(
            math.log(query_position + 1) * layer["heads"][str(head)]["ssmax_scale"]
        )
        for key_category_index, key_category in enumerate(KEY_CATEGORIES):
            assert metrics[f"attention_mass_to_{key_category}_keys"]["mean"] == pytest.approx(
                float(attention_mass_by_key_category[head, key_category_index]), abs=1e-6
            )
            assert metrics[f"argmax_key_is_{key_category}"]["mean"] == pytest.approx(
                float(argmax_key_by_category[head, key_category_index]), abs=1e-6
            )
        assert metrics["attention_mass_to_allowed_keys"]["mean"] == pytest.approx(1.0)
        assert metrics["attention_mass_normalization_error"]["max"] <= 1e-6
        assert metrics["attention_mass_category_partition_error"]["max"] <= 1e-6
    assert report["protocol"]["key_categories"] == list(KEY_CATEGORIES)
    collector.close()


def test_state_merge_requires_exact_disjoint_manifest_coverage(tmp_path: Path) -> None:
    sequences = [_sequence("row-a", 3), _sequence("row-b", 9, offset=1)]
    manifest = _manifest(tmp_path, sequences, maximum=1)
    states = []
    for sequence in sequences:
        model = _ToySSMax(qk_norm=False).eval()
        collector = SSMaxAttentionDiagnosticsCollector(
            model,
            manifest,
            distribution_sample_capacity=1,
            query_chunk_size=1,
        )
        x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(sequence.dataset_index))
        with collector.capture_batch(
            sample_ids=[sequence.sample_id],
            input_ids=sequence.input_ids.unsqueeze(0),
            token_type_ids=sequence.token_type_ids.unsqueeze(0),
            loss_masks=sequence.loss_masks.unsqueeze(0),
            valid_tokens=sequence.valid_tokens.unsqueeze(0),
        ):
            model(x, sequence.token_type_ids.unsqueeze(0))
        states.append(collector.export_state())
        assert all(
            distribution["sample"]["count"] <= 1
            for distribution in states[-1]["distributions"].values()
        )
        collector.close()

    report = SSMaxAttentionDiagnosticsCollector.finalize_states(
        manifest, states, checkpoint_identity={"name": "merged"}
    )
    reverse_report = SSMaxAttentionDiagnosticsCollector.finalize_states(
        manifest, list(reversed(states)), checkpoint_identity={"name": "merged"}
    )
    assert reverse_report == report
    assert report["coverage"]["sample_ids"] == ["row-a", "row-b"]
    for category in ("all", "image", "prompt", "response"):
        metrics = report["layers"]["attention"]["heads"]["0"]["categories"][category]
        assert metrics["attention_mass_to_allowed_keys"]["count"] == 2
        assert sum(
            metrics[f"attention_mass_to_{key_category}_keys"]["mean"]
            for key_category in KEY_CATEGORIES
        ) == pytest.approx(1.0, abs=1e-6)
        assert sum(
            metrics[f"argmax_key_is_{key_category}"]["mean"] for key_category in KEY_CATEGORIES
        ) == pytest.approx(1.0, abs=1e-6)
        assert metrics["attention_mass_normalization_error"]["max"] <= 1e-6
        assert metrics["attention_mass_category_partition_error"]["max"] <= 1e-6

    legacy_states = []
    for state in states:
        legacy_state = json.loads(json.dumps(state))
        legacy_state.pop("metric_schema")
        legacy_state["distributions"] = {
            key: value
            for key, value in legacy_state["distributions"].items()
            if not key.rsplit("\0", 1)[-1].startswith(("attention_mass_", "argmax_key_"))
        }
        legacy_states.append(legacy_state)
    legacy_report = SSMaxAttentionDiagnosticsCollector.finalize_states(
        manifest, legacy_states, checkpoint_identity={"name": "legacy"}
    )
    assert legacy_report["protocol"]["name"].endswith("-v1")
    assert (
        "attention_mass_to_image_keys"
        not in legacy_report["layers"]["attention"]["heads"]["0"]["categories"]["all"]
    )
    legacy_comparison = compare_ssmax_attention_reports(legacy_report, legacy_report)
    assert legacy_comparison["comparisons"]
    assert all("key_routing" not in record for record in legacy_comparison["comparisons"])
    with pytest.raises(ValueError, match="different metric schemas"):
        SSMaxAttentionDiagnosticsCollector.finalize_states(
            manifest, [legacy_states[0], states[1]], checkpoint_identity={"name": "mixed"}
        )
    with pytest.raises(ValueError, match="overlap"):
        SSMaxAttentionDiagnosticsCollector.finalize_states(
            manifest, [states[0], states[0]], checkpoint_identity={"name": "bad"}
        )


def test_operational_batch_helper_partitions_and_captures(tmp_path: Path) -> None:
    content_ids = ["0" * 64 for _ in range(10)]
    content_ids[3] = "a" * 64
    content_ids[9] = "b" * 64
    sequences = [
        _sequence(f"pixmo_caption:3:{content_ids[3]}", 3),
        _sequence(f"pixmo_caption:9:{content_ids[9]}", 9, offset=1),
    ]
    manifest = _manifest(tmp_path, sequences, maximum=1)
    payload = manifest.as_dict()
    payload["population"] = {
        "source": "pixmo_caption",
        "selected_dataset_indices": [3, 9],
        "selected_content_ids": [content_ids[3], content_ids[9]],
    }
    manifest = SSMaxProbeManifest.from_dict(payload)
    examples = {
        sequence.dataset_index: {
            "input_ids": sequence.input_ids,
            "token_type_ids": sequence.token_type_ids,
            "loss_masks": sequence.loss_masks,
            "router_token_mask": sequence.valid_tokens,
        }
        for sequence in sequences
    }

    class Dataset:
        def get(self, index: int, epoch: int):
            assert epoch == 0
            return examples[index]

    def collate(rows):
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

    model = _ToySSMax(qk_norm=False).eval()
    collector = SSMaxAttentionDiagnosticsCollector(model, manifest)
    batches = list(
        iter_ssmax_probe_batches(
            Dataset(),
            manifest,
            content_ids=content_ids,
            collate=collate,
            rank=1,
            world_size=2,
            batch_size=1,
        )
    )
    assert [batch.dataset_indices for batch in batches] == [(9,)]

    def forward(batch):
        x = torch.nn.functional.one_hot(batch["input_ids"] % 8, num_classes=8).float()
        model(x, batch["token_type_ids"])

    state = capture_ssmax_probe_batches(collector, batches, forward_batch=forward)
    assert state["seen_sample_ids"] == [sequences[1].sample_id]
    collector.close()

    changed_content_ids = list(content_ids)
    changed_content_ids[9] = "c" * 64
    with pytest.raises(ValueError, match="sample/content identity"):
        list(
            iter_ssmax_probe_batches(
                Dataset(),
                manifest,
                content_ids=changed_content_ids,
                collate=collate,
                rank=1,
                world_size=2,
                batch_size=1,
            )
        )

    changed_input = examples[9]["input_ids"].clone()
    changed_input[0] += 1
    examples[9]["input_ids"] = changed_input
    with pytest.raises(ValueError, match="input_ids prefix differs"):
        list(
            iter_ssmax_probe_batches(
                Dataset(),
                manifest,
                content_ids=content_ids,
                collate=collate,
                rank=1,
                world_size=2,
                batch_size=1,
            )
        )


def test_compare_reports_flags_entropy_and_logit_collapse(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, maximum=1)
    sequence = _sequence("row-a", 3)

    def collect(scale: float, name: str):
        model = _ToySSMax(qk_norm=False).eval()
        with torch.no_grad():
            model.attention.ssmax_scale.mul_(scale)
        collector = SSMaxAttentionDiagnosticsCollector(model, manifest)
        x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(99))
        with collector.capture_batch(
            sample_ids=["row-a"],
            input_ids=sequence.input_ids.unsqueeze(0),
            token_type_ids=sequence.token_type_ids.unsqueeze(0),
            loss_masks=sequence.loss_masks.unsqueeze(0),
            valid_tokens=sequence.valid_tokens.unsqueeze(0),
        ):
            model(x, sequence.token_type_ids.unsqueeze(0))
        return collector.finalize(checkpoint_identity={"name": name})

    baseline = collect(0.2, "baseline")
    candidate = collect(20.0, "candidate")
    comparison = compare_ssmax_attention_reports(
        baseline,
        candidate,
        entropy_drop_threshold=0.01,
        effective_context_fraction_ratio_threshold=0.99,
        absolute_logit_q99_ratio_threshold=2.0,
        q_magnitude_ratio_threshold=2.0,
    )
    assert comparison["flag_count"] > 0
    reasons = {reason for flag in comparison["flags"] for reason in flag["reasons"]}
    assert "absolute_logit_q99_growth" in reasons
    assert "post_ssmax_query_magnitude_growth" in reasons
    record = next(
        record
        for record in comparison["comparisons"]
        if record["layer"] == "attention" and record["head"] == 0 and record["category"] == "image"
    )
    routing = record["key_routing"]
    for key_category in KEY_CATEGORIES:
        destination = routing["destinations"][key_category]
        assert destination["attention_mass"]["delta"] == pytest.approx(
            destination["attention_mass"]["candidate_mean"]
            - destination["attention_mass"]["baseline_mean"]
        )
        assert destination["argmax_share"]["delta"] == pytest.approx(
            destination["argmax_share"]["candidate"] - destination["argmax_share"]["baseline"]
        )
    assert sum(
        routing["destinations"][key_category]["attention_mass"]["candidate_mean"]
        for key_category in KEY_CATEGORIES
    ) == pytest.approx(1.0, abs=1e-6)
    assert sum(
        routing["destinations"][key_category]["argmax_share"]["candidate"]
        for key_category in KEY_CATEGORIES
    ) == pytest.approx(1.0, abs=1e-6)
    assert routing["checks"]["attention_mass_normalization_error"]["candidate"] <= 1e-6
    assert routing["checks"]["attention_mass_category_partition_error"]["candidate"] <= 1e-6
