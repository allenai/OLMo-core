from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from olmo_core.eval import vision_alignment_promotion as promotion


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "eval" / "vision_alignment_state_text.py"
    )
    spec = importlib.util.spec_from_file_location("vision_alignment_state_text_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sentinel(tmp_path: Path) -> dict:
    paths = tmp_path / "data_paths.txt"
    paths.write_text("".join(f"s3://bucket/{index}.npy\n" for index in range(128)))
    value = {
        "format": promotion.TEXT_SENTINEL_FORMAT,
        "version": 1,
        "parent_checkpoint": str(tmp_path / "parent"),
        "parent_checkpoint_config_sha256": "a" * 64,
        "parent_data_paths": {
            "path": str(paths),
            "sha256": promotion.sha256_file(paths),
            "count": 128,
        },
        "selection": {
            "algorithm": "evenly-spaced-parent-path-first-window-v1",
            "examples": 128,
            "sequence_length": 256,
            "dtype": "uint32-little-endian",
            "source_indices": list(range(128)),
        },
        "rows": [
            {
                "source_index": index,
                "source_path": f"s3://bucket/{index}.npy",
                "start": 0,
                "tokens": [token % 1000 for token in range(index, index + 257)],
            }
            for index in range(128)
        ],
        "content_sha256": "",
    }
    value["content_sha256"] = promotion.canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    # Exercise the strict JSON representation used by the actual producer.
    return json.loads(json.dumps(value))


def test_evaluate_text_uses_all_supervised_positions(tmp_path: Path) -> None:
    module = _load_module()

    class FakeTrainModule:
        device = torch.device("cpu")

        def eval_batch(self, batch, *, return_response_logits):
            assert return_response_logits is True
            assert batch["input_ids"].device == self.device
            labels = batch["labels"].reshape(-1)
            logits = torch.full((labels.numel(), 1001), -10.0)
            logits[torch.arange(labels.numel()), labels] = 10.0
            return SimpleNamespace(logits=logits)

    result = module._evaluate_text(FakeTrainModule(), _sentinel(tmp_path), batch_size=8)
    assert result["token_ce"].shape == (32_768,)
    assert result["argmax"].shape == (32_768,)
    assert torch.all(result["argmax"] < 1000)


def test_model_state_descriptors_include_non_image_rows(monkeypatch) -> None:
    module = _load_module()

    class Part(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = torch.nn.Linear(1, 1, bias=False)
            self.lm = torch.nn.Module()
            self.lm.embeddings = torch.nn.Embedding(100_352, 1)

    class FakeTrainModule:
        def __init__(self):
            self.model_parts = [Part()]

        @staticmethod
        def _persistent_model_buffer_state_dict():
            return {}

    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)

    def gather(output, value, *, group):
        output[0] = value

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    descriptors = module._model_state_descriptors(FakeTrainModule(), ["vision.*"])
    assert descriptors["vision.weight"]["kind"] == "frozen_tensor"
    non_image = descriptors["lm.embeddings.weight[non_image_rows]"]
    assert non_image["kind"] == "non_image_embedding_rows"
    assert non_image["shape"] == [100_346, 1]
