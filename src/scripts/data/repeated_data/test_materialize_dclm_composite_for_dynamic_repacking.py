#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_dclm_composite_for_dynamic_repacking as subject


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FlattenCompositeTest(unittest.TestCase):
    def test_moves_component_padding_to_the_end_and_preserves_documents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifests = root / "manifests"
            data = root / "data"
            manifests.mkdir()
            data.mkdir()
            components = []
            component_specs = (
                ("base", [1, 99, 2, 99, 99, 99, 99, 99], [(0, 2), (2, 4)], 4),
                ("extension", [3, 99, 4, 99], [(0, 2), (2, 4)], 4),
            )
            for name, values, rows, real_tokens in component_specs:
                token_path = data / f"{name}.uint32"
                np.asarray(values, dtype=np.uint32).tofile(token_path)
                metadata_path = data / f"{name}.csv.gz"
                with gzip.open(metadata_path, "wt") as handle:
                    for start, end in rows:
                        handle.write(f"{start},{end}\n")
                    for offset in range(real_tokens, len(values)):
                        handle.write(f"{offset},{offset + 1}\n")
                entry = {
                    "path": str(token_path.relative_to(root)),
                    "start_instance": 0,
                    "num_instances": len(values) // 4,
                    "available_instances": len(values) // 4,
                    "available_tokens": len(values),
                }
                component = {
                    "format": "olmo-token-subset-v1",
                    "selection": {
                        "requested_tokens": real_tokens,
                        "sequence_length": 4,
                        "selected_documents": len(rows),
                        "selected_real_document_tokens": real_tokens,
                        "padding_eos_tokens": len(values) - real_tokens,
                        "selected_tokens": len(values),
                        "selected_instances": len(values) // 4,
                    },
                    "materialized": {
                        "path": str(token_path.relative_to(root)),
                        "token_sha256": digest(token_path),
                        "document_metadata_path": str(metadata_path.relative_to(root)),
                        "document_metadata_sha256": digest(metadata_path),
                    },
                    "entries_sha256": subject.entries_sha256([entry]),
                    "entries": [entry],
                }
                component_path = manifests / f"{name}.json"
                component_path.write_text(json.dumps(component))
                components.append(
                    {
                        "manifest": str(component_path.relative_to(root)),
                        "manifest_sha256": digest(component_path),
                        "requested_tokens": real_tokens,
                        "selected_real_document_tokens": real_tokens,
                        "selected_instances": len(values) // 4,
                    }
                )

            composite_entries = [
                json.loads((manifests / f"{name}.json").read_text())["entries"][0]
                for name, *_ in component_specs
            ]
            composite = {
                "format": "olmo-token-subset-v1",
                "name": "toy-composite",
                "selection": {
                    "requested_tokens": 8,
                    "sequence_length": 4,
                    "selected_documents": 4,
                    "selected_real_document_tokens": 8,
                    "padding_eos_tokens": 4,
                    "selected_tokens": 12,
                    "selected_instances": 3,
                },
                "components": components,
                "entries_sha256": subject.entries_sha256(composite_entries),
                "entries": composite_entries,
            }
            composite_path = manifests / "composite.json"
            composite_path.write_text(json.dumps(composite))
            output_manifest = manifests / "flat.json"
            output_tokens = data / "flat.uint32"

            result = subject.ensure_flattened_pool(
                composite_manifest=composite_path,
                output_manifest=output_manifest,
                output_tokens=output_tokens,
                manifest_base_dir=root,
            )
            self.assertEqual(
                np.fromfile(output_tokens, dtype=np.uint32).tolist(),
                [1, 99, 2, 99, 3, 99, 4, 99, 99, 99, 99, 99],
            )
            with gzip.open(output_tokens.with_suffix(".csv.gz"), "rt") as handle:
                self.assertEqual(
                    handle.read().splitlines(),
                    ["0,2", "2,4", "4,6", "6,8"],
                )
            self.assertEqual(result["entries"][0]["start_instance"], 0)
            self.assertEqual(result["entries"][0]["num_instances"], 3)

            reused = subject.ensure_flattened_pool(
                composite_manifest=composite_path,
                output_manifest=output_manifest,
                output_tokens=output_tokens,
                manifest_base_dir=root,
            )
            self.assertEqual(reused["materialized"]["token_sha256"], digest(output_tokens))


if __name__ == "__main__":
    unittest.main()
