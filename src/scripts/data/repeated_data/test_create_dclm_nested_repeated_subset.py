from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import create_dclm_nested_repeated_subset as nested
import numpy as np


class CreateNestedRepeatedSubsetTest(unittest.TestCase):
    def test_default_paths_follow_target_and_immediate_base(self) -> None:
        self.assertEqual(
            nested.default_output_root(111_000_000, 333_000_000).name,
            "dclm_0802_nested_111m_from_333m",
        )
        self.assertEqual(
            nested.default_output_manifest(111_000_000).name,
            "dclm_0802_repeated_train_111m.json",
        )

    def test_builds_document_prefix_from_immediate_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            base_tokens = np.concatenate(
                [np.arange(9_000, dtype=np.uint32), np.full(3_288, 100257, dtype=np.uint32)]
            )
            token_path = tmp / "base.npy"
            base_tokens.tofile(token_path)

            metadata_path = tmp / "base.csv.gz"
            with nested.deterministic_gzip_text_writer(metadata_path) as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerows(((0, 3_000), (3_000, 6_000), (6_000, 9_000), (9_000, 12_288)))

            ledger_path = tmp / "base.documents.csv.gz"
            with nested.deterministic_gzip_text_writer(ledger_path) as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    (
                        "shard_path",
                        "document_index",
                        "start",
                        "end",
                        "selection_key",
                        "partition_key",
                    )
                )
                writer.writerows(
                    (
                        ("shard", 0, 0, 3_000, "a", "p"),
                        ("shard", 1, 3_000, 6_000, "b", "p"),
                        ("shard", 2, 6_000, 9_000, "c", "p"),
                    )
                )

            entries = [
                {
                    "path": token_path.name,
                    "start_instance": 0,
                    "num_instances": 3,
                    "available_instances": 3,
                    "available_tokens": 12_288,
                }
            ]
            base_manifest = tmp / "base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "format": "olmo-token-subset-v1",
                        "source": {},
                        "selection": {
                            "method": "global-sha256-document-order-prefix",
                            "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
                            "domain": "dclm-train-repeated-sample-v1",
                            "seed": 1,
                            "requested_tokens": 9_000,
                            "selected_documents": 3,
                            "boundary_key": "c",
                        },
                        "materialized": {
                            "path": token_path.name,
                            "token_sha256": nested.sha256_file(token_path),
                            "document_metadata_path": metadata_path.name,
                            "document_metadata_sha256": nested.sha256_file(metadata_path),
                        },
                        "source_document_ledger": {
                            "path": ledger_path.name,
                            "sha256": nested.sha256_file(ledger_path),
                        },
                        "entries": entries,
                        "entries_sha256": nested.manifest_digest(entries),
                    }
                )
                + "\n"
            )

            output_root = tmp / "nested"
            output_manifest = tmp / "nested.json"
            result = nested.build(
                argparse.Namespace(
                    base_manifest=base_manifest,
                    base_manifest_reference="base.json",
                    base_manifest_base_dir=tmp,
                    output_root=output_root,
                    output_manifest=output_manifest,
                    manifest_base_dir=tmp,
                    target_tokens=5_000,
                    alignment_tokens=4_096,
                    eos_token_id=100257,
                    skip_base_token_hash=False,
                )
            )

            self.assertEqual(result["name"], "repeated_train_5000_nested_in_9000")
            self.assertEqual(result["source"]["nested_base_manifest"], "base.json")
            self.assertEqual(result["selection"]["selected_documents"], 2)
            self.assertEqual(result["selection"]["selected_real_document_tokens"], 6_000)
            self.assertEqual(result["selection"]["selected_tokens"], 8_192)
            self.assertTrue(result["nestedness_audit"]["passed"])

            output = np.fromfile(
                output_root / "dclm_0802_repeated_train_5000_uint32.npy", dtype=np.uint32
            )
            np.testing.assert_array_equal(output[:6_000], base_tokens[:6_000])
            self.assertTrue(np.all(output[6_000:] == 100257))


if __name__ == "__main__":
    unittest.main()
