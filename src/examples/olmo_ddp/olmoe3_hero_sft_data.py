"""SFT packing uses actual record boundaries, never EOS strings quoted in responses."""

import gzip
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from olmo_core.data import NumpyPackedFSLDataset, NumpyPackedFSLDatasetConfig
from olmo_core.data.utils import InstancePacker, write_array_to_disk


class SFTPackedDataset(NumpyPackedFSLDataset):
    """Keep core's bin packing/loading, with authoritative conversation metadata."""

    def _pack_documents_from_source_into_instances(self, *source_paths):
        assert len(source_paths) == 1
        with gzip.open(Path(source_paths[0]).with_suffix(".csv.gz"), "rt") as handle:
            indices = np.array(
                [tuple(map(int, line.split(","))) for line in handle], dtype=self.indices_dtype
            )
        assert indices[0, 0] == 0 and np.all(indices[1:, 0] == indices[:-1, 1])
        assert np.all(indices[:, 1] - indices[:, 0] <= self.sequence_length)
        instances, indices, total = InstancePacker(self.sequence_length).pack_documents(indices)
        offsets, docs, start = [], [], 0
        for instance in instances:
            offsets.extend([start, start + len(instance)])
            start += len(instance)
            docs.extend(instance)
        write_array_to_disk(indices.reshape(-1), self._get_document_indices_path(*source_paths))
        write_array_to_disk(
            np.array(offsets, dtype=self.indices_dtype),
            self._get_instance_offsets_path(*source_paths),
        )
        write_array_to_disk(
            np.array(docs, dtype=self.indices_dtype), self._get_docs_by_instance_path(*source_paths)
        )
        return len(instances), total

    def __getitem__(self, index):
        result = super().__getitem__(index)
        assert len(self._source_path_groups) == 1
        paths = self._source_path_groups[0]
        offsets = np.memmap(
            self._get_instance_offsets_path(*paths), mode="r", dtype=self.indices_dtype
        ).reshape(-1, 2)
        docs = np.memmap(
            self._get_docs_by_instance_path(*paths), mode="r", dtype=self.indices_dtype
        )
        indices = np.memmap(
            self._get_document_indices_path(*paths), mode="r", dtype=self.indices_dtype
        ).reshape(-1, 2)
        first, last = offsets[int(index)]
        selected = indices[docs[int(first) : int(last)]]
        lengths = (selected[:, 1] - selected[:, 0]).astype(np.int32).tolist()
        padding = self.sequence_length - sum(lengths)
        if padding:
            lengths.append(padding)
        assert sum(lengths) == self.sequence_length and min(lengths) > 0
        result["doc_lens"] = torch.tensor(lengths, dtype=torch.int32)
        return result


@dataclass(kw_only=True)
class SFTPackedDatasetConfig(NumpyPackedFSLDatasetConfig):
    """The usual packed config with CSV-driven record identity."""

    def build(self):
        self.validate()
        paths, metadata, masks = self._resolve_paths_metadata(
            allow_mix=True, label_mask_paths=self.label_mask_paths
        )
        assert len(paths) == 1 and masks is not None
        dataset = SFTPackedDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dtype=self.get_dtype(),
            metadata=metadata,
            include_instance_metadata=self.include_instance_metadata,
            generate_doc_lengths=True,
            bos_token_id=self.tokenizer.bos_token_id,
            instance_filter_config=None,
            long_doc_strategy=self.long_doc_strategy,
            label_mask_paths=masks,
            source_group_size=1,
        )
        return self._finalize(dataset)


def self_test():
    """A literal interior EOS must not reset the model or discard any labels."""
    with tempfile.TemporaryDirectory(prefix="sft-boundary-test-") as tmp:
        root = Path(tmp)
        tokens = np.array([1, 9, 2, 3, 4, 9, 1, 5, 6, 9], dtype=np.uint32)
        mask = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.bool_)
        tokens.tofile(root / "tokens.npy")
        mask.tofile(root / "mask.npy")
        with gzip.open(root / "tokens.csv.gz", "wt") as handle:
            handle.write("0,6\n6,10\n")
        dataset = SFTPackedDataset(
            str(root / "tokens.npy"),
            sequence_length=16,
            pad_token_id=0,
            eos_token_id=9,
            vocab_size=12,
            dtype=np.uint32,
            label_mask_paths=[str(root / "mask.npy")],
            generate_doc_lengths=True,
            source_group_size=1,
        )
        dataset.work_dir = root / "cache"
        dataset.prepare()
        assert len(dataset) == 1
        item = dataset[0]
        assert item["doc_lens"].tolist() == [6, 4, 6]
        assert np.array_equal(item["input_ids"][:10].numpy(), tokens)
        assert np.array_equal(item["label_mask"][:10].numpy(), mask)
        assert not item["label_mask"][10:].any()
