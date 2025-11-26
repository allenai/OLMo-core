from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ..config import StrEnum

__all__ = ["DataCollator"]

log = logging.getLogger(__name__)


def _domain_to_expert_label(source_name: str) -> torch.Tensor:
    """
    Default domain-based expert label mapping.
    
    Mapping for 3-expert setup (Expert 3 is masked/duplicate of Expert 1):
    - Expert 0 (Math): mj_finemath4plus, mj_finemath
    - Expert 1 (General): everything else (academic, technical, web content)
    - Expert 2 (Code): starcoder, code
    - Expert 3: Unused/masked
    """
    source_lower = source_name.lower().strip()
    
    if source_lower.startswith("mj_finemath4plus") or source_lower.startswith("mj_finemath"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    if source_lower.startswith("starcoder") or "code" in source_lower:
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


class PaddingDirection(StrEnum):
    """Specifies the direction to pad instances when needed."""
    left = "left"
    right = "right"


@dataclass
class DataCollator:
    """
    The default data collator used by :class:`~olmo_core.data.data_loader.TextDataLoaderBase` subclasses.
    
    Args:
        pad_token_id: Token ID used for padding.
        pad_direction: Direction to pad sequences ("left" or "right").
        expert_labels_file: Optional path to a JSON file containing pre-computed optimal expert labels.
            If provided, labels are looked up by sequence index. Falls back to domain-based labeling
            for sequences not in the file.
    """

    pad_token_id: int
    pad_direction: PaddingDirection = PaddingDirection.right
    expert_labels_file: Optional[str] = None
    
    # Internal state (not part of dataclass fields for serialization)
    _expert_labels_cache: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    _logged_flags: Dict[str, bool] = field(default_factory=dict, repr=False, compare=False)

    def _load_expert_labels(self) -> Optional[Dict[str, Any]]:
        """Load expert labels from file on first use."""
        if self._expert_labels_cache is not None:
            return self._expert_labels_cache
        
        if self.expert_labels_file is None:
            return None
        
        labels_path = Path(self.expert_labels_file)
        if not labels_path.exists():
            log.warning(f"Expert labels file not found: {labels_path}, using domain-based labels")
            self._expert_labels_cache = {}
            return self._expert_labels_cache
        
        log.info(f"Loading expert labels from {labels_path}")
        with open(labels_path, "r") as f:
            data = json.load(f)
        
        self._expert_labels_cache = data.get("labels", {})
        
        # Log statistics
        metadata = data.get("metadata", {})
        stats = metadata.get("statistics", {})
        num_labels = len(self._expert_labels_cache) if self._expert_labels_cache else 0
        log.info(f"  Loaded {num_labels} labels")
        if stats:
            log.info(f"  Expert distribution: {stats.get('expert_distribution', 'N/A')}")
        
        return self._expert_labels_cache

    def _get_expert_label(self, source_name: str, sequence_index: int) -> torch.Tensor:
        """
        Get expert label for a sequence.
        
        Uses pre-computed optimal labels if available, otherwise falls back to domain-based labeling.
        """
        labels = self._load_expert_labels()
        
        # Try optimal label lookup
        if labels and sequence_index >= 0:
            str_index = str(sequence_index)
            if str_index in labels:
                expert_id = labels[str_index].get("expert_id", 1)
                one_hot = torch.zeros(4, dtype=torch.float32)
                one_hot[expert_id] = 1.0
                return one_hot
        
        # Fall back to domain-based labeling
        return _domain_to_expert_label(source_name)

    def __call__(
        self, items: Union[Sequence[Dict[str, Any]], Sequence[torch.Tensor]]
    ) -> Dict[str, Any]:
        """Create a batch from a sequence of instances."""
        assert items
        
        # One-time logging
        if not self._logged_flags.get('first_call'):
            self._logged_flags['first_call'] = True
            if isinstance(items[0], dict):
                log.info(f"DataCollator: {len(items)} items, keys={list(items[0].keys())}")
                if self.expert_labels_file:
                    log.info(f"  Using optimal labels from: {self.expert_labels_file}")
        
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_label_mask = []
        all_indices = []
        all_metadata = []
        all_instance_mask = []
        all_doc_lens = []
        all_max_doc_lens = []
        all_expert_labels = []
        max_docs = max(
            (len(x["doc_lens"]) if isinstance(x, dict) and "doc_lens" in x else 0 for x in items)
        )

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs
            all_input_ids.append(
                F.pad(input_ids.to(dtype=torch.long), pad_shape, value=self.pad_token_id)
            )

            # Pad attention mask
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(attention_mask.to(dtype=torch.float), pad_shape, value=0.0)
                )

            # Pad attention bias
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(attention_bias, pad_shape + pad_shape, value=pad_value)
                )

            # Pad label mask
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(label_mask.to(dtype=torch.bool), pad_shape, value=False)
                )

            # Indices
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Instance mask
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Document lengths
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

            # Metadata
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

            # Expert labels (for supervised router training)
            expert_labels = x.get("expert_labels") if isinstance(x, dict) else None
            if expert_labels is None and metadata is not None:
                source_name = metadata.get("source_name") if isinstance(metadata, dict) else None
                sequence_index = index.item() if index is not None else -1
                if source_name:
                    expert_labels = self._get_expert_label(source_name, sequence_index)
            
            if expert_labels is not None:
                if not isinstance(expert_labels, torch.Tensor):
                    expert_labels = torch.tensor(expert_labels)
                all_expert_labels.append(expert_labels)

        # Build output dict
        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_instance_mask:
            out["instance_mask"] = torch.stack(all_instance_mask)
        if all_doc_lens:
            out["doc_lens"] = torch.stack(all_doc_lens)
        if all_max_doc_lens:
            out["max_doc_lens"] = all_max_doc_lens
        if all_metadata:
            out["metadata"] = all_metadata
        if all_expert_labels:
            out["expert_labels"] = torch.stack(all_expert_labels)

        return out
