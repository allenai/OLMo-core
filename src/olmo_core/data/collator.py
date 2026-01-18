from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..config import StrEnum

__all__ = ["DataCollator", "DEFAULT_SOFT_DOMAIN_PRIORS"]

log = logging.getLogger(__name__)


# Default soft domain priors: maps domain type to probability distribution [math, general, code, unused]
# These are the default soft priors when use_soft_domain_labels=True
DEFAULT_SOFT_DOMAIN_PRIORS: Dict[str, List[float]] = {
    "math": [0.5, 0.5, 0.0, 0.0],      # Math: 50% math expert, 50% general expert
    "code": [0.0, 0.5, 0.5, 0.0],      # Code: 50% general expert, 50% code expert
    "general": [0.0, 1.0, 0.0, 0.0],   # General: 100% general expert
}


def _domain_to_expert_label(
    source_name: str,
    soft_priors: Optional[Dict[str, List[float]]] = None,
) -> torch.Tensor:
    """
    Domain-based expert label mapping.
    
    Args:
        source_name: The source/domain name (e.g., "starcoder", "mj_finemath4plus")
        soft_priors: Optional dict mapping domain type ("math", "code", "general") to
            probability distributions over experts. If None, returns hard one-hot labels.
            Example: {"math": [0.5, 0.5, 0.0, 0.0], "code": [0.0, 0.5, 0.5, 0.0]}
    
    Returns:
        Tensor of shape (4,) representing expert label distribution.
        
    Mapping for 3-expert setup (Expert 3 is masked/duplicate of Expert 1):
    - Expert 0 (Math): mj_finemath4plus, mj_finemath
    - Expert 1 (General): everything else (academic, technical, web content)
    - Expert 2 (Code): starcoder, code
    - Expert 3: Unused/masked
    """
    source_lower = source_name.lower().strip()
    
    # Determine domain type
    if source_lower.startswith("mj_finemath4plus") or source_lower.startswith("mj_finemath"):
        domain_type = "math"
    elif source_lower.startswith("starcoder") or "code" in source_lower:
        domain_type = "code"
    else:
        domain_type = "general"
    
    # Return soft priors if provided
    if soft_priors is not None and domain_type in soft_priors:
        priors = soft_priors[domain_type]
        return torch.tensor(priors, dtype=torch.float32)
    
    # Default hard one-hot labels
    if domain_type == "math":
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    elif domain_type == "code":
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    else:
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
        expert_labels_file: Optional path to a JSON file containing pre-computed optimal expert labels
            (per-sequence). If provided, labels are looked up by sequence index.
        expert_labels_dir: Optional path to a directory containing per-token expert labels
            (one .npz file per sequence). Takes precedence over expert_labels_file if both provided.
        soft_domain_priors: Optional dict mapping domain types ("math", "code", "general") to
            probability distributions over experts [math_expert, general_expert, code_expert, unused].
            Example: {"math": [0.5, 0.5, 0.0, 0.0], "code": [0.0, 0.5, 0.5, 0.0]}
            If provided, domain-based labels will be soft distributions instead of one-hot.
            Set to DEFAULT_SOFT_DOMAIN_PRIORS for the default soft priors.
    """

    pad_token_id: int
    pad_direction: PaddingDirection = PaddingDirection.right
    expert_labels_file: Optional[str] = None
    expert_labels_dir: Optional[str] = None
    soft_domain_priors: Optional[Dict[str, List[float]]] = None
    
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

    def _get_per_token_labels(self, sequence_index: int) -> Optional[torch.Tensor]:
        """
        Load per-token expert labels from directory.
        
        Returns:
            Tensor of shape (seq_len-1,) with expert IDs, or None if not found.
        """
        if self.expert_labels_dir is None:
            return None
        
        labels_dir = Path(self.expert_labels_dir)
        label_path = labels_dir / f"seq_{sequence_index:08d}.npz"
        
        if not label_path.exists():
            return None
        
        try:
            data = np.load(label_path)
            labels = torch.from_numpy(data["labels"].astype(np.int64))
            return labels
        except Exception as e:
            if not self._logged_flags.get('per_token_load_error'):
                self._logged_flags['per_token_load_error'] = True
                log.warning(f"Error loading per-token labels from {label_path}: {e}")
            return None

    def _get_expert_label(self, source_name: str, sequence_index: int) -> torch.Tensor:
        """
        Get expert label for a sequence (per-sequence).
        
        Uses pre-computed optimal labels if available, otherwise falls back to domain-based labeling.
        If soft_domain_priors is set, returns soft probability distributions instead of one-hot.
        """
        labels = self._load_expert_labels()
        
        # Try optimal label lookup (always returns hard labels from file)
        if labels and sequence_index >= 0:
            str_index = str(sequence_index)
            if str_index in labels:
                expert_id = labels[str_index].get("expert_id", 1)
                one_hot = torch.zeros(4, dtype=torch.float32)
                one_hot[expert_id] = 1.0
                return one_hot
        
        # Fall back to domain-based labeling (soft or hard depending on soft_domain_priors)
        return _domain_to_expert_label(source_name, soft_priors=self.soft_domain_priors)

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
                if self.soft_domain_priors:
                    log.info(f"  Using soft domain priors: {self.soft_domain_priors}")
        
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
            # Priority: 1) per-token from dir, 2) per-sequence from file, 3) domain-based
            expert_labels = x.get("expert_labels") if isinstance(x, dict) else None
            sequence_index = int(index) if index is not None else -1
            
            if expert_labels is None:
                # Try per-token labels first (from directory)
                if self.expert_labels_dir is not None and sequence_index >= 0:
                    per_token_labels = self._get_per_token_labels(sequence_index)
                    if per_token_labels is not None:
                        expert_labels = per_token_labels  # Shape: (seq_len-1,)
                
                # Fall back to per-sequence labels
            if expert_labels is None and metadata is not None:
                source_name = metadata.get("source_name") if isinstance(metadata, dict) else None
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
