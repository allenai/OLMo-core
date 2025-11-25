from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Union

import torch
import torch.nn.functional as F

from ..config import StrEnum

__all__ = ["DataCollator"]


def _source_name_to_expert_label(source_name: str) -> torch.Tensor:
    """Convert source_name to expert label tensor (one-hot, shape: 4).
    
    Mapping for 3-expert setup (Expert 3 is masked/duplicate of Expert 1):
    - Expert 0 (Math): [1, 0, 0, 0] - mj_finemath4plus, mj_finemath
    - Expert 1 (General): [0, 1, 0, 0] - everything else (academic, technical, web content)
    - Expert 2 (Code): [0, 0, 1, 0] - starcoder, code
    - Expert 3: Unused/masked (duplicate of Expert 1)
    """
    source_lower = source_name.lower().strip()
    
    # Math expert (expert 0)
    if source_lower.startswith("mj_finemath4plus") or source_lower.startswith("mj_finemath"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Code expert (expert 2)
    if source_lower.startswith("starcoder") or "code" in source_lower:
        return torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    
    # General expert (expert 1) - everything else including academic/technical
    return torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


class PaddingDirection(StrEnum):
    """
    Specifies the direction to pad instances when needed.
    """

    left = "left"
    right = "right"


@dataclass
class DataCollator:
    """
    The default data collator used by :class:`~olmo_core.data.data_loader.TextDataLoaderBase` subclasses.
    """

    pad_token_id: int
    pad_direction: PaddingDirection = PaddingDirection.right

    def __call__(
        self, items: Union[Sequence[Dict[str, Any]], Sequence[torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Create a batch from a sequence of instances.
        """
        assert items
        
        # Debug logging on first call
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            import logging
            log = logging.getLogger(__name__)
            log.info(f"🔍 DataCollator receiving {len(items)} items")
            log.info(f"  First item type: {type(items[0])}")
            if isinstance(items[0], dict):
                log.info(f"  First item keys: {list(items[0].keys())}")
                log.info(f"  First item has 'index': {'index' in items[0]}")
                log.info(f"  First item has 'metadata': {'metadata' in items[0]}")
                log.info(f"  First item has 'expert_labels': {'expert_labels' in items[0]}")
                if 'metadata' not in items[0]:
                    log.error(f"❌ CRITICAL: Items don't have 'metadata'! Dataset must have include_instance_metadata=True and metadata configured.")
                elif isinstance(items[0].get('metadata'), dict):
                    log.info(f"  First item metadata: {items[0]['metadata']}")
                else:
                    log.warning(f"⚠️  First item metadata is not a dict: {type(items[0].get('metadata'))}")
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

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float),
                        pad_shape,
                        value=0.0,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Pad label mask.
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(
                        label_mask.to(dtype=torch.bool),
                        pad_shape,
                        value=False,
                    )
                )

            # Indices.
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))
            elif isinstance(x, dict) and not hasattr(self, '_warned_no_index'):
                self._warned_no_index = True
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"⚠️  Item is dict but has no 'index' field! Keys: {list(x.keys())}")

            # Instance mask.
            instance_mask = x.get("instance_mask") if isinstance(x, dict) else None
            if instance_mask is not None:
                all_instance_mask.append(torch.tensor(instance_mask))

            # Document lengths.
            doc_lens = x.get("doc_lens") if isinstance(x, dict) else None
            if doc_lens is not None:
                doc_pad_shape = (0, max_docs - len(doc_lens))
                all_doc_lens.append(F.pad(doc_lens, doc_pad_shape, value=0))
                all_max_doc_lens.append(int(doc_lens.max()))

            # Metadata.
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)
            
            # Log metadata status on first item if debug not already logged
            if not hasattr(self, '_metadata_debug_logged'):
                self._metadata_debug_logged = True
                import logging
                log = logging.getLogger(__name__)
                log.info(f"🔍 Collator metadata check:")
                log.info(f"  Item has metadata: {metadata is not None}")
                if metadata is not None:
                    log.info(f"  Metadata type: {type(metadata)}")
                    if isinstance(metadata, dict):
                        log.info(f"  Metadata keys: {list(metadata.keys())}")
                        log.info(f"  Metadata source_name: {metadata.get('source_name', 'NOT FOUND')}")

            # Expert labels (for supervised router training).
            expert_labels = x.get("expert_labels") if isinstance(x, dict) else None
            if expert_labels is None and metadata is not None:
                # Inject expert_labels from metadata.source_name if not already present
                source_name = metadata.get("source_name") if isinstance(metadata, dict) else None
                if source_name:
                    expert_labels = _source_name_to_expert_label(source_name)
                    if not hasattr(self, '_expert_label_injected_logged'):
                        self._expert_label_injected_logged = True
                        import logging
                        log = logging.getLogger(__name__)
                        log.info(f"✅ Injected expert_labels from source_name='{source_name}'")
            
            if expert_labels is not None:
                if not isinstance(expert_labels, torch.Tensor):
                    expert_labels = torch.tensor(expert_labels)
                all_expert_labels.append(expert_labels)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        elif not hasattr(self, '_warned_no_indices'):
            self._warned_no_indices = True
            import logging
            log = logging.getLogger(__name__)
            log.warning(f"⚠️  all_indices is empty! No items had 'index' field. Processed {len(items)} items.")
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
            if not hasattr(self, '_expert_labels_added_logged'):
                self._expert_labels_added_logged = True
                import logging
                log = logging.getLogger(__name__)
                log.info(f"✅ Added expert_labels to batch: shape={out['expert_labels'].shape}, count={len(all_expert_labels)}")
        elif not hasattr(self, '_no_expert_labels_warned'):
            self._no_expert_labels_warned = True
            import logging
            log = logging.getLogger(__name__)
            log.warning(f"⚠️  No expert_labels created! Items had metadata: {len(all_metadata)}/{len(items)}, all_metadata={all_metadata[:3] if all_metadata else 'None'}")

        return out
