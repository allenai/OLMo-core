from typing import Any, Dict, Iterable, Iterator, Optional

import torch
import torch.distributed as dist

from ..exceptions import OLMoConfigurationError
from .evaluator import Evaluator
from .metrics import MeanMetric


class MultimodalLMEvaluator(Evaluator):
    """Response-token loss and perplexity for multimodal batches.

    This evaluator pairs with a multimodal train module whose ``eval_batch()`` returns
    the summed, per-token-weighted CE loss. Keeping that loss reduced avoids materializing
    full-sequence vocabulary logits during Stage 1 evaluation.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Iterable[Dict[str, Any]],
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        deterministic: bool = True,
    ):
        super().__init__(name=name, batches=batches, device=device, deterministic=deterministic)
        self.ce_loss = MeanMetric(device=device, process_group=process_group)

    def update_metrics(
        self,
        batch: Dict[str, Any],
        ce_loss: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
    ) -> None:
        del logits
        if ce_loss is None:
            return
        if ce_loss.numel() != 1:
            raise OLMoConfigurationError(
                "MultimodalLMEvaluator expects a scalar summed CE loss, "
                f"got shape {tuple(ce_loss.shape)}"
            )

        loss_weights = batch["loss_masks"].to(device=self.ce_loss.device).float()
        valid = loss_weights > 0
        if (labels := batch.get("labels")) is not None:
            valid &= labels.to(device=self.ce_loss.device) != -100
        weight = loss_weights.masked_select(valid).sum()
        self.ce_loss.update(ce_loss.detach() / weight.clamp_min(1.0), weight)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        ce_loss = self.ce_loss.compute()
        return {"CE loss": ce_loss, "PPL": torch.exp(ce_loss)}

    def reset_metrics(self) -> None:
        self.ce_loss.reset()


class MultimodalBlankImageEvaluator(MultimodalLMEvaluator):
    """Evaluate response loss after replacing normalized image patches with zeros.

    The token sequence, response labels, crop geometry, and image-placement indices remain
    unchanged. Because image preprocessing normalizes pixel channels, zeros represent a
    mean-color blank control without introducing crop-count or padding mismatches. Comparing its
    CE against the ordinary evaluator is a content-reliance diagnostic; it is not by itself a
    complete measure of visual understanding.

    :param args: Positional arguments forwarded to :class:`MultimodalLMEvaluator`.
    :param kwargs: Keyword arguments forwarded to :class:`MultimodalLMEvaluator`.

    :raises OLMoConfigurationError: If a batch lacks an image tensor.
    """

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in super().__iter__():
            images = batch.get("images")
            if not isinstance(images, torch.Tensor):
                raise OLMoConfigurationError(
                    "Image-ablation evaluation requires a tensor-valued 'images' batch field"
                )
            transformed = dict(batch)
            transformed["images"] = torch.zeros_like(images)
            yield transformed
