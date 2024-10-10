from typing import Any, Dict, Iterable, Optional, Sequence, Set

import torch
import torch.distributed as dist

from ..data import DataCollator, NumpyFSLDataLoader, NumpyPaddedFSLDataset
from ..distributed.utils import get_fs_local_rank, get_rank, get_world_size
from ..exceptions import OLMoConfigurationError
from ..utils import get_default_device
from .evaluator import Evaluator
from .metrics import MeanMetric


class LMEvaluator(Evaluator):
    """
    Language modeling evaluator that computes cross entropy loss and perplexity over one or more
    datasets.

    .. important::
        The :data:`batches` generated from these evaluators must contain a "metadata" field which
        should be a list of dictionaries, and each dictionary item in the list should contain
        a string field called "label" which indicates which dataset the data file is associated
        with, and should be included in the ``labels`` argument to this class.

    :param labels: All of the task labels.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Iterable[Dict[str, Any]],
        labels: Sequence[str],
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            name=name, batches=batches, device=device, dp_process_group=dp_process_group
        )
        self.metrics = {
            label: MeanMetric(device=device, process_group=dp_process_group) for label in labels
        }

    @classmethod
    def from_numpy_dataset(
        cls,
        dataset: NumpyPaddedFSLDataset,
        *,
        name: str,
        global_batch_size: int,
        collator: DataCollator,
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        seed: int = 0,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
    ) -> "LMEvaluator":
        """
        Initialize an :class:`LMEvaluator` from a :class:`~olmo_core.data.numpy_dataset.NumpyPaddedFSLDataset`.
        """
        labels: Set[str] = set()
        for path, metadata in zip(dataset.paths, dataset.metadata):
            if "label" not in metadata:
                raise OLMoConfigurationError(
                    f"Missing dataset 'label' in metadata for '{path}' dataset"
                )
            labels.add(metadata["label"])

        dataset.prepare()

        device = device or get_default_device()
        data_loader = NumpyFSLDataLoader(
            dataset,
            global_batch_size=global_batch_size,
            collator=collator,
            work_dir=dataset.work_dir,
            seed=seed,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            target_device_type=device.type,
            num_threads=num_threads,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        return cls(
            name=name,
            batches=data_loader,
            labels=list(labels),
            device=device,
            dp_process_group=dp_process_group,
        )

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: torch.Tensor, logits: torch.Tensor
    ) -> None:
        del logits
        for idx, (metadata, tokens_loss) in enumerate(zip(batch["metadata"], ce_loss)):
            metric = self.metrics[metadata["label"]]
            if "label_mask" in batch:
                tokens_loss = tokens_loss.masked_select(batch["label_mask"][idx][1:])
            metric.update(tokens_loss)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for label in sorted(self.metrics.keys()):
            metric = self.metrics[label]
            # In this case we probably haven't called '.update()' on this metric yet,
            # so we do so here with dummy values. Since we pass 0.0 in for weight this won't
            # affect the final value.
            # This can happen when the evaluator contains multiple tasks/datasets and we didn't
            # get to this one within the current evaluation loop.
            metric.update(0.0, 0.0)
            ce_loss = metric.compute()  # could be nan but that's okay.
            out[f"{label}/CE loss"] = ce_loss
            out[f"{label}/PPL"] = torch.exp(ce_loss)
        return out

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
