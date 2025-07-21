from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.distributed as dist
from torch.nn import functional as F

from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.utils import move_to_device
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.nn.blt.config import BLTConfig

from ...common import ReduceType
from .train_module import TransformerTrainModule


def _pad_right(
    tensors: List[torch.Tensor],
):
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_shape = (0, max_len - t.size(0))
        padded.append(F.pad(t, pad_shape, value=0))
    return torch.stack(padded, dim=0)


class TransformerBLTTrainModule(TransformerTrainModule):
    def __init__(self, *args, blt_config: BLTConfig, **kwargs):
        super().__init__(*args, **kwargs)

        self.blt_config = blt_config

        if self.blt_config.tokenizer is None:
            raise ValueError("BLTTrainModule requires a ByteTokenizerConfig in blt_config.tokenizer")

        self.tokenizer = self.blt_config.tokenizer.build()

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        batch["blt_config"] = self.blt_config

        # this has had the byte collator + ByteFSLDataset applied, no need to patch
        if "patch_lens" in batch:
            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)

            return input_ids, labels, batch
        else:
            # this hasn't, so it's an eval batch, we need to manually convert to bytes.
            # assumes ICLMultiChoiceTaskDataset collation
            all_input_ids = []
            all_patch_lens = []
            all_dc_input_ids = []
            all_ctx = []
            all_continuation = []

            all_ctx_len = []
            all_dc_len = []
            all_cont_len = []

            for idx in range(len(batch["input_ids"])):
                prev_ctx_len = batch["ctx_len"][idx].item()
                prev_dc_len = batch["dc_len"][idx].item()
                prev_cont_len = batch["cont_len"][idx].item()

                # there is at least one case where ctx_len + cont_len > input_ids length on hellaswag
                # ctx: `"High jump: We see a black opening screen. We see a blue track with people running on it. We`
                # cont: ` then see a man do a high jump at 5\'9 ".`
                # input ids: `"High jump: We see a black opening screen. We see a blue track with people running on it. We then see a man do a high jump at 5\'9` (missing the final `".`)
                # so reconstruct input ids from ctx + cont.
                input_ids, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(
                    batch["ctx"][idx].tolist()[:prev_ctx_len] + batch["continuation"][idx].tolist()[:prev_cont_len], add_bos=True
                )
                dc_input_ids, _ = self.tokenizer.get_tokens_and_patch_lengths(
                    batch["dc_input_ids"][idx].tolist()[:prev_dc_len], add_bos=True
                )
                ctx, _ = self.tokenizer.get_tokens_and_patch_lengths(
                    batch["ctx"][idx].tolist()[:prev_ctx_len], add_bos=True
                )
                continuation, _ = self.tokenizer.get_tokens_and_patch_lengths(
                    batch["continuation"][idx].tolist()[:prev_cont_len], add_bos=False
                )
                assert len(input_ids) == len(ctx) + len(continuation)
                assert len(continuation) > 0

                ctx_len = len(ctx)
                dc_len = len(dc_input_ids)
                cont_len = len(continuation)

                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_patch_lens.append(torch.tensor(patch_lens, dtype=torch.long))
                all_dc_input_ids.append(torch.tensor(dc_input_ids, dtype=torch.long))
                all_ctx.append(torch.tensor(ctx, dtype=torch.long))
                all_continuation.append(torch.tensor(continuation, dtype=torch.long))
                all_ctx_len.append(ctx_len)
                all_dc_len.append(dc_len)
                all_cont_len.append(cont_len)

            device = batch["input_ids"].device

            batch["input_ids"] = _pad_right(all_input_ids).to(device)
            batch["patch_lens"] = _pad_right(all_patch_lens).to(device)
            batch["dc_input_ids"] = _pad_right(all_dc_input_ids).to(device)
            batch["ctx"] = _pad_right(all_ctx).to(device)
            batch["continuation"] = _pad_right(all_continuation).to(device)
            batch["ctx_len"] = torch.tensor(all_ctx_len, dtype=torch.long).to(device)
            batch["dc_len"] = torch.tensor(all_dc_len, dtype=torch.long).to(device)
            batch["cont_len"] = torch.tensor(all_cont_len, dtype=torch.long).to(device)

            input_ids = batch.pop("input_ids")
            # not ideal to hardcode but we are specializing to ICLMultiChoiceTaskDataset anyway..
            labels = F.pad(
                input_ids[..., 1:], (0, 1, 0, 0), value=-100
            )

            return input_ids, labels, batch

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self.model.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        batch_metrics = {}

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
                loss, metrics = self.model_forward(  # type: ignore
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                for key, value in metrics.items():  # type: ignore
                    batch_metrics[key] = batch_metrics.get(key, 0.0) + get_local_tensor(value.detach())

                # Run backward pass.
                loss.backward()  # type: ignore

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            raise NotImplementedError("SkipStepOptimizer not implemented for BLTTrainModule")

        for key, value in batch_metrics.items():
            self.record_metric(
                key,
                value,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

        # epoch
        if self.trainer.steps_per_epoch is not None:
            self.record_metric(
                "epoch",
                self.trainer.global_step / self.trainer.steps_per_epoch,
                ReduceType.mean,
            )