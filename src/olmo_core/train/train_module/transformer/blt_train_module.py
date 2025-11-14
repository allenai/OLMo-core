import time
from typing import Any, Dict, Optional, Tuple, List
import random
import torch
import torch.distributed as dist
from torch.nn import functional as F

from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.utils import move_to_device
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.nn.blt.config import BLTConfig
from olmo_core.nn.blt import utils as blt_utils
from olmo_core.nn.lm_head import LMOutputWithLoss

from ...common import ReduceType
from .train_module import TransformerTrainModule


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
        batch["step"] = self.trainer.global_step 

        # this has had the byte collator + ByteFSLDataset applied, no need to patch
        if "patch_lens" in batch:
            # apply gradual boundary compression - only during training
            if self.blt_config.gradual_boundary_compression_kind in {"bpe", "entropy", "cross_entropy"}:
                if "bpe_merges" not in batch:
                    raise ValueError("To use bpe gradual boundary compression, bpe_merges must be computed and included in the batch.")

                if self.trainer.global_step >= self.blt_config.gradual_boundary_compression_steps:
                    current_ratio = self.blt_config.target_ratio
                else:
                    current_ratio = self.blt_config.start_ratio + (self.blt_config.target_ratio - self.blt_config.start_ratio) * (self.trainer.global_step / self.blt_config.gradual_boundary_compression_steps)

                patch_lens = [[l for l in lengths if l > 0] for lengths in batch["patch_lens"].tolist()]
                n_bytes = (batch["labels"] != self.label_ignore_index).sum(dim=1)

                for example_bpe_merges, example_patch_lens, example_n_bytes in zip(batch["bpe_merges"], patch_lens, n_bytes):
                    for merge_idx in example_bpe_merges:
                        if example_n_bytes / len(example_patch_lens) >= current_ratio:
                            break

                        if merge_idx + 2 >= len(example_patch_lens):
                            continue

                        # offset by one due to bos
                        example_patch_lens[merge_idx + 1] = example_patch_lens[merge_idx + 1] + example_patch_lens[merge_idx + 2]
                        del example_patch_lens[merge_idx + 2]

                patch_lens = blt_utils.pad_right([torch.tensor(lengths) for lengths in patch_lens])

                batch["patch_lens"].zero_()
                batch["patch_lens"][:, :patch_lens.shape[1]] = patch_lens

                batch["original_input_ids"] = None # can't use
            elif self.blt_config.gradual_boundary_compression_kind == "space":
                epsilon = 1e-6

                if self.trainer.global_step >= self.blt_config.gradual_boundary_compression_steps:
                    current_ratio = self.blt_config.target_ratio
                else:
                    current_ratio = self.blt_config.start_ratio + (self.blt_config.target_ratio - self.blt_config.start_ratio) * (self.trainer.global_step / self.blt_config.gradual_boundary_compression_steps)
                
                n_bytes = (batch["labels"] != self.label_ignore_index).sum(dim=1)

                subword_ratio = n_bytes / (batch["patch_lens"] > 0).sum(-1)
                space_ratio = n_bytes / (batch["space_patch_lens"] > 0).sum(-1)

                interp = (current_ratio - subword_ratio) / (space_ratio - subword_ratio).clamp(min=epsilon)
                interp = torch.clamp(interp, 0.0, 1.0)

                patch_lens = []

                for example_interp, example_subword_patch_lens, example_space_patch_lens in zip(interp, batch["patch_lens"], batch["space_patch_lens"]):
                    example_interp = example_interp.item()

                    subword_patch_end_indices = set((torch.cumsum(example_subword_patch_lens, dim=0) - 1).tolist())
                    space_patch_end_indices = set((torch.cumsum(example_space_patch_lens, dim=0) - 1).tolist())

                    subword_intersect = subword_patch_end_indices - space_patch_end_indices
                    space_intersect = space_patch_end_indices - subword_patch_end_indices
        
                    patch_end_indices = (
                        (subword_patch_end_indices & space_patch_end_indices) |
                        set(random.sample(list(subword_intersect), int(len(subword_intersect) * (1 - example_interp)))) |
                        set(random.sample(list(space_intersect), int(len(space_intersect) * example_interp)))
                    )
                    patch_end_indices = torch.tensor(sorted(list(patch_end_indices)), dtype=torch.long)
                    example_patch_lens = torch.zeros_like(patch_end_indices)
                    example_patch_lens[0] = 1
                    example_patch_lens[1:] = patch_end_indices[1:] - patch_end_indices[:-1]

                    patch_lens.append(example_patch_lens)

                patch_lens = blt_utils.pad_right(patch_lens)

                batch["patch_lens"].zero_()
                batch["patch_lens"][:, :patch_lens.shape[1]] = patch_lens

                batch["original_input_ids"] = None # can't use   

            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)

            return input_ids, labels, batch
        else:
            # this hasn't, so it's an eval batch, we need to manually convert to bytes.
            # assumes ICLMultiChoiceTaskDataset collation
            all_original_input_ids = []
            all_input_ids = []
            all_expanded_input_ids = []
            all_patch_lens = []
            all_space_patch_lens = []
            all_dc_input_ids = []
            all_ctx = []
            all_continuation = []

            all_ctx_len = []
            all_dc_len = []
            all_cont_len = []

            device = batch["input_ids"].device

            for idx in range(len(batch["input_ids"])):
                prev_ctx_len = batch["ctx_len"][idx].item()
                prev_dc_len = batch["dc_len"][idx].item()
                prev_cont_len = batch["cont_len"][idx].item()

                # there is at least one case where ctx_len + cont_len > input_ids length on hellaswag
                # ctx: `"High jump: We see a black opening screen. We see a blue track with people running on it. We`
                # cont: ` then see a man do a high jump at 5\'9 ".`
                # input ids: `"High jump: We see a black opening screen. We see a blue track with people running on it. We then see a man do a high jump at 5\'9` (missing the final `".`)
                # so reconstruct input ids from ctx + cont.
                # + [0] since skip_last=True
                original_input_ids = batch["ctx"][idx].tolist()[:prev_ctx_len] + batch["continuation"][idx].tolist()[:prev_cont_len] + [0]
                input_ids, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(
                    original_input_ids, add_bos=True, skip_last=True,
                )
                space_patch_lens = self.tokenizer.get_space_patch_lengths(input_ids)
                while len(space_patch_lens) < len(patch_lens):
                    space_patch_lens.append(0)

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

                # append \n since boundary predictor has lookahead
                input_ids += self.tokenizer.encode("\n" * self.model.local_encoder.boundary_predictor_lookahead) # type: ignore
                expanded_input_ids = self.tokenizer.expand_byte_ids(input_ids)
    
                all_original_input_ids.append(torch.tensor(original_input_ids, dtype=torch.long))
                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_expanded_input_ids.append(torch.tensor(expanded_input_ids, dtype=torch.long))
                all_patch_lens.append(torch.tensor(patch_lens, dtype=torch.long))
                all_space_patch_lens.append(torch.tensor(space_patch_lens, dtype=torch.long))
                all_dc_input_ids.append(torch.tensor(dc_input_ids, dtype=torch.long))
                all_ctx.append(torch.tensor(ctx, dtype=torch.long))
                all_continuation.append(torch.tensor(continuation, dtype=torch.long))
                all_ctx_len.append(ctx_len)
                all_dc_len.append(dc_len)
                all_cont_len.append(cont_len)

            batch["original_input_ids"] = blt_utils.pad_right(all_original_input_ids).to(device)
            batch["input_ids"] = blt_utils.pad_right(all_input_ids).to(device)
            batch["expanded_input_ids"] = blt_utils.pad_right(all_expanded_input_ids).to(device)
            batch["attention_mask"] = blt_utils.pad_right([torch.ones_like(t) for t in all_input_ids]).to(device)
            batch["patch_lens"] = blt_utils.pad_right(all_patch_lens).to(device)
            batch["space_patch_lens"] = blt_utils.pad_right(all_space_patch_lens).to(device)
            batch["dc_input_ids"] = blt_utils.pad_right(all_dc_input_ids).to(device)
            batch["ctx"] = blt_utils.pad_right(all_ctx).to(device)
            batch["continuation"] = blt_utils.pad_right(all_continuation).to(device)
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
        start_time = time.time()

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

        if self.blt_config.patching == "dolma2":
            patch_lens = batch["patch_lens"]
        elif self.blt_config.patching == "space":
            patch_lens = batch["space_patch_lens"]
        else:
            raise ValueError(f"Unknown patching type: {self.blt_config.patching}")

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        shifted_patch_lens = F.pad(
            patch_lens[:, 1:],
            (0, 1),
            value=0,
        )
        batch_num_patches_for_loss = move_to_device((shifted_patch_lens != 0).sum(), self.device)

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
                out, metrics = self.model_forward(  # type: ignore
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    patch_loss_div_factor=batch_num_patches_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                metrics["mean_byte_len"] = (labels != self.label_ignore_index).float().mean()  # type: ignore
                metrics["max_byte_len"] = (labels != self.label_ignore_index).float().max()  # type: ignore
                metrics["mean_patch_len"] = (patch_lens != 0).float().mean()  # type: ignore
                metrics["max_patch_len"] = (patch_lens != 0).float().max()  # type: ignore

                for key, value in metrics.items():  # type: ignore
                    if isinstance(value, torch.Tensor):
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + get_local_tensor(value.detach()) / num_micro_batches
                    else:
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + value / num_micro_batches

                # Run backward pass.
                out.loss.backward()  # type: ignore

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

        # time
        self.record_metric(
            "time_per_batch",
            (time.time() - start_time),
            ReduceType.mean,
        )

        # epoch
        if self.trainer.steps_per_epoch is not None:
            self.record_metric(
                "epoch",
                self.trainer.global_step / self.trainer.steps_per_epoch,
                ReduceType.mean,
            )

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ):
        # TODO: (epwalsh) Currently all of our evaluators require the full logits locally,
        # but when we're using CP/TP we usually can't materialize the full logits locally (due to OOMs).
        # However we could at least support in-loop PPL evals with a little work in the evaluator
        # code to handle the sharded logits.
        if self.cp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support context parallelism yet, "
                "please disable in-loop evals"
            )
        if self.tp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support tensor parallelism yet, "
                "please disable in-loop evals"
            )

        orig_ctx, orig_cont, orig_ctx_len, orig_cont_len = (
            batch["ctx"],
            batch["continuation"],
            batch["ctx_len"],
            batch["cont_len"],
        )
        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        self.model.eval()

        with self._eval_batch_context():
            eval_mode = model_kwargs.get("eval_mode", None)

            if eval_mode == "subword":
                with self._model_forward_context():
                    out, _ = self.model.subword_forward(  # type: ignore
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="none",
                        return_logits=True,
                        **model_kwargs,
                    )
                # subword_forward gives us logits over the original (Dolma2) tokens.
                # so we need to change the batch tokens / token info back to subword token space from byte space.
                batch["input_ids"] = batch["original_input_ids"]
                batch["ctx"] = orig_ctx
                batch["continuation"] = orig_cont
                batch["ctx_len"] = orig_ctx_len
                batch["cont_len"] = orig_cont_len
            elif eval_mode == "teacher":
                with self._model_forward_context():
                    out, _ = self.model.teacher_forward(  # type: ignore
                        batch["original_input_ids"],
                    )

                # teacher_forward gives us logits over the original (Dolma2) tokens.
                # so we need to change the batch tokens / token info back to subword token space from byte space.
                batch["input_ids"] = batch["original_input_ids"]
                batch["ctx"] = orig_ctx
                batch["continuation"] = orig_cont
                batch["ctx_len"] = orig_ctx_len
                batch["cont_len"] = orig_cont_len
            elif eval_mode is not None:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")
            else:
                with self._model_forward_context():
                    out, _ = self.model.student_forward(  # type: ignore
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="none",
                        return_logits=True,
                        **model_kwargs,
                    )

        return out