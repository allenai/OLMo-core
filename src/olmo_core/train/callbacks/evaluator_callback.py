import logging
import math
import time
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyVSLDatasetConfig,
    TextDataLoaderBase,
    TokenizerConfig,
)
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.utils import (
    cuda_sync_debug_mode,
    format_float,
    gc_cuda,
    get_default_device,
    move_to_device,
)

from ..common import Duration, MetricMergeStrategy
from ..train_module import EvalBatchSizeUnit, EvalBatchSpec, TransformerTrainModule
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from olmo_eval import HFTokenizer

    from ..trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class EvaluatorCallback(Callback):
    """
    Runs in-loop evaluations for a :class:`~olmo_core.train.train_module.TransformerTrainModule`
    periodically during training.
    """

    evaluators: List[Evaluator] = field(default_factory=list)
    """
    The evaluators to run.
    """

    eval_interval: Optional[int] = 1000
    """
    The interval (in steps) with which to run the evaluators.
    """

    fixed_steps: Optional[List[int]] = None
    """
    A list of fixed steps at which to run the evaluators.
    """

    eval_on_startup: bool = False
    """
    Whether to run an evaluation when the trainer starts up.
    """

    eval_on_finish: bool = False
    """
    Whether to run an evaluation when training finishes.
    """

    cancel_after_first_eval: bool = False
    """
    If ``True``, cancel the run after running evals for the first time.
    This combined with ``eval_on_startup=True`` is useful if you just want to run in-loop evals
    without training any longer.
    """

    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """
    The duration to run each evaluator for.
    """

    log_interval: int = 5
    """
    How often to log eval progress to the console during an eval loop.
    """

    def post_attach(self):
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' only supports the '{TransformerTrainModule.__name__}' train module"
            )

    def pre_train(self):
        if self.eval_on_startup:
            self.perform_eval()

    def post_train(self):
        if self.eval_on_finish:
            self.perform_eval()

    def post_step(self):
        if self.step <= 1:
            return

        if (self.eval_interval is not None and self.step % self.eval_interval == 0) or (
            self.fixed_steps is not None and self.step in self.fixed_steps
        ):
            self.perform_eval()

    def perform_eval(self, prefix: str = "eval"):
        """
        Run evaluation on all evaluators and record metrics.

        :param prefix: Prefix for metric names (e.g., "eval" or "eval/merged").
            Metrics will be recorded as "{prefix}/{evaluator.name}/{metric_name}".
        """
        # Put model in eval train mode.
        # TODO: make sure grads will be zeroed at this point
        #  self.trainer.optim.zero_grad(set_to_none=True)
        #  self.trainer.model.eval()
        dp_world_size = get_world_size(self.trainer.dp_process_group)

        evaluator_times = []
        evaluator_names = []
        evaluator_bs = []

        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.display_name} evals...")
            start_time = time.monotonic()
            evaluator.reset_metrics()
            eval_step = 0
            eval_tokens = 0
            for batch in evaluator:
                eval_step += 1
                eval_tokens += batch["input_ids"].numel() * dp_world_size

                batch = move_to_device(batch, get_default_device())
                with torch.no_grad():
                    # Run forward pass, get logits and un-reduced CE loss.
                    labels = get_labels(batch)
                    output = self.trainer.train_module.eval_batch(batch, labels=labels)
                    assert isinstance(output, LMOutputWithLoss)
                    logits, _, ce_loss, _ = output

                    # NOTE: might have host-device syncs here but that's okay.
                    with cuda_sync_debug_mode(0):
                        evaluator.update_metrics(batch, ce_loss, logits)

                if self.eval_duration.due(step=eval_step, tokens=eval_tokens, epoch=1):
                    self._log_progress(evaluator, eval_step)
                    break

                if eval_step % self.log_interval == 0 or eval_step == evaluator.total_batches:
                    self._log_progress(evaluator, eval_step)

            # NOTE: going to have a host-device sync here but that's okay. It's only once
            # per evaluator.
            metrics_str = []
            evaluation_names = []
            with cuda_sync_debug_mode(0):
                metrics = evaluator.compute_metrics()
                for name, value in metrics.items():
                    evaluation_names.append(name)
                    metrics_str.append(f"    {name}={format_float(value.item())}")
                    self.trainer.record_metric(f"{prefix}/{evaluator.name}/{name}", value)

            evaluator_times.append(time.monotonic() - start_time)
            evaluator_names.append(evaluation_names)
            evaluator_bs.append(eval_step)

            gc_cuda()
            log.info(
                f"Finished {evaluator.display_name} evals in {time.monotonic() - start_time:.1f} seconds. Metrics:\n"
                + "\n".join(metrics_str)
            )

        # Sort by evaluator_times in ascending order
        sorted_evaluators = sorted(
            zip(evaluator_names, evaluator_bs, evaluator_times), key=lambda x: x[2]
        )

        # Record evaluation speed.
        eval_speeds = []
        for names, bs, t in sorted_evaluators:
            name = names[0]
            eval_speeds.append(f"    {name} (+variants): {t:.1f} sec ({bs} batches)")
        total_time = sum(evaluator_times)
        total_bs = sum(int(bs) if bs is not None else 0 for bs in evaluator_bs)
        eval_speeds.append(
            f"    Total evaluation time: {total_time:.1f} seconds ({total_bs} batches)"
        )
        log.info("Evaluation speed:\n" + "\n".join(eval_speeds))

        self.trainer.record_metric(
            "throughput/in-loop eval time (s)", total_time, merge_strategy=MetricMergeStrategy.sum
        )
        self.trainer.record_metric(
            "throughput/in-loop eval batches", total_bs, merge_strategy=MetricMergeStrategy.sum
        )

        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "canceled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' because we're calling this from all ranks at the same time.
            )

    def _log_progress(self, evaluator: Evaluator, eval_step: int):
        if evaluator.total_batches is not None:
            log.info(f"[eval={evaluator.name},step={eval_step}/{evaluator.total_batches}]")
        else:
            log.info(f"[eval={evaluator.name},step={eval_step}]")


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    eval_dataset: NumpyDatasetConfig
    eval_interval: Optional[int] = 1000
    fixed_steps: Optional[List[int]] = None
    eval_on_startup: bool = False
    eval_on_finish: bool = False
    cancel_after_first_eval: bool = False
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    deterministic: bool = True
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        dataset_max_sequence_length: int
        if isinstance(self.eval_dataset, NumpyVSLDatasetConfig):
            dataset_max_sequence_length = self.eval_dataset.max_sequence_length
        else:
            assert hasattr(self.eval_dataset, "sequence_length")
            dataset_max_sequence_length = self.eval_dataset.sequence_length  # type: ignore

        batch_spec = trainer.train_module.eval_batch_spec
        if (
            batch_spec.max_sequence_length is not None
            and dataset_max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for the LM eval dataset ({dataset_max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        global_eval_batch_size: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            global_eval_batch_size = batch_spec.rank_batch_size * get_world_size(
                trainer.dp_process_group
            )
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            global_eval_batch_size = (
                batch_spec.rank_batch_size
                * dataset_max_sequence_length
                * get_world_size(trainer.dp_process_group)
            )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        dataset = self.eval_dataset.build()
        if not isinstance(dataset, NumpyPaddedFSLDataset):
            raise OLMoConfigurationError(
                f"Expected a padded FSL dataset, got '{dataset.__class__.__name__}' instead"
            )

        if not isinstance(trainer.data_loader, TextDataLoaderBase):
            raise OLMoConfigurationError(
                f"Expected a text-based data loader, got '{dataset.__class__.__name__}' instead"
            )

        evaluator = LMEvaluator.from_numpy_dataset(
            dataset,
            name="lm",
            global_batch_size=global_eval_batch_size,
            collator=trainer.data_loader.collator,
            device=trainer.device,
            dp_process_group=trainer.dp_process_group,
            deterministic=self.deterministic,
        )
        return EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            log_interval=self.log_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            eval_duration=self.eval_duration,
            eval_on_finish=self.eval_on_finish,
        )


@cache
def _all_tasks() -> Set[str]:
    from olmo_eval import list_tasks

    return set(list_tasks())


def _pack_batches_by_length(
    indices: List[int],
    lengths: List[int],
    token_budget: int,
    pad_multiple: int = 128,
) -> List[List[int]]:
    """
    Group ``indices`` into batches of similar-length instances, packing each batch up to
    ``token_budget`` *padded* tokens.

    Instances are sorted by length so each batch only has to pad up to a small multiple of
    ``pad_multiple`` (matching the downstream task's collate function), instead of every batch
    padding to the global maximum. This removes the padding compute wasted by length-heterogeneous
    batches.

    The set of ``indices`` is preserved exactly (only their grouping/order changes), so per-instance
    eval results are unchanged: a transformer forward has no cross-instance interaction, and queries
    are right-padded so causal attention never attends to padding.

    :param indices: The instance indices this rank should evaluate.
    :param lengths: Per-instance (unpadded) query lengths, indexed by instance index.
    :param token_budget: Maximum padded tokens (``batch_size * padded_seq_len``) per batch.
    :param pad_multiple: Sequences are padded to a multiple of this, matching the collate function.

    :returns: A list of batches, each a list of instance indices.
    """

    def padded(n: int) -> int:
        return pad_multiple * math.ceil(max(n, 1) / pad_multiple)

    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0
    for idx in sorted(indices, key=lambda i: lengths[i]):
        new_max = max(current_max, lengths[idx])
        # Start a new batch if adding this instance would exceed the padded-token budget.
        if current and padded(new_max) * (len(current) + 1) > token_budget:
            batches.append(current)
            current = [idx]
            current_max = lengths[idx]
        else:
            current.append(idx)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


@cache
def _fast_iclmetric_class():
    """
    Build (once) a drop-in subclass of ``olmo_eval.ICLMetric`` that defers host-device syncs.

    olmo_eval's ``ICLMetric.update`` pulls ~8 scalars per (instance, choice) back to the host
    *inside* its Python loop (``float(...)``, ``int(...)``, ``.item()``) — each a CUDA sync that
    stalls the GPU. Over a task like hellaswag (~40k instances) that is the dominant cost of the
    metric step. This subclass performs the *identical* math but keeps every computed value on the
    device, moves batch metadata to the host once per call, and converts all scalars to Python
    floats in a single batched ``.tolist()`` per update — so the stored state (and therefore every
    computed metric) is numerically identical, just far fewer syncs.

    Defined lazily so olmo_eval remains an optional import.
    """
    import torch.nn.functional as F
    from olmo_eval import ICLMetric
    from olmo_eval.metrics import LOG_2_OF_E

    class _FastICLMetric(ICLMetric):
        def update(self, batch, lm_logits=None, dc_lm_logits=None):  # type: ignore[override]
            if lm_logits is None:
                for state in (
                    self.loglikelihoods,
                    self.celosses,
                    self.bpbs,
                    self.loglikelihoods_no_leading_space,
                    self.celosses_no_leading_space,
                    self.bpbs_no_leading_space,
                    self.labels,
                ):
                    state.append((None, None, None))
                return

            if self.metric_type == "pmi_dc":
                assert (
                    dc_lm_logits is not None
                ), "PMI_DC acc type selected but no domain conditional logits provided"

            # One-time host transfers of the metadata used for indexing / dict keys (the divisors
            # used in the math stay on-device, so dividing by them does not sync).
            doc_ids = batch["doc_id"].tolist()
            cont_ids = batch["cont_id"].tolist()
            label_ids = batch["label_id"].tolist()
            ctx_lens = batch["ctx_len"].tolist()
            cont_lens = batch["cont_len"].tolist()
            dc_lens = batch["dc_len"].tolist() if self.metric_type == "pmi_dc" else None
            fast_mc = "choice_ids" in batch
            choice_ids_cpu = batch["choice_ids"].tolist() if fast_mc else None

            keys: List[Any] = []  # (doc_id, _cont_id, label_id)
            ll, ce, bp, lln, cen, bpn = [], [], [], [], [], []

            for idx in range(len(doc_ids)):
                doc_id = doc_ids[idx]
                ctx_len = ctx_lens[idx]
                cont_len = cont_lens[idx]
                cont_tokens = batch["continuation"][idx][:cont_len]
                lm_cont_logits = lm_logits[idx][ctx_len - 1 : ctx_len + cont_len - 1]

                if fast_mc:
                    choices = batch["choice_ids"][idx]
                else:
                    choices = [cont_tokens]

                for choice_idx, choice_token in enumerate(choices):
                    if fast_mc:
                        _cont_id = choice_idx
                        _cont_tokens = choice_token.unsqueeze(-1)
                        # is_empty_choice, read from the one-time host copy (no per-choice sync).
                        if choice_ids_cpu[idx][choice_idx] == -1:
                            continue
                    else:
                        # Non-fast continuations are real token ids (never -1), so the original's
                        # empty-choice check is always False here; skip it (and its sync) entirely.
                        _cont_id = cont_ids[idx]
                        _cont_tokens = cont_tokens

                    cont_log_probs_sum = -F.cross_entropy(
                        lm_cont_logits, _cont_tokens, reduction="sum"
                    )

                    if self.metric_type == "pmi_dc":
                        assert dc_lm_logits is not None and dc_lens is not None
                        dc_len = dc_lens[idx]
                        dc_lm_cont_logits = dc_lm_logits[idx][dc_len - 1 : dc_len + cont_len - 1]
                        dc_cont_log_probs_sum = -F.cross_entropy(
                            dc_lm_cont_logits, _cont_tokens, reduction="sum"
                        )
                        log_likelihood = cont_log_probs_sum / dc_cont_log_probs_sum
                        celoss = -log_likelihood
                        bpb = -log_likelihood
                        log_likelihood_nls = log_likelihood
                        celoss_nls = celoss
                        bpb_nls = bpb
                    elif self.metric_type == "acc" or self.metric_type == "f1":
                        log_likelihood = cont_log_probs_sum
                        celoss = -cont_log_probs_sum / batch["cont_str_len"][idx]
                        bpb = -cont_log_probs_sum / batch["cont_byte_len"][idx] * LOG_2_OF_E
                        log_likelihood_nls = cont_log_probs_sum
                        celoss_nls = (
                            -cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                        )
                        bpb_nls = (
                            -cont_log_probs_sum
                            / batch["cont_byte_len_no_leading_space"][idx]
                            * LOG_2_OF_E
                        )
                    elif self.metric_type in ["len_norm", "ce_loss", "bpb"]:
                        log_likelihood = cont_log_probs_sum / batch["cont_str_len"][idx]
                        celoss = -cont_log_probs_sum / batch["cont_str_len"][idx]
                        bpb = -cont_log_probs_sum / batch["cont_byte_len"][idx] * LOG_2_OF_E
                        log_likelihood_nls = (
                            cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                        )
                        celoss_nls = (
                            -cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                        )
                        bpb_nls = (
                            -cont_log_probs_sum
                            / batch["cont_byte_len_no_leading_space"][idx]
                            * LOG_2_OF_E
                        )
                    else:
                        raise ValueError(self.metric_type)

                    keys.append((doc_id, _cont_id, label_ids[idx]))
                    ll.append(log_likelihood)
                    ce.append(celoss)
                    bp.append(bpb)
                    lln.append(log_likelihood_nls)
                    cen.append(celoss_nls)
                    bpn.append(bpb_nls)

            if not keys:
                return

            # Single batched host transfer for every scalar accumulated in this update call.
            ll_f = torch.stack(ll).tolist()
            ce_f = torch.stack(ce).tolist()
            bp_f = torch.stack(bp).tolist()
            lln_f = torch.stack(lln).tolist()
            cen_f = torch.stack(cen).tolist()
            bpn_f = torch.stack(bpn).tolist()
            for k, (doc_id, _cont_id, label_id) in enumerate(keys):
                self.labels.append((doc_id, _cont_id, label_id))
                self.loglikelihoods.append((doc_id, _cont_id, ll_f[k]))
                self.celosses.append((doc_id, _cont_id, ce_f[k]))
                self.bpbs.append((doc_id, _cont_id, bp_f[k]))
                self.loglikelihoods_no_leading_space.append((doc_id, _cont_id, lln_f[k]))
                self.celosses_no_leading_space.append((doc_id, _cont_id, cen_f[k]))
                self.bpbs_no_leading_space.append((doc_id, _cont_id, bpn_f[k]))

    return _FastICLMetric


class DownstreamEvaluator(Evaluator):
    metric_type_to_label = {
        "f1_v1": "F1 score",
        "acc_v1": "accuracy",
        "len_norm_v1": "length-normalized accuracy",
        "pmi_dc_v1": "PMI-DC accuracy",
        "ce_loss_v1": "CE loss",
        "bpb_v1": "BPB",
        "soft_v1": "soft loss",
        "soft_log_v1": "log soft loss",
        "f1_v2": "F1 score v2",
        "acc_v2": "accuracy v2",
        "len_norm_v2": "length-normalized accuracy v2",
        "pmi_dc_v2": "PMI-DC accuracy v2",
        "ce_loss_v2": "CE loss v2",
        "bpb_v2": "BPB v2",
        "soft_v2": "soft loss v2",
        "soft_log_v2": "log soft loss v2",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        batch_spec: EvalBatchSpec,
        tokenizer: "HFTokenizer",
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        lazy: bool = False,
        sort_by_length: bool = False,
        fast_metric: bool = True,
    ):
        from olmo_eval import ICLMetric

        if task not in _all_tasks():
            raise OLMoConfigurationError(f"Unknown downstream eval task: '{task}'")

        self.label = task
        self.batch_spec = batch_spec
        self.tokenizer = tokenizer
        self.device = device  # set here for _build_data_loader() to use
        self.dp_process_group = dp_process_group
        self.sort_by_length = sort_by_length
        self.fast_metric = fast_metric
        self.metric: Optional[ICLMetric] = None
        if lazy:
            log.info(f"Initializing lazy DownstreamEvaluator for task '{self.label}'")

        super().__init__(
            name=name,
            batches=None if lazy else self._build_data_loader(),
            batches_factory=self._build_data_loader if lazy else None,
            device=device,
        )

    @property
    def display_name(self) -> str:
        return f"{self.name} '{self.label}'"

    def _build_data_loader(self) -> DataLoader:
        from olmo_eval import ICLMetric, ICLMultiChoiceTaskDataset, build_task

        log.info(f"Building downstream eval task dataset for '{self.label}'...")

        task_dataset: ICLMultiChoiceTaskDataset
        if self.batch_spec.fixed_sequence_length:
            assert self.batch_spec.max_sequence_length is not None
            task_dataset = build_task(
                self.label,
                self.tokenizer,
                model_ctx_len=self.batch_spec.max_sequence_length,
                fixed_ctx_len=True,
            )
        elif self.batch_spec.max_sequence_length is not None:
            task_dataset = build_task(
                self.label, self.tokenizer, model_ctx_len=self.batch_spec.max_sequence_length
            )
        else:
            task_dataset = build_task(self.label, self.tokenizer)

        metric_cls = ICLMetric
        if self.fast_metric:
            try:
                metric_cls = _fast_iclmetric_class()
            except Exception as e:
                # Never let the optimization break eval — fall back to the stock metric.
                log.warning(
                    f"Could not build fast ICLMetric ({e!r}); falling back to olmo_eval.ICLMetric."
                )
                metric_cls = ICLMetric
        self.metric = metric_cls(metric_type=task_dataset.metric_type).to(
            self.device or get_default_device()
        )
        # Determine this rank's set of instance indices, preserving the exact sharding behavior of
        # DistributedSampler so the set of instances each rank evaluates (and therefore the reduced
        # metric) is unchanged. Only the *grouping* of indices into batches changes below.
        if is_distributed():
            sampler: DistributedSampler = DistributedSampler(
                task_dataset,  # type: ignore
                drop_last=False,
                shuffle=False,
                num_replicas=get_world_size(self.dp_process_group),
                rank=get_rank(self.dp_process_group),
            )
            rank_indices = list(sampler)
        else:
            rank_indices = list(range(len(task_dataset)))  # type: ignore

        if (
            self.batch_spec.max_sequence_length is not None
            and task_dataset.max_sequence_length > self.batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for downstream eval task '{self.label}' ({task_dataset.max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({self.batch_spec.max_sequence_length:,d} tokens)"
            )

        rank_batch_size_instances: int
        if self.batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            rank_batch_size_instances = self.batch_spec.rank_batch_size
        elif self.batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            if self.batch_spec.fixed_sequence_length:
                assert self.batch_spec.max_sequence_length is not None
                if self.batch_spec.rank_batch_size % self.batch_spec.max_sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"The eval batch size ({self.batch_spec.rank_batch_size} tokens) must be divisible "
                        f"by the maximum eval sequence length ({self.batch_spec.max_sequence_length:,d} tokens)"
                    )
                rank_batch_size_instances = (
                    self.batch_spec.rank_batch_size // self.batch_spec.max_sequence_length
                )
            else:
                rank_batch_size_instances = (
                    self.batch_spec.rank_batch_size // task_dataset.max_sequence_length
                )
        else:
            raise NotImplementedError(self.batch_spec.batch_size_unit)

        # Optionally (opt-in, single-rank only) group similar-length instances together and pack
        # each batch up to a token budget. ICL tasks have highly variable sequence lengths, so
        # length-heterogeneous batches waste compute on padding (a batch of mostly-short sequences
        # still pads to the longest one). Length bucketing removes that waste without changing any
        # per-instance score (see _pack_batches_by_length), but yields a per-rank-dependent number
        # of batches, so it must not be used under distributed/FSDP eval (collectives would desync).
        # Skipped for fixed_sequence_length (e.g. TP), where every query is padded to full ctx len.
        batch_sampler: List[List[int]]
        if self.sort_by_length and not self.batch_spec.fixed_sequence_length:
            lengths = [len(sample["query"]) for sample in task_dataset.samples]  # type: ignore
            if self.batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
                token_budget = self.batch_spec.rank_batch_size
            else:
                token_budget = rank_batch_size_instances * task_dataset.max_sequence_length
            batch_sampler = _pack_batches_by_length(rank_indices, lengths, token_budget)
            log.info(
                f"Using length-bucketed eval batching for downstream task '{self.label}': "
                f"{len(batch_sampler)} batches over {len(rank_indices)} instances "
                f"(token budget {token_budget:,d}, max sequence length {task_dataset.max_sequence_length:,d})"
            )
        else:
            batch_sampler = [
                rank_indices[i : i + rank_batch_size_instances]
                for i in range(0, len(rank_indices), rank_batch_size_instances)
            ]
            log.info(
                f"Using per-rank batch size of {rank_batch_size_instances} instances "
                f"for downstream eval task '{self.label}' with max sequence length {task_dataset.max_sequence_length:,d} tokens"
            )

        return DataLoader(
            task_dataset,  # type: ignore
            batch_sampler=batch_sampler,
            collate_fn=task_dataset.collate_fn,
            num_workers=0,
        )

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
    ) -> None:
        del ce_loss
        assert self.metric is not None
        if logits is None:
            raise RuntimeError(
                "Downstream evaluators require full logits, but logits are None. "
                "This happens when context parallelism (CP) or tensor parallelism (TP) is enabled. "
                "Please disable downstream evals when using CP or TP."
            )
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        assert self.metric is not None
        metric_type_to_value = self.metric.compute()
        outputs = {}
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        if self.metric is not None:
            self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    tasks: List[str]
    tokenizer: TokenizerConfig
    eval_interval: Optional[int] = 1000
    fixed_steps: Optional[List[int]] = None
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_on_startup: bool = False
    eval_on_finish: bool = False
    cancel_after_first_eval: bool = False
    log_interval: int = 5
    lazy: bool = False
    enabled: bool = True
    sort_by_length: bool = False
    """
    Group similar-length instances into batches to remove padding waste. This does not change eval
    results (see :func:`_pack_batches_by_length`), but it produces a per-rank-dependent number of
    batches, so it is **only safe for single-rank eval**: under FSDP/distributed eval every rank must
    run the same number of forward passes or the parameter all-gathers will desync and hang. Leave
    ``False`` (fixed-size batching, equal batch counts per rank) for distributed runs.
    """
    fast_metric: bool = True
    """
    Use a sync-deferred drop-in for ``olmo_eval.ICLMetric`` that computes identical metrics with far
    fewer host-device syncs (see :func:`_fast_iclmetric_class`). Safe under distributed eval (it only
    changes when scalars are copied to the host, not the math). Falls back to the stock metric if it
    cannot be built.
    """

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_eval import HFTokenizer

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List[Evaluator] = []
        for task in sorted(self.tasks):
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                    lazy=self.lazy,
                    sort_by_length=self.sort_by_length,
                    fast_metric=self.fast_metric,
                )
            )

        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
            eval_on_finish=self.eval_on_finish,
        )
