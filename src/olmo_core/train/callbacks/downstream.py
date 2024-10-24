from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union
from typing import TYPE_CHECKING, List, Optional
import abc
import os
import json, gzip

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import importlib_resources
from importlib_resources.abc import Traversable
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from sklearn.metrics import f1_score
from torchmetrics import Metric
import datasets

from olmo.util import load_hf_dataset
from olmo.tokenizer import Tokenizer

from dataclasses import dataclass, field
from olmo_core.distributed.utils import (
    get_world_size, 
    get_default_device, 
    get_fs_local_rank, 
    get_rank, 
)
from olmo_core.eval import Evaluator
from olmo_core.exceptions import OLMoConfigurationError
import logging
from ..common import Duration
from .callback import Callback, CallbackConfig
from .evaluator_callback import EvaluatorCallback

if TYPE_CHECKING:
    from ..trainer import Trainer
log = logging.getLogger(__name__)


# Map from oe-eval metrics to metrics used here
METRIC_FROM_OE_EVAL = {"acc_raw": "acc", "acc_per_char": "len_norm", "acc_uncond": "pmi_dc"}
LOG_2_OF_E = 1.44269504089


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type="acc") -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type

        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.loglikelihoods = []
        self.labels = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        if self.metric_type == "pmi_dc":
            assert dc_lm_logits is not None, "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
            # [cont_len]: continuation is padded for batching
            cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]]
            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][
                batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
            ]

            log_likelihood: torch.Tensor
            if self.metric_type == "pmi_dc":
                assert dc_lm_logits is not None
                # get domain conditional continuation logits: [cont_len, vocab]
                dc_lm_cont_logits = dc_lm_logits[idx][
                    batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                ]

                # gather log-probs at continuation token indices but divide by domain conditional prob
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / torch.gather(dc_lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                )
            elif self.metric_type == "acc" or self.metric_type == "f1":
                # gather log-probs at continuation token indices
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
            elif self.metric_type == "len_norm" or self.metric_type == "ce_loss":
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum() / batch["cont_str_len"][idx]
                )
                if self.metric_type == "ce_loss":
                    log_likelihood = -log_likelihood
            elif self.metric_type == "bpb":
                # bits per byte
                log_likelihood = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            else:
                raise ValueError(self.metric_type)

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(
                torch.Tensor((doc_id, cont_id, log_likelihood)).to(batch["continuation"][idx].device)
            )
            self.labels.append(
                torch.LongTensor((doc_id, cont_id, batch["label_id"][idx])).to(batch["label_id"][idx].device)
            )

    def compute(self) -> torch.Tensor:
        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id.item() not in label_dict:
                label_dict[doc_id.item()] = label_id.item()

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if int(doc_id.item()) not in loglikelihood_dict:
                loglikelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in loglikelihood_dict[int(doc_id.item())]:
                loglikelihood_dict[int(doc_id.item())][int(cont_id.item())] = loglikelihood

        # compute acc
        correct = []
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue
            if self.metric_type in ["ce_loss", "bpb"]:
                correct.append(loglikelihoods[0])  # Only one answer is scored
            else:
                correct.append(1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0)

            if self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                preds.append(torch.argmax(loglikelihoods).item())
                labels.append(label_dict[doc_id])

        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = f1_score(labels, preds, pos_label=0, average="macro")
        else:
            score = sum(correct) / len(correct)

        return torch.tensor(score)


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts
        self.current_prompt = None
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    # log.info(
                    #     f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                    #     + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    # )

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    query = ctx + continuation[:-1]
                    query = query[-self.model_ctx_len :]
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    dc_query = dc + continuation[:-1]

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "cont_byte_len": cont_byte_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                        }
                    )

                doc_id += 1
    
    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample["ctx"]) > max_ctx_len:
                max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["query"], max_len=max_query_len)))
            dc_queries.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len))
            )

            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
        }

        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> List[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class OEEvalTask(ICLMultiChoiceTaskDataset):
    """Generic class for OE evaluation tasks"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split=None,
        metric_type=None,
        prompts=[None],  # List of prompt variants to use
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        requests_list = []
        configs = []
        for ds_name in dataset_names:
            config, requests = load_oe_eval_requests(self.dataset_path, ds_name, split)
            requests_list.append(requests)
            configs.append(config)
        if metric_type is not None:
            self.metric_type = metric_type
        else:
            # Use metric type from associated task config
            for config in configs:
                if config is not None:
                    metric_type_raw = config["task_config"].get("primary_metric")
                    if metric_type_raw is not None:
                        # acc, len_norm, pmi_dc
                        metric_type = METRIC_FROM_OE_EVAL[metric_type_raw]
                        if self.metric_type is not None and self.metric_type != metric_type:
                            raise ValueError(f"Conflicting metric types: {self.metric_type} and {metric_type}")
                        self.metric_type = metric_type
        self.dataset = requests_list

        # prep examples
        self.prep_examples()

    def prep_examples(self):
        current_doc_id_offset = 0
        max_doc_id = 0
        for requests in self.dataset:
            current_doc_id_offset += max_doc_id
            max_doc_id = 0  # Max doc id seen in this dataset
            for request in requests:
                doc = request["doc"]
                doc_id = request["doc_id"]
                if doc_id >= 1000000:
                    # Hacky implementation of unconditional requests in oe-eval
                    # Not supported here for now
                    continue
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                assert (
                    request["request_type"] == "loglikelihood"
                ), f"Unsupported request type: {request['request_type']}"

                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                request_dict = request["request"]
                continuation_str = request_dict["continuation"]
                label_id = request["label"]
                cont_id = request["idx"]
                if self.metric_type in ["ce_loss", "bpb"]:
                    if label_id != cont_id:
                        # Skip non-target continuations for ce_loss and bpb
                        continue
                    else:
                        # Treat as instance with just one continuation
                        cont_id = 0
                        label_id = 0
                doc_text = request_dict["context"]
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}):"
                        + f"\ndoc_text: {doc_text}\ncontinuation: {continuation_str}"
                    )
                cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                continuation = self.token_encode(continuation_str)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]
                # this will be different from len(ctx) when truncated by model_ctx_len
                actual_ctx_len = len(query) - len(continuation) + 1

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id + current_doc_id_offset,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": actual_ctx_len,
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

    def doc_to_text(self, doc) -> str:
        raise NotImplementedError

    def doc_to_continuations(self, doc) -> List[str]:
        raise NotImplementedError

    def doc_to_label(self, doc) -> int:
        raise NotImplementedError


class DownstreamEvaluator(Evaluator):
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
        metrics: List[Metric],
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            name=name, batches=batches, device=device, dp_process_group=dp_process_group
        )
        assert len(labels) == len(metrics)
        self.metrics = {
            label: metric for (label, metric) in zip(labels, metrics)
        }
    
    @classmethod
    def from_oe_dataset(
        cls,
        dataset: OEEvalTask,
        *,
        name: str,
        batch_size: int,
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        seed: int = 0,
        # num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
    ) -> "DownstreamEvaluator":
        ds_eval_sampler = DistributedSampler(
            dataset,
            drop_last=False,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=seed,
        )
        ds_eval_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            sampler=ds_eval_sampler,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            # persistent_workers=data_config.persistent_workers,
            # timeout=data_config.timeout,
        )
        metric = ICLMetric(metric_type=dataset.metric_type)

        evaluator = DownstreamEvaluator(
            name=name,
            batches=ds_eval_dataloader,
            labels=[name],
            metrics=[metric.to(device)],
            device=device,
            dp_process_group=dp_process_group,
        )
        return evaluator

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: torch.Tensor, logits: torch.Tensor
    ) -> None:
        del ce_loss
        for metric in self.metrics.values():
            metric.update(batch, logits)  # type: ignore

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for label in sorted(self.metrics.keys()):
            metric = self.metrics[label]
            key = f"eval/downstream/{label}_{metric.metric_type}"
            if metric.metric_type in ["ce_loss", "bpb"]:
                key = key.replace("/downstream/", f"/downstream_{metric.metric_type}/")
            out[f'{label}/{key}'] = metric.compute()
        return out

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    labels: List[str]
    tokenizer: str
    eval_batch_size: int
    eval_interval: int = 1000
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5

    def build(self, trainer: "Trainer") -> Callback:
        evaluators = []
        for label in self.labels:
            task_class = label_to_task_map[label]
            task_class, task_kwargs = task_class
            dataset = task_class(tokenizer=Tokenizer.from_pretrained(self.tokenizer), **task_kwargs)

            evaluator = DownstreamEvaluator.from_oe_dataset(
                dataset,
                name=label,
                batch_size=self.eval_batch_size,
                device=trainer.device,
            )
            evaluators.append(evaluator)
        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )

def load_oe_eval_requests(path: str, name: Optional[str] = None, split: Optional[str] = None):
    """
    Loads an oe-eval request file from `olmo_data/oe_eval_tasks`.
    TODO: Add support from loading from S3 instead?
    """
    dataset_rel_path = os.path.join("oe_eval_tasks", path)
    if name is not None:
        dataset_rel_path = os.path.join(dataset_rel_path, name)
    with get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"OE Eval dataset not found in directory {dataset_rel_path}")
        data_file = dataset_path / "requests.jsonl.gz"
        if not data_file.is_file():
            data_file = dataset_path / "requests.jsonl"
        if not data_file.is_file():
            raise FileNotFoundError(
                f"OE Eval dataset file requests-{split}.jsonl(.gz) missing in directory {dataset_rel_path}"
            )
        requests = []
        if data_file.suffix == ".gz":
            with gzip.open(data_file, "r") as file:
                for line in file:
                    requests.append(json.loads(line.decode("utf-8").strip()))
        else:
            with open(data_file, "r") as file:
                for line2 in file:
                    requests.append(json.loads(line2.strip()))
        config = None
        config_file = dataset_path / "config.json"
        if config_file.is_file():
            with open(config_file, "r") as file:
                config = json.load(file)
        return config, requests

def _get_data_traversable(data_rel_path: str) -> Traversable:
    return importlib_resources.files("eval").joinpath(data_rel_path)

@contextmanager
def get_data_path(data_rel_path: str) -> Generator[Path, None, None]:
    try:
        with importlib_resources.as_file(_get_data_traversable(data_rel_path)) as path:
            yield path
    finally:
        pass

label_to_task_map = {
    # "pubmedqa_rc": (OEEvalTask, {"dataset_path": "pubmedqa", "dataset_name": "rc_3shot", "metric_type": "acc"}),
    "pubmedqa_mc": (OEEvalTask, {"dataset_path": "pubmedqa", "dataset_name": "mc_3shot", "metric_type": "acc"}),
    "scifact_rc": (OEEvalTask, {"dataset_path": "scifact", "dataset_name": "rc_3shot", "metric_type": "acc"}),
    # "scifact_mc": (OEEvalTask, {"dataset_path": "scifact", "dataset_name": "mc_3shot", "metric_type": "acc"}),
    # "covidfact_rc": (OEEvalTask, {"dataset_path": "covidfact", "dataset_name": "rc_2shot", "metric_type": "acc"}),
    # "covidfact_mc": (OEEvalTask, {"dataset_path": "covidfact", "dataset_name": "mc_2shot", "metric_type": "acc"}),
    # "healthver_rc": (OEEvalTask, {"dataset_path": "healthver", "dataset_name": "rc_3shot", "metric_type": "acc"}),
    # "healthver_mc": (OEEvalTask, {"dataset_path": "healthver", "dataset_name": "mc_3shot", "metric_type": "acc"}),
    # "scicite_rc": (OEEvalTask, {"dataset_path": "scicite", "dataset_name": "rc_3shot", "metric_type": "acc"}),
    # "scicite_mc": (OEEvalTask, {"dataset_path": "scicite", "dataset_name": "mc_3shot", "metric_type": "acc"}),
    # "acl_arc_rc": (OEEvalTask, {"dataset_path": "acl_arc", "dataset_name": "rc_6shot", "metric_type": "acc"}),
    # "acl_arc_mc": (OEEvalTask, {"dataset_path": "acl_arc", "dataset_name": "mc_6shot", "metric_type": "acc"}),
    # "datafinder_reco_sc_rc": (OEEvalTask, {"dataset_path": "datafinder_reco_sc", "dataset_name": "rc_5shot", "metric_type": "acc"}),
    # "datafinder_reco_sc_mc": (OEEvalTask, {"dataset_path": "datafinder_reco_sc", "dataset_name": "mc_5shot", "metric_type": "acc"}),
}
