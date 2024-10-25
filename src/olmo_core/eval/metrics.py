from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..config import StrEnum
from ..distributed.utils import all_gather, all_reduce_value
from ..utils import get_default_device, move_to_device

__all__ = ["Metric", "MeanMetric", "ICLMetric", "ICLMetricType"]


LOG_2_OF_E = 1.44269504089


class Metric(metaclass=ABCMeta):
    """
    Base class for evaluation metrics.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.device = device if device is not None else get_default_device()
        self.process_group = process_group

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the metric.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Compute the metric.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric.
        """
        raise NotImplementedError

    def as_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        return value.to(device=self.device, non_blocking=self.device.type != "cpu")


class MeanMetric(Metric):
    """
    Computes the mean over a stream of values.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(device=device, process_group=process_group)
        self.weighted_sum = torch.tensor(0.0, device=self.device)
        self.weight = torch.tensor(0.0, device=self.device)

    def update(
        self, value: Union[float, torch.Tensor], weight: Union[float, torch.Tensor] = 1.0
    ) -> None:
        """
        :param value: The latest value to update the metric with. Could be a tensor of values.
        :param weight: The corresponding weight(s) for the value. Should be the same shape as ``value``.
        """
        value = self.as_tensor(value)
        weight = torch.broadcast_to(self.as_tensor(weight), value.shape)
        if value.numel() == 0:
            return
        self.weighted_sum += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the mean over the values and weights given.
        """
        weighted_sum = all_reduce_value(
            self.weighted_sum, device=self.device, group=self.process_group
        )
        weight = all_reduce_value(self.weight, device=self.device, group=self.process_group)
        return weighted_sum / weight

    def reset(self) -> None:
        self.weighted_sum.zero_()
        self.weight.zero_()


class ICLMetricType(StrEnum):
    """
    An enumeration of the different metric types used for in-context learning (ICL) tasks.
    """

    f1 = "f1"
    """
    F1 score.
    """

    acc = "acc"
    """
    Accuracy computed from raw log-likelihood scores.
    """

    len_norm = "len_norm"
    """
    Accuracy computed from length-normalized log-likelihood scores.
    """

    pmi_dc = "pmi_dc"
    """
    Accuracy computed from `domain conditional pointwise mutual information
    <https://arxiv.org/pdf/2104.08315>`_ scores.
    """

    ce_loss = "ce_loss"
    """
    Cross-entropy loss.
    """

    bpb = "bpb"
    """
    Bits per byte.
    """


class ICLMetric(Metric):
    """
    A general metric for in-context learning (ICL) tasks which implements several different
    variants defined by :class:`ICLMetricType`. This is used by :class:`ICLEvaluator`.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        metric_type: ICLMetricType = ICLMetricType.acc,
    ):
        super().__init__(device=device, process_group=process_group)
        self.metric_type = ICLMetricType(metric_type)
        # A list of "tuple tensors" that contain (doc_id, cont_id, log_likelihood)
        self.log_likelihoods: List[torch.Tensor] = []
        # A list of "tuple tensors" that contain (doc_id, cont_id, label_id)
        self.labels: List[torch.Tensor] = []

    def update(
        self,
        batch: Dict[str, Any],
        lm_logits: torch.Tensor,
        dc_lm_logits: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update the metric.

        :param batch: The eval batch. This requires specific fields in the batch.
        :param lm_logits: The language modeling logits for each item in the batch.
        :param dc_lm_logits: The "domain conditional" language modeling logits, required when the metric type
            is :data:`ICLMetricType.pmi_dc`. These logits are computed over each answer string along
            with a domain premise.
        """
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        if self.metric_type == ICLMetricType.pmi_dc:
            assert (
                dc_lm_logits is not None
            ), f"{ICLMetricType.pmi_dc} acc type selected but no domain conditional logits provided"

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
            if self.metric_type == ICLMetricType.pmi_dc:
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
            elif self.metric_type == ICLMetricType.acc or self.metric_type == ICLMetricType.f1:
                # gather log-probs at continuation token indices
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
            elif (
                self.metric_type == ICLMetricType.len_norm
                or self.metric_type == ICLMetricType.ce_loss
            ):
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_str_len"][idx]
                )
                if self.metric_type == ICLMetricType.ce_loss:
                    log_likelihood = -log_likelihood
            elif self.metric_type == ICLMetricType.bpb:
                # bits per byte
                log_likelihood = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            else:
                raise NotImplementedError(self.metric_type)

            self.log_likelihoods.append(
                torch.Tensor((doc_id, cont_id, log_likelihood)).to(
                    batch["continuation"][idx].device
                )
            )
            self.labels.append(
                torch.LongTensor((doc_id, cont_id, batch["label_id"][idx])).to(
                    batch["label_id"][idx].device
                )
            )

    def compute(self) -> torch.Tensor:
        all_log_likelihoods = move_to_device(
            torch.cat(
                all_gather(
                    move_to_device(torch.stack(self.log_likelihoods), self.device),
                    group=self.process_group,
                )
            ),
            torch.device("cpu"),
        )
        all_labels = move_to_device(
            torch.cat(
                all_gather(
                    move_to_device(torch.stack(self.labels), self.device),
                    group=self.process_group,
                )
            ),
            torch.device("cpu"),
        )

        log_likelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict: Dict[int, int] = {}

        # collect log likelihoods by doc ID.
        for doc_id, cont_id, log_likelihood in all_log_likelihoods:
            if int(doc_id.item()) not in log_likelihood_dict:
                log_likelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in log_likelihood_dict[int(doc_id.item())]:
                log_likelihood_dict[int(doc_id.item())][int(cont_id.item())] = log_likelihood  # type: ignore

        # collect labels by doc ID.
        for doc_id, cont_id, label_id in all_labels:
            if doc_id.item() not in label_dict:
                label_dict[doc_id.item()] = label_id.item()  # type: ignore

        # compute acc
        correct = []
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            labels = []

        for doc_id in log_likelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(log_likelihood_dict[doc_id].keys())
            log_likelihoods = torch.tensor([-float("inf")] * num_continuations)

            skip_document = False
            for cont_id in log_likelihood_dict[doc_id]:
                try:
                    log_likelihoods[cont_id] = log_likelihood_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue
            if self.metric_type in ["ce_loss", "bpb"]:
                correct.append(log_likelihoods[0])  # Only one answer is scored
            else:
                correct.append(
                    1.0 if torch.argmax(log_likelihoods).item() == label_dict[doc_id] else 0.0
                )

            if self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                preds.append(torch.argmax(log_likelihoods).item())
                labels.append(label_dict[doc_id])

        if self.metric_type == "f1":
            from sklearn.metrics import f1_score

            assert preds is not None
            assert labels is not None

            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = f1_score(labels, preds, pos_label=0)
        else:
            score = sum(correct) / len(correct)

        return torch.tensor(score)

    def reset(self) -> None:
        self.log_likelihoods.clear()
        self.labels.clear()
