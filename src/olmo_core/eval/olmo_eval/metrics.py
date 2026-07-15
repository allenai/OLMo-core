import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from .util import all_gather_object

LOG_2_OF_E = 1.44269504089


log = logging.getLogger(__name__)


T = TypeVar("T")


def dist_combine_lists(x: List[T]) -> List[T]:
    all_lists = all_gather_object(x)
    return [item for sublist in all_lists for item in sublist]


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type="acc") -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type

        self.add_state("loglikelihoods", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("celosses", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("bpbs", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("labels", default=[], dist_reduce_fx=dist_combine_lists)

        self.add_state(
            "loglikelihoods_no_leading_space", default=[], dist_reduce_fx=dist_combine_lists
        )
        self.add_state("celosses_no_leading_space", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("bpbs_no_leading_space", default=[], dist_reduce_fx=dist_combine_lists)

    def reset(self):
        self.loglikelihoods: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.celosses: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.bpbs: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.labels: List[Tuple[Optional[int], Optional[int], Optional[int]]] = []

        self.loglikelihoods_no_leading_space: List[
            Tuple[Optional[int], Optional[int], Optional[float]]
        ] = []
        self.celosses_no_leading_space: List[
            Tuple[Optional[int], Optional[int], Optional[float]]
        ] = []
        self.bpbs_no_leading_space: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []

    def update(
        self,
        batch: Dict[str, Any],
        lm_logits: Optional[torch.Tensor] = None,
        dc_lm_logits: Optional[torch.Tensor] = None,
    ):
        # NOTE: `lm_logits` could be none for some ranks if using pipeline parallelism. We still
        # need to add something to these state lists though in order for them to get synchronized
        # for reasons not clear to me other than the fact that torchmetrics is jenky a.f.
        if lm_logits is None:
            self.loglikelihoods.append((None, None, None))
            self.celosses.append((None, None, None))
            self.bpbs.append((None, None, None))

            self.loglikelihoods_no_leading_space.append((None, None, None))
            self.celosses_no_leading_space.append((None, None, None))
            self.bpbs_no_leading_space.append((None, None, None))

            self.labels.append((None, None, None))
            return

        if self.metric_type == "pmi_dc":
            assert (
                dc_lm_logits is not None
            ), "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
            doc_id = int(doc_id)
            cont_id = int(cont_id)

            # [cont_len]: continuation is padded for batching
            cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]]
            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][
                batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
            ]

            if "choice_ids" in batch:
                fast_mc = True
                choice_ids = batch["choice_ids"][idx]
            else:
                fast_mc = False
                choice_ids = [cont_tokens]

            # For each choice token, calculate metrics and append as separate entries
            for choice_idx, choice_token in enumerate(choice_ids):
                if fast_mc:
                    _cont_id = choice_idx
                    _cont_tokens = choice_token.unsqueeze(-1)
                else:
                    _cont_id = cont_id
                    _cont_tokens = cont_tokens

                # Skip choices for Qs with less than the max choices (for questions w/ different nubmers of choices)
                is_empty_choice = (choice_token.unsqueeze(-1).unsqueeze(-1) == -1).all().item()
                if is_empty_choice:
                    continue

                # Compute the sum of log probabilities at continuation token indices.
                # NOTE: -F.cross_entropy("sum") is equivalent to gather(log_softmax(logits), indices).sum()
                cont_log_probs_sum = -F.cross_entropy(lm_cont_logits, _cont_tokens, reduction="sum")

                log_likelihood: torch.Tensor
                celoss: torch.Tensor
                bpb: torch.Tensor
                log_likelihood_no_leading_space: torch.Tensor
                celoss_no_leading_space: torch.Tensor
                bpb_no_leading_space: torch.Tensor
                if self.metric_type == "pmi_dc":
                    assert dc_lm_logits is not None
                    # get domain conditional continuation logits: [cont_len, vocab]
                    dc_lm_cont_logits = dc_lm_logits[idx][
                        batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                    ]

                    dc_cont_log_probs_sum = -F.cross_entropy(
                        dc_lm_cont_logits, _cont_tokens, reduction="sum"
                    )

                    # gather log-probs at continuation token indices but divide by domain conditional prob
                    log_likelihood = cont_log_probs_sum / dc_cont_log_probs_sum
                    celoss = -log_likelihood
                    bpb = -log_likelihood  # the normalization factors cancel out

                    log_likelihood_no_leading_space = log_likelihood
                    celoss_no_leading_space = celoss
                    bpb_no_leading_space = bpb
                elif self.metric_type == "acc" or self.metric_type == "f1":
                    # gather log-probs at continuation token indices
                    log_likelihood = cont_log_probs_sum
                    celoss = -cont_log_probs_sum / batch["cont_str_len"][idx]
                    bpb = -cont_log_probs_sum / batch["cont_byte_len"][idx] * LOG_2_OF_E

                    log_likelihood_no_leading_space = cont_log_probs_sum
                    celoss_no_leading_space = (
                        -cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                    )
                    bpb_no_leading_space = (
                        -cont_log_probs_sum
                        / batch["cont_byte_len_no_leading_space"][idx]
                        * LOG_2_OF_E
                    )
                elif self.metric_type in ["len_norm", "ce_loss", "bpb"]:
                    log_likelihood = cont_log_probs_sum / batch["cont_str_len"][idx]
                    celoss = -cont_log_probs_sum / batch["cont_str_len"][idx]
                    bpb = -cont_log_probs_sum / batch["cont_byte_len"][idx] * LOG_2_OF_E

                    log_likelihood_no_leading_space = (
                        cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                    )
                    celoss_no_leading_space = (
                        -cont_log_probs_sum / batch["cont_str_len_no_leading_space"][idx]
                    )
                    bpb_no_leading_space = (
                        -cont_log_probs_sum
                        / batch["cont_byte_len_no_leading_space"][idx]
                        * LOG_2_OF_E
                    )
                else:
                    raise ValueError(self.metric_type)

                self.labels.append((doc_id, _cont_id, int(batch["label_id"][idx])))
                self.loglikelihoods.append((doc_id, _cont_id, float(log_likelihood)))
                self.celosses.append((doc_id, _cont_id, float(celoss)))
                self.bpbs.append((doc_id, _cont_id, float(bpb)))

                self.loglikelihoods_no_leading_space.append(
                    (doc_id, _cont_id, float(log_likelihood_no_leading_space))
                )
                self.celosses_no_leading_space.append(
                    (doc_id, _cont_id, float(celoss_no_leading_space))
                )
                self.bpbs_no_leading_space.append((doc_id, _cont_id, float(bpb_no_leading_space)))

    def compute(self) -> Dict[str, torch.Tensor]:
        # Task "suffix" -> tensor

        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        loglikelihood_no_leading_space_dict: Dict[int, Dict[int, float]] = {}
        label_dict: Dict[int, int] = {}
        celoss_dict: Dict[int, Dict[int, float]] = {}
        celoss_no_leading_space_dict: Dict[int, Dict[int, float]] = {}
        bpb_dict: Dict[int, Dict[int, float]] = {}
        bpb_no_leading_space_dict: Dict[int, Dict[int, float]] = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id is None or cont_id is None or label_id is None:
                continue

            if doc_id not in label_dict:
                label_dict[doc_id] = label_id

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if doc_id is None or cont_id is None or loglikelihood is None:
                continue

            if doc_id not in loglikelihood_dict:
                loglikelihood_dict[doc_id] = {}

            if cont_id not in loglikelihood_dict[doc_id]:
                loglikelihood_dict[doc_id][cont_id] = loglikelihood

        # collect loglikelihoods no leading space
        for doc_id, cont_id, loglikelihood in self.loglikelihoods_no_leading_space:
            if doc_id is None or cont_id is None or loglikelihood is None:
                continue

            if doc_id not in loglikelihood_no_leading_space_dict:
                loglikelihood_no_leading_space_dict[doc_id] = {}

            if cont_id not in loglikelihood_no_leading_space_dict[doc_id]:
                loglikelihood_no_leading_space_dict[doc_id][cont_id] = loglikelihood

        # collect celosses
        for doc_id, cont_id, celoss_val in self.celosses:
            if doc_id is None or cont_id is None or celoss_val is None:
                continue

            if doc_id not in celoss_dict:
                celoss_dict[doc_id] = {}

            if cont_id not in celoss_dict[doc_id]:
                celoss_dict[doc_id][cont_id] = celoss_val

        # collect celosses no leading space
        for doc_id, cont_id, celoss_val in self.celosses_no_leading_space:
            if doc_id is None or cont_id is None or celoss_val is None:
                continue

            if doc_id not in celoss_no_leading_space_dict:
                celoss_no_leading_space_dict[doc_id] = {}

            if cont_id not in celoss_no_leading_space_dict[doc_id]:
                celoss_no_leading_space_dict[doc_id][cont_id] = celoss_val

        # collect bpbs
        for doc_id, cont_id, bpb_val in self.bpbs:
            if doc_id is None or cont_id is None or bpb_val is None:
                continue

            if doc_id not in bpb_dict:
                bpb_dict[doc_id] = {}

            if cont_id not in bpb_dict[doc_id]:
                bpb_dict[doc_id][cont_id] = bpb_val

        # collect bpbs no leading space
        for doc_id, cont_id, bpb_val in self.bpbs_no_leading_space:
            if doc_id is None or cont_id is None or bpb_val is None:
                continue

            if doc_id not in bpb_no_leading_space_dict:
                bpb_no_leading_space_dict[doc_id] = {}

            if cont_id not in bpb_no_leading_space_dict[doc_id]:
                bpb_no_leading_space_dict[doc_id][cont_id] = bpb_val

        # compute acc
        correct_no_leading_space = []
        correct = []
        celoss = []
        celoss_no_leading_space = []
        bpb = []
        bpb_no_leading_space = []
        soft_score = []
        soft_log_score = []
        soft_score_no_leading_space = []
        soft_log_score_no_leading_space = []
        preds: Optional[List[float]] = None
        preds_no_leading_space: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            preds_no_leading_space = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)
            loglikelihoods_no_leading_space = torch.tensor([-float("inf")] * num_continuations)
            celosses = torch.tensor([float("inf")] * num_continuations)
            celosses_no_leading_space = torch.tensor([float("inf")] * num_continuations)
            bpbs = torch.tensor([float("inf")] * num_continuations)
            bpbs_no_leading_space = torch.tensor([float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                    loglikelihoods_no_leading_space[cont_id] = loglikelihood_no_leading_space_dict[
                        doc_id
                    ][cont_id]
                    celosses[cont_id] = celoss_dict[doc_id][cont_id]
                    celosses_no_leading_space[cont_id] = celoss_no_leading_space_dict[doc_id][
                        cont_id
                    ]
                    bpbs[cont_id] = bpb_dict[doc_id][cont_id]
                    bpbs_no_leading_space[cont_id] = bpb_no_leading_space_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue

            if self.metric_type == "ce_loss":
                celoss.append(celosses[0])  # Only one answer is scored
                celoss_no_leading_space.append(celosses_no_leading_space[0])
            elif self.metric_type == "bpb":
                bpb.append(bpbs[0])  # Only one answer is scored
                bpb_no_leading_space.append(bpbs_no_leading_space[0])
            elif self.metric_type == "f1":
                assert preds is not None
                assert preds_no_leading_space is not None
                assert labels is not None
                preds.append(torch.argmax(loglikelihoods).item())
                preds_no_leading_space.append(torch.argmax(loglikelihoods_no_leading_space).item())
                labels.append(label_dict[doc_id])
            else:
                correct.append(
                    1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0
                )
                correct_no_leading_space.append(
                    1.0
                    if torch.argmax(loglikelihoods_no_leading_space).item() == label_dict[doc_id]
                    else 0.0
                )
                celoss.append(celosses[label_dict[doc_id]].item())
                celoss_no_leading_space.append(celosses_no_leading_space[label_dict[doc_id]].item())
                bpb.append(bpbs[label_dict[doc_id]].item())
                bpb_no_leading_space.append(bpbs_no_leading_space[label_dict[doc_id]].item())
                soft_score.append(torch.softmax(loglikelihoods, dim=0)[label_dict[doc_id]].item())
                soft_log_score.append(
                    torch.log_softmax(loglikelihoods, dim=0)[label_dict[doc_id]].item()
                )
                soft_score_no_leading_space.append(
                    torch.softmax(loglikelihoods_no_leading_space, dim=0)[label_dict[doc_id]].item()
                )
                soft_log_score_no_leading_space.append(
                    torch.log_softmax(loglikelihoods_no_leading_space, dim=0)[
                        label_dict[doc_id]
                    ].item()
                )

        # v1 vs. v2 corresponds to whether we add a 1 to the num chars or num bytes when normalizing the answer length. See https://github.com/allenai/OLMo-in-loop-evals/pull/6

        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = self.custom_f1_score(labels, preds, pos_label=0)
            score_no_leading_space = self.custom_f1_score(
                labels, preds_no_leading_space, pos_label=0
            )
            return {
                "f1_v1": torch.tensor(score),
                "f1_v2": torch.tensor(score_no_leading_space),
            }
        elif self.metric_type == "ce_loss":
            return {
                "ce_loss_v1": torch.tensor(
                    sum(celoss_no_leading_space) / len(celoss_no_leading_space)
                ),
                "ce_loss_v2": torch.tensor(sum(celoss) / len(celoss)),
            }
        elif self.metric_type == "bpb":
            return {
                "bpb_v1": (sum(bpb_no_leading_space) / len(bpb_no_leading_space)).clone().detach(),
                "bpb_v2": (sum(bpb) / len(bpb)).clone().detach(),
            }
        else:
            return {
                f"{self.metric_type}_v1": torch.tensor(sum(correct) / len(correct)),
                f"{self.metric_type}_v2": torch.tensor(sum(correct) / len(correct)),
                "ce_loss_v1": torch.tensor(
                    sum(celoss_no_leading_space) / len(celoss_no_leading_space)
                ),
                "ce_loss_v2": torch.tensor(sum(celoss) / len(celoss)),
                "bpb_v1": torch.tensor(sum(bpb_no_leading_space) / len(bpb_no_leading_space)),
                "bpb_v2": torch.tensor(sum(bpb) / len(bpb)),
                "soft_v1": torch.tensor(
                    sum(soft_score_no_leading_space) / len(soft_score_no_leading_space)
                ),
                "soft_v2": torch.tensor(sum(soft_score) / len(soft_score)),
                "soft_log_v1": torch.tensor(
                    sum(soft_log_score_no_leading_space) / len(soft_log_score_no_leading_space)
                ),
                "soft_log_v2": torch.tensor(sum(soft_log_score) / len(soft_log_score)),
            }

    def custom_f1_score(self, y_true, y_pred, pos_label=1):
        y_true = list(y_true)
        y_pred = list(y_pred)
        tp = sum((yt == pos_label) and (yp == pos_label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != pos_label) and (yp == pos_label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == pos_label) and (yp != pos_label) for yt, yp in zip(y_true, y_pred))

        if tp + fp == 0 or tp + fn == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
