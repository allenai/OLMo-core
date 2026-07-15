import abc
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

import datasets
import torch

from .tokenizer import Tokenizer
from .util import load_hf_dataset, load_oe_eval_requests

log = logging.getLogger(__name__)

# Map from oe-eval metrics to metrics used here
METRIC_FROM_OE_EVAL = {
    "acc_raw": "acc",
    "acc_per_char": "len_norm",
    "acc_uncond": "pmi_dc",
    "logits_per_byte": "bpb",
}


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        fixed_ctx_len: bool = False,
        fast_mc: bool = False,
        split="validation",
        metric_type=None,  # Override default metric type
        prompts: Optional[List[Optional[str]]] = None,  # List of prompt variants to use
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.fixed_ctx_len = fixed_ctx_len
        self.fast_mc = fast_mc
        self.prompts = prompts or [None]
        self.current_prompt: Optional[str] = None
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
        self._max_sequence_length: Optional[int] = None

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        new_samples = []
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

                # Add BOS token if it is exists in the tokenizer
                if (
                    self.tokenizer.bos_token_id is not None
                    and ctx[0] != self.tokenizer.bos_token_id
                ):
                    ctx = [self.tokenizer.bos_token_id] + ctx

                if doc_id == 0:
                    log.info(f"First tokens of in-loop eval context: {ctx[:5]}")

                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    )

                for cont_id, continuation_str in enumerate(continuations):
                    # The original implementation did not count the first character (usually the leading space) as
                    # part of the continuation length (e.g., " A", " " is not counted). The OLMES standard does not
                    # do this, but we track both for backwards compatibility.
                    cont_str_len_no_leading_space = len(continuation_str) - 1
                    cont_byte_len_no_leading_space = len(continuation_str[1:].encode("utf-8"))

                    cont_str_len = len(continuation_str)
                    cont_byte_len = len(continuation_str.encode("utf-8"))

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
                    new_samples.append(
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
                            "cont_str_len_no_leading_space": cont_str_len_no_leading_space,
                            "cont_byte_len_no_leading_space": cont_byte_len_no_leading_space,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                        }
                    )

                doc_id += 1

        # Fast MCQA:
        # Only pass a single request, and group together all continuations as tokens
        if self.fast_mc:
            # Get unique doc IDs
            unique_doc_ids = {
                sample["doc_id"] for sample in new_samples if isinstance(sample["doc_id"], int)
            }

            # Create new samples list for fast MC
            fast_mc_samples = []

            # Process each unique document
            for doc_id in unique_doc_ids:
                # Get all samples for this doc_id
                doc_samples = [s for s in new_samples if s["doc_id"] == doc_id]

                # Sort by continuation ID
                doc_samples.sort(
                    key=lambda x: (
                        float(x["cont_id"]) if isinstance(x["cont_id"], (int, float)) else 0.0
                    )
                )

                # Create new sample with distractor continuations
                base_sample = doc_samples[0].copy()
                choices = [s["continuation"] for s in doc_samples]

                # Assert all continuations are length 1
                for choice in choices:
                    if not isinstance(choice, (list, tuple)):
                        raise TypeError(
                            f"Expected continuation to be a list or tuple, got {type(choice)}"
                        )
                    assert len(choice) == 1, f"Expected continuation length 1, got {len(choice)}"

                # Take first token of each continuation
                choices = [
                    choice[0] if isinstance(choice, (list, tuple)) else choice for choice in choices
                ]

                base_sample["choices"] = choices
                base_sample["fast_mc"] = True

                fast_mc_samples.append(base_sample)

            # Add fast MC samples to main samples list
            new_samples = fast_mc_samples

        self.samples = new_samples

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

    @property
    def max_sequence_length(self) -> int:
        if self._max_sequence_length is None:
            max_seq_len = 0
            for sample in self.samples:
                if len(sample["query"]) > max_seq_len:
                    max_seq_len = len(sample["query"])
            # Pad to multiple of 128 for efficiency.
            # TODO (epwalsh): make that configurable
            max_seq_len = 128 * math.ceil(max_seq_len / 128)
            self._max_sequence_length = max_seq_len

        assert (
            self._max_sequence_length != 0
        ), f'Max sequence length for "{self.dataset_name}" cannot be 0. Found {self.samples} samples.'

        return self._max_sequence_length

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

        # Pad to multiple of 128 for efficiency.
        # TODO (epwalsh): make that configurable
        max_query_len = 128 * math.ceil(max_query_len / 128)
        assert max_query_len <= self.max_sequence_length

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        choice_ids = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        cont_str_len_no_leading_space = []
        cont_byte_len_no_leading_space = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len))
            )
            continuations.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len)
                )
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])
            cont_str_len_no_leading_space.append(sample["cont_str_len_no_leading_space"])
            cont_byte_len_no_leading_space.append(sample["cont_byte_len_no_leading_space"])
            if self.fast_mc:
                choice_ids.append(sample["choices"])

            queries.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(
                        sample["query"],
                        max_len=self.model_ctx_len if self.fixed_ctx_len else max_query_len,
                    )
                )
            )
            dc_queries.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len)
                )
            )

            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(
                cont_lens
            ),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "cont_str_len_no_leading_space": torch.LongTensor(cont_str_len_no_leading_space),
            "cont_byte_len_no_leading_space": torch.LongTensor(cont_byte_len_no_leading_space),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
        }

        if self.fast_mc:
            # Pad choice_ids with -1 (for Qs with different numbers of choices)
            max_choices_len = max(len(choices) for choices in choice_ids)
            padded_choice_ids = []
            for choices in choice_ids:
                padding = [-1] * (max_choices_len - len(choices))
                padded_choice_ids.append(choices + padding)
            choice_ids = padded_choice_ids
            batch["choice_ids"] = torch.LongTensor(choice_ids)

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


class PIQA(ICLMultiChoiceTaskDataset):
    """PIQA sends context in the following fashion: "Question: GOAL\nAnswer:"
    space added as prefix to each continuation

    implement PMI_DC

    {
        'goal': "How do I ready a guinea pig cage for it's new occupants?",
        'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.',
        'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.',
        'label': 0
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="piqa",
        dataset_name="plain_text",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + doc["sol1"], " " + doc["sol2"]]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class HellaSwag(ICLMultiChoiceTaskDataset):
    """HellaSwag concats "ACTIVITY_LABEL: CTX_A CTX_B.capitalize()" to form context and then sends endings as continuations
        space added as prefix to each continuation

    {
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'],
        'label': '3'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="hellaswag",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")

        return text

    def doc_to_text(self, doc):
        return self.preprocess(
            doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        )

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + self.preprocess(ending) for ending in doc["endings"]]

    def doc_to_label(self, doc):
        return int(doc["label"])

    def doc_to_domain_conditional(self, doc):
        domain_conditional = self.preprocess(doc["ctx_b"].capitalize())

        # ensure non 0 len domain conditional
        if len(domain_conditional) == 0:
            return self.preprocess(doc["ctx_a"]).split(" ")[-1]

        return domain_conditional


class WinoGrande(ICLMultiChoiceTaskDataset):
    """Prompt: split sentence at _ "SENTENCE[:idx] + OPTION1/OPTION2", where idx = SENTENCE.index("_")
        implement PMI_DC
        acc, random at 50%
        continuation is everything in setnence after '_' (" SENTENCE[idx:].strip()")

        Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')
        Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')

    {
        'sentence': 'People think _ is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.',
        'option1': 'Samantha',
        'option2': 'Rebecca',
        'answer': '2'
    }

    TODO: might need to write custom metric for Winogrande
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="winogrande",
        dataset_name="winogrande_xl",
        **kwargs,
    ):
        # all winogrande datasets have same val set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def prep_examples(self):
        """Overwrite for WinoGrande as multiple ctx, single continuation"""
        doc_id = 0
        for doc in self.dataset:
            # here ctx is a list
            ctxs = self.doc_to_text(doc)
            dcs = self.doc_to_domain_conditional(doc)

            continuation_str = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)

            # The original implementation did not count the first character (usually the leading space) as
            # part of the continuation length (e.g., " A", " " is not counted). The OLMES standard does not
            # do this, but we track both for backwards compatibility.
            cont_str_len_no_leading_space = len(continuation_str) - 1
            cont_byte_len_no_leading_space = len(continuation_str[1:].encode("utf-8"))

            cont_str_len = len(continuation_str)
            cont_byte_len = len(continuation_str.encode("utf-8"))

            # tokenize
            continuation = self.token_encode(continuation_str)

            for cont_id, (ctx, dc) in enumerate(zip(ctxs, dcs)):
                ctx = self.token_encode(ctx)

                # Add BOS token if it is exists in the tokenizer
                if (
                    self.tokenizer.bos_token_id is not None
                    and ctx[0] != self.tokenizer.bos_token_id
                ):
                    ctx = [self.tokenizer.bos_token_id] + ctx

                if doc_id == 0:
                    log.info(f"First tokens of in-loop eval context: {ctx[:5]}")

                dc = self.token_encode(dc)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]

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
                        "ctx_len": len(ctx),
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "cont_str_len_no_leading_space": cont_str_len_no_leading_space,
                        "cont_byte_len_no_leading_space": cont_byte_len_no_leading_space,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

            doc_id += 1

    def doc_to_text(self, doc):
        # special case where there are multiple ctx and single continuation
        pronoun_loc = doc["sentence"].index("_")

        ctx = []
        for option in [doc["option1"], doc["option2"]]:
            ctx.append(doc["sentence"][:pronoun_loc] + option)

        return ctx

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()

    def doc_to_label(self, doc):
        return int(doc["answer"]) - 1

    def doc_to_domain_conditional(self, doc):
        """same number of domain conditionals as context"""
        return [doc["option1"], doc["option2"]]


class OpenBookQA(ICLMultiChoiceTaskDataset):
    """OBQA: question_stem is sent as context (no special prompt format) and choices are sent as continuation
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question_stem': 'Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as',
        'choices': {'text': ['Deep sea animals', 'fish', 'Long Sea Fish', 'Far Sea Animals'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="openbookqa",
        dataset_name="main",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return doc["question_stem"]

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return doc["question_stem"].strip().split(" ")[-1]


class BoolQ(ICLMultiChoiceTaskDataset):
    """Prompt: "PASSAGE\nQuestion: QUESTION?\nAnswer:"
    acc, random at 50% (SuperGLUE)
    continuation: yes, no

    {
        'question': 'is ncis new orleans over for the season',
        'passage': 'NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="boolq",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return doc["passage"] + "\nQuestion: " + doc["question"] + "?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["answer"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SciQ(ICLMultiChoiceTaskDataset):
    """SciQ sends context as "SUPPORT\nQuestion: QUESTION\nAnswer:" and then distractors + correct_answer as continuations
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question': 'Who proposed the theory of evolution by natural selection?',
        'distractor3': 'Scopes',
        'distractor1': 'Linnaeus',
        'distractor2': 'shaw',
        'correct_answer': 'darwin',
        'support': ''
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="sciq",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return doc["support"].strip() + "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["distractor1"],
            " " + doc["distractor2"],
            " " + doc["distractor3"],
            " " + doc["correct_answer"],
        ]

    def doc_to_label(self, doc):
        del doc
        return 3

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcEasy(ICLMultiChoiceTaskDataset):
    """ArcEasy creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
        space added as prefix to each continuation

    {
        'question': 'Which technology was developed most recently?',
        'choices': {'text': ['cellular telephone', 'television', 'refrigerator', 'airplane'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path: str = "ai2_arc",
        dataset_name: Optional[str] = "ARC-Easy",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

        if doc["answerKey"] in num_to_letter:
            doc["answerKey"] = num_to_letter[doc["answerKey"]]

        return ["A", "B", "C", "D", "E"].index(doc["answerKey"])

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcChallenge(ArcEasy):
    """ArcChallenge follows the same prompt format as ArcEasy.
    implement PMI_DC
    """

    metric_type = "len_norm"  # Ideally "pmi_dc"

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Challenge",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )


class ArcEasyCELoss(ArcEasy):
    """ArcEasyCELoss is ARCEasy using an alternate ce_loss metric"""

    metric_type = "ce_loss"

    def doc_to_continuations(self, doc):
        # We only consider the correct answer for this metric
        answer = doc["choices"]["text"][self.doc_to_label(doc)]
        return [" " + answer]

    def doc_to_label(self, doc):
        del doc
        return 0


class BasicArithmetic(ArcEasy):
    """This is a basic arithmetic task follows the same prompt format as ArcEasy.
    Example:
    {"id": "q85_1d1d_max1d_plus",
    "question": "Calculate 2 + 5 =",
    "choices": {"text": ["8", "7", "6", "17"],
    "label": ["A", "B", "C", "D"]},
    "answerKey": "B", "type_tag": "easy"}

    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="allenai/basic_arithmetic",
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )


class CommonsenseQA(ArcEasy):
    """CommonsenseQA
    Example:
    {'id': 'e68fb2448fd74e402aae9982aa76e527',
    'question': 'Where are  you likely to find a hamburger?',
    'question_concept': 'hamburger',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['fast food restaurant', 'pizza', 'ground up dead cows', 'mouth', 'cow carcus']},
    'answerKey': 'A'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="tau/commonsense_qa",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )


class SocialIQa(ICLMultiChoiceTaskDataset):
    """SocialIQa
    Example:
    {'context': 'Jordan was in charge of taking the food on the camping trip and left all the food at home.',
     'question': 'How would Jordan feel afterwards?',
     'answerA': 'horrible that he let his friends down on the camping trip',
     'answerB': "happy that he doesn't need to do the cooking on the trip",
     'answerC': 'very proud and accomplished about the camping trip', 'label': '1'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="social_i_qa",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["context"] + " " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["answerA"],
            " " + doc["answerB"],
            " " + doc["answerC"],
        ]

    def doc_to_label(self, doc):
        return int(doc["label"]) - 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class COPA(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE.strip()[:-1] because/therefore"
    Req_loglikelihood('The pair of students came under scrutiny by the teacher because', ' the students both received excellent grades.'
    continuations: CHOICE1/CHOICE2

    "cause": "because",
    "effect": "therefore",

    implement PMI_DC
    acc, random at 50%

    {
        'premise': 'The pair of students came under scrutiny by the teacher.',
        'choice1': 'The students both received excellent grades.',
        'choice2': 'Their responses on the assignment were identical.',
        'question': 'cause',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="copa",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        connector = "because" if doc["question"] == "cause" else "therefore"

        # remove the period
        return doc["premise"].strip()[:-1] + " " + connector

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        def convert_choice(choice):
            return choice[0].lower() + choice[1:]

        return [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "because" if doc["question"] == "cause" else "therefore"


class RTE(ICLMultiChoiceTaskDataset):
    """Prompt: "SENTENCE1\nQuestion: SENTENCE2 True or False?\nAnswer:"
    implement PMI_DC
    acc, random at 50% (GLUE)
    continuations: True, False

    {
        'sentence1': 'The number of Danes opposed to swapping the krone for the euro has increased slightly to 35.3 percent, up from 34.6 percent in April, according to a poll published on Thursday by Danske Bank.',
        'sentence2': 'The introduction of the euro has been opposed.',
        'label': 0,
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="rte",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return doc["sentence1"] + "\nQuestion: " + doc["sentence2"] + " True or False?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class CommitmentBank(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE\nQuestion: HYPOTHESIS. True, False or Neither?\nAnswer:"
    continuations: True, False, Neither

        implement PMI_DC
        acc/F1, random at 33% acc. (SuperGLUE)

    {
        'premise': 'Then they would awake, terrified and sweating, to find themselves in white starched linen, in a comfortable bed, in peaceful England. And all would be well. It may be said that although he survived it the siege nevertheless had a bad effect on the Collector.',
        'hypothesis': 'the siege nevertheless had a bad effect on the Collector',
        'label': 0
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="cb",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return (
            doc["premise"]
            + "\nQuestion: "
            + doc["hypothesis"]
            + ". True, False or Neither?\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MRPC(ICLMultiChoiceTaskDataset):
    """Prompt for MRPC is formed using "Sentence 1: SENTENCE1\nSentence 2: SENTENCE2\nQuestion: Do both sentences mean the same thing?\nAnswer:"
    acc/F1, random at 50% acc. (GLUE)
    continuations: yes and no

    {
        'sentence1': 'In fiction : Edward P. Jones ( " The Known World " ) and Scott Spencer ( " A Ship Made of Paper " ) .',
        'sentence2': 'The fifth nominee for fiction is Scott Spencer , for A Ship Made of Paper .',
        'label': 0
    }
    """

    metric_type = "f1"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="mrpc",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return (
            "Sentence 1: "
            + self.preprocess(doc["sentence1"])
            + "\nSentence 2: "
            + self.preprocess(doc["sentence2"])
            + "\nQuestion: Do both sentences mean the same thing?\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['label'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SST2(ICLMultiChoiceTaskDataset):
    """SST2 task formats prompts as "SENTENCE\nQuestion: Is this sentence positive or negative?\nAnswer:"
    some preprocessing done on sentence

    constructs 2 requests, 1 for positive and another for negative
    positive and negative have just 1 token in tokenizer
    positive: 1313
    negative: 2430

    implement PMI_DC
    acc, random at 50% (GLUE)

    {
        'sentence': "harrison 's flowers puts its heart in the right place , but its brains are in no particular place at all . ",
        'label': 1,
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="sst2",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return (
            self.preprocess(doc["sentence"])
            + "\nQuestion: Is this sentence positive or negative?\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        # # {1: "positive", 0: "negative"}
        return [" negative", " positive"]

    def doc_to_label(self, doc):
        # {1: "positive", 0: "negative"}
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MMLU(ICLMultiChoiceTaskDataset):
    """MMLU creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
           space added as prefix to each continuation

       {
           'question': "Which of the following terms describes the body's ability to maintain its normal state?",
           'subject': 'anatomy',
           'choices': ['Anabolism', 'Catabolism', 'Tolerance', 'Homeostasis'],
    '       answer': 3
        }
    """

    metric_type = "len_norm"  # Ideally pmi_dc

    _subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }

    _categories = {
        "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other": ["other", "business", "health"],
    }

    def __init__(
        self,
        tokenizer,
        dataset_path="hails/mmlu_no_train",
        dataset_name=None,
        split="validation",
        prompt_variations=None,
        mc_labels=False,
        metric_type=None,
        **kwargs,
    ):
        dataset_names = []
        # Collect the relevant categories
        if dataset_name is not None and dataset_name in MMLU._categories:
            for sub_cat in MMLU._categories[dataset_name]:
                for name, cats in MMLU._subcategories.items():
                    if sub_cat in cats:
                        dataset_names.append(name)
        elif dataset_name in MMLU._subcategories:
            dataset_names.append(dataset_name)
        else:  # E.g., "math"
            for name, cats in MMLU._subcategories.items():
                if dataset_name in cats:
                    dataset_names.append(name)
        self.dev_set = {}
        self.mc_labels = mc_labels
        prompts: List[Union[None, str]] = [None]
        if prompt_variations is not None:
            if prompt_variations == 1:
                prompts = [None, "inst", "inst+1", "inst+2", "inst+3", "inst+4", "inst+5"]
            elif prompt_variations == 2:
                prompts = ["inst+5"]
            else:
                raise ValueError(f"Unknown prompt variations: {prompt_variations}")
            # Need to grab the dev set for the few-shot prompts
            for name in dataset_names:
                dev_set = load_hf_dataset(dataset_path, name, "dev")
                self.dev_set[name] = dev_set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_names,
            split=split,
            prompts=prompts,
            metric_type=metric_type,
            **kwargs,
        )

    def doc_to_text(self, doc):
        def format_example(doc, keys):
            question_prefix = ""
            if not self.mc_labels:
                question_prefix = "Question: "  # To make context more clear
            question = question_prefix + doc["question"].strip()
            choices = ""
            if self.mc_labels:
                choices = "".join(
                    [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
                )
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        output_text = format_example(doc, keys)

        if self.current_prompt is not None:
            prefix = ""
            if "inst" in self.current_prompt:
                subject = doc.get("subject").replace("_", " ")
                prefix = f"The following are multiple choice questions (with answers) about {subject}:\n\n"
            num_shots = re.findall("\\+(\\d+)", self.current_prompt)
            if num_shots:
                dev_set = self.dev_set.get(doc.get("subject"), [])
                num_shots_int = int(num_shots[0])
                for idx, dev_doc in enumerate(dev_set):
                    if idx >= num_shots_int:
                        break
                    if self.mc_labels:
                        answer = keys[dev_doc["answer"]]
                    else:
                        answer = dev_doc["choices"][dev_doc["answer"]]
                    prefix += format_example(dev_doc, keys) + " " + answer + "\n\n"
            output_text = prefix + output_text
        return output_text

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        if self.mc_labels:
            choices = [" A", " B", " C", " D"]
        else:
            choices = [" " + choice for choice in doc["choices"]]
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only need correct answer for these metrics
            return [choices[doc["answer"]]]
        else:
            return choices

    def doc_to_label(self, doc):
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only the correct answer is provided for these metrics
            return 0
        return doc["answer"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class TriviaQACELoss(ICLMultiChoiceTaskDataset):
    """Sample TriviaQA entity with some fields suppressed. For CE Loss we only consider the "value"
    field as the answer to score.

    {
        'question': 'Which Lloyd Webber musical premiered in the US on 10th December 1993?',
        'question_id': 'tc_33',
        'answer': {
            'aliases': ['Sunset Blvd', ...],
            'normalized_aliases': ['sunset boulevard', ...],
            'normalized_value': 'sunset boulevard',
            'value': 'Sunset Boulevard'
        }
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="trivia_qa",
        dataset_name="rc.wikipedia.nocontext",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"]["value"]]

    def doc_to_label(self, doc):
        del doc
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class NaturalQuestionsCELoss(ICLMultiChoiceTaskDataset):
    """Sample NaturalQuestions entity. For CE Loss we only consider the first answer entry to score.

    {
        'question': 'when was the last time anyone was on the moon',
        'answer': ['14 December 1972 UTC', 'December 1972']
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="nq_open",
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"][0]]

    def doc_to_label(self, doc):
        del doc
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class OEEvalTask(ICLMultiChoiceTaskDataset):
    """Generic class for OE evaluation tasks"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        fixed_ctx_len: bool = False,
        fast_mc: bool = False,
        split=None,
        metric_type=None,
        prompts: Optional[List[Optional[str]]] = None,  # List of prompt variants to use
    ):
        assert prompts is None  # not used

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.fixed_ctx_len = fixed_ctx_len
        self.fast_mc = fast_mc
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
                            raise ValueError(
                                f"Conflicting metric types: {self.metric_type} and {metric_type}"
                            )
                        self.metric_type = metric_type
        self.dataset = requests_list

        # prep examples
        self.prep_examples()

        self._max_sequence_length: Optional[int] = None

    def prep_examples(self):
        current_doc_id_offset = 0
        max_doc_id = 0
        for requests in self.dataset:
            current_doc_id_offset += max_doc_id
            max_doc_id = 0  # Max doc id seen in this dataset

            new_samples = []
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
                    or request["request_type"] == "generate_until_and_loglikelihood"
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
                    if label_id is None:
                        label_id = 0
                    if label_id != cont_id and not isinstance(label_id, (str, list)):
                        # Skip non-target continuations for ce_loss and bpb
                        continue
                    else:
                        # Treat as instance with just one continuation
                        cont_id = 0
                        label_id = 0
                doc_text = request_dict["context"]
                ctx = self.token_encode(doc_text)

                # Add BOS token if it is exists in the tokenizer
                if (
                    self.tokenizer.bos_token_id is not None
                    and ctx[0] != self.tokenizer.bos_token_id
                ):
                    ctx = [self.tokenizer.bos_token_id] + ctx

                if doc_id == 0:
                    log.info(f"First tokens of in-loop eval context: {ctx[:5]}")

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

                # The original implementation did not count the first character (usually the leading space) as
                # part of the continuation length (e.g., " A", " " is not counted). The OLMES standard does not
                # do this, but we track both for backwards compatibility.
                cont_str_len_no_leading_space = len(continuation_str) - 1
                cont_byte_len_no_leading_space = len(continuation_str[1:].encode("utf-8"))

                cont_str_len = len(continuation_str)
                cont_byte_len = len(continuation_str.encode("utf-8"))

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
                new_samples.append(
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
                        "cont_str_len_no_leading_space": cont_str_len_no_leading_space,
                        "cont_byte_len_no_leading_space": cont_byte_len_no_leading_space,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

            # Fast MCQA:
            # Only pass a single request, and group together all continuations as tokens
            if self.fast_mc:
                # Get unique doc IDs
                unique_doc_ids = set(sample["doc_id"] for sample in new_samples)

                # Create new samples list for fast MC
                fast_mc_samples = []

                # Process each unique document
                for doc_id in unique_doc_ids:
                    # Get all samples for this doc_id
                    doc_samples = [s for s in new_samples if s["doc_id"] == doc_id]

                    # Sort by continuation ID
                    doc_samples.sort(key=lambda x: x["cont_id"])

                    # Create new sample with distractor continuations
                    base_sample = doc_samples[0].copy()
                    choices = [s["continuation"] for s in doc_samples]

                    # Assert all continuations are length 1
                    for choice in choices:
                        assert (
                            len(choice) == 1
                        ), f"Expected continuation length 1, got {len(choice)}"

                    # Take first token of each continuation
                    choices = [choice[0] for choice in choices]

                    base_sample["choices"] = choices
                    base_sample["fast_mc"] = True

                    fast_mc_samples.append(base_sample)

                # Add fast MC samples to main samples list
                new_samples = fast_mc_samples

            self.samples = new_samples

    def doc_to_text(self, doc) -> str:
        del doc
        raise NotImplementedError

    def doc_to_continuations(self, doc) -> List[str]:
        del doc
        raise NotImplementedError

    def doc_to_label(self, doc) -> int:
        del doc
        raise NotImplementedError


# This is a backwards-compatible task map to previous OLMo in-loop configurations
LABEL_TO_TASK_MAP_ORIG = {
    "piqa": PIQA,
    "hellaswag": HellaSwag,
    "winogrande": WinoGrande,
    "openbook_qa": OpenBookQA,
    "boolq": BoolQ,
    "sciq": SciQ,
    "arc_easy": ArcEasy,
    "arc_easy_ppl": ArcEasyCELoss,
    "arc_challenge": ArcChallenge,
    "basic_arithmetic": BasicArithmetic,
    "copa": COPA,
    "rte": RTE,
    "commitment_bank": CommitmentBank,
    "mrpc": MRPC,
    "sst2": SST2,
    "commonsense_qa": CommonsenseQA,
    "social_iqa": SocialIQa,
    "trivia_qa_wiki_ppl": TriviaQACELoss,
    "natural_qs_open_ppl": NaturalQuestionsCELoss,
    "mmlu_stem_test": (MMLU, {"dataset_name": "stem", "split": "test"}),
    "mmlu_humanities_test": (MMLU, {"dataset_name": "humanities", "split": "test"}),
    "mmlu_social_sciences_test": (MMLU, {"dataset_name": "social_sciences", "split": "test"}),
    "mmlu_other_test": (MMLU, {"dataset_name": "other", "split": "test"}),
    "mmlu_stem": (MMLU, {"dataset_name": "stem"}),
    "mmlu_humanities": (MMLU, {"dataset_name": "humanities"}),
    "mmlu_social_sciences": (MMLU, {"dataset_name": "social_sciences"}),
    "mmlu_other": (MMLU, {"dataset_name": "other"}),
    "mmlu_stem_bpb": (MMLU, {"dataset_name": "stem", "metric_type": "bpb"}),
    "mmlu_humanities_bpb": (MMLU, {"dataset_name": "humanities", "metric_type": "bpb"}),
    "mmlu_social_sciences_bpb": (MMLU, {"dataset_name": "social_sciences", "metric_type": "bpb"}),
    "mmlu_other_bpb": (MMLU, {"dataset_name": "other", "metric_type": "bpb"}),
    "mmlu_stem_var": (MMLU, {"dataset_name": "stem", "prompt_variations": 1}),
    "mmlu_humanities_var": (MMLU, {"dataset_name": "humanities", "prompt_variations": 1}),
    "mmlu_social_sciences_var": (MMLU, {"dataset_name": "social_sciences", "prompt_variations": 1}),
    "mmlu_other_var": (MMLU, {"dataset_name": "other", "prompt_variations": 1}),
    "mmlu_stem_mc_5shot": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_mc_5shot": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_stem_mc_5shot_test": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_mc_5shot_test": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_mc_5shot_test": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
        },
    ),
    "mmlu_other_mc_5shot_test": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    # Paste in all oe-eval tasks from output of scripts/list_evals_from_oe_eval.py
    "arc_challenge_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "basic_skills_arithmetic_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "basic_skills_arithmetic",
            "dataset_name": "rc_5shot",
            "metric_type": "acc",
        },
    ),
    "basic_skills_coding_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "basic_skills_coding", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "basic_skills_common_knowledge_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "basic_skills_common_knowledge",
            "dataset_name": "rc_5shot",
            "metric_type": "acc",
        },
    ),
    "basic_skills_logical_reasoning_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "basic_skills_logical_reasoning",
            "dataset_name": "rc_5shot",
            "metric_type": "acc",
        },
    ),
    "basic_skills_pattern_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "basic_skills_pattern", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "basic_skills_string_operations_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "basic_skills_string_operations",
            "dataset_name": "rc_5shot",
            "metric_type": "acc",
        },
    ),
    "boolq_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "boolq_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "boolq_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "copycolors_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "acc"},
    ),
    "copycolors_xl_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "acc"},
    ),
    "copycolors_10way_fast": (
        OEEvalTask,
        {
            "dataset_path": "copycolors",
            "dataset_name": "10way",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "copycolors_xl_10way_fast": (
        OEEvalTask,
        {
            "dataset_path": "copycolors",
            "dataset_name": "xl_10way",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "csqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "csqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "csqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "hellaswag_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "openbookqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "piqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "piqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "sciq_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "socialiqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "socialiqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "winogrande_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "winogrande_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    # (DEPRICATED) BPB-only versions of the above tasks. By default, in-loop evals will calculate
    # the BPB and accuracy metrics, so there is no need to use these keys. We keep them for
    # backwards compatibility.
    "mmlu_stem_var_bpb": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_humanities_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "mmlu_other_var_bpb": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 1, "metric_type": "bpb"},
    ),
    "arc_challenge_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "boolq_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "copycolors_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "bpb"},
    ),
    "copycolors_xl_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "bpb"},
    ),
    "csqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "csqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "csqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "piqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "piqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "sciq_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "socialiqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
}

# This standardizes the metrics we should eval for the ladder.
# Train and test sets are added when applicable.
# No subsampling happens in these sets.
LABEL_TO_TASK_MAP_LADDER = {
    "arc_challenge_val_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "arc_challenge",
            "dataset_name": "val_rc_5shot",
            "metric_type": "len_norm",
        },
    ),
    "arc_challenge_val_bpb_5shot": (
        OEEvalTask,
        {
            "dataset_path": "arc_challenge",
            "dataset_name": "val_rc_5shot",
            "metric_type": "bpb",
        },
    ),
    "arc_challenge_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_test_rc_5shot": (
        OEEvalTask,
        {
            "dataset_path": "arc_challenge",
            "dataset_name": "test_rc_5shot",
            "metric_type": "len_norm",
        },
    ),
    "arc_challenge_test_bpb_5shot": (
        OEEvalTask,
        {
            "dataset_path": "arc_challenge",
            "dataset_name": "test_rc_5shot",
            "metric_type": "bpb",
        },
    ),
    "arc_challenge_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_test_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "arc_challenge",
            "dataset_name": "test_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "arc_easy_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "arc_easy",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "arc_easy_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_test_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_test_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "arc_easy",
            "dataset_name": "test_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "boolq_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "boolq",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "csqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "csqa_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "csqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "csqa_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "csqa",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "hellaswag_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "hellaswag",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "openbookqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "openbookqa",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "openbookqa_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_test_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_test_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "openbookqa",
            "dataset_name": "test_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "piqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "piqa_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "piqa",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "socialiqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "socialiqa",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "winogrande_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "winogrande_val_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_val_mc_5shot_fast": (
        OEEvalTask,
        {
            "dataset_path": "winogrande",
            "dataset_name": "val_mc_5shot",
            "metric_type": "acc",
            "fast_mc": True,
        },
    ),
    "mmlu_stem_val_rc_var": (MMLU, {"dataset_name": "stem", "prompt_variations": 1}),
    "mmlu_stem_val_rc_5shot": (MMLU, {"dataset_name": "stem", "prompt_variations": 2}),
    "mmlu_stem_val_bpb_5shot": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_stem_val_mc_5shot": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_stem_val_mc_5shot_fast": (
        MMLU,
        {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True, "fast_mc": True},
    ),
    "mmlu_stem_test_rc_var": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 1},
    ),
    "mmlu_stem_test_bpb_var": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_rc_5shot": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_stem_test_bpb_5shot": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_mc_5shot": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_stem_test_mc_5shot_fast": (
        MMLU,
        {
            "dataset_name": "stem",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "fast_mc": True,
        },
    ),
    "mmlu_humanities_val_rc_var": (MMLU, {"dataset_name": "humanities", "prompt_variations": 1}),
    "mmlu_humanities_val_rc_5shot": (MMLU, {"dataset_name": "humanities", "prompt_variations": 2}),
    "mmlu_humanities_val_bpb_var": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_bpb_5shot": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_val_mc_5shot_fast": (
        MMLU,
        {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True, "fast_mc": True},
    ),
    "mmlu_humanities_test_rc_var": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 1},
    ),
    "mmlu_humanities_test_rc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_humanities_test_bpb_var": (
        MMLU,
        {
            "dataset_name": "humanities",
            "split": "test",
            "prompt_variations": 2,
            "metric_type": "bpb",
        },
    ),
    "mmlu_humanities_test_bpb_5shot": (
        MMLU,
        {
            "dataset_name": "humanities",
            "split": "test",
            "prompt_variations": 2,
            "metric_type": "bpb",
        },
    ),
    "mmlu_humanities_test_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_test_mc_5shot_fast": (
        MMLU,
        {
            "dataset_name": "humanities",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "fast_mc": True,
        },
    ),
    "mmlu_social_sciences_val_rc_var": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 1},
    ),
    "mmlu_social_sciences_val_rc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2},
    ),
    "mmlu_social_sciences_val_bpb_var": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_bpb_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_val_mc_5shot_fast": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "prompt_variations": 2,
            "mc_labels": True,
            "fast_mc": True,
        },
    ),
    "mmlu_social_sciences_test_rc_var": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 1},
    ),
    "mmlu_social_sciences_test_rc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_social_sciences_test_bpb_var": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "metric_type": "bpb",
        },
    ),
    "mmlu_social_sciences_test_bpb_5shot": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "metric_type": "bpb",
        },
    ),
    "mmlu_social_sciences_test_mc_5shot": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
        },
    ),
    "mmlu_social_sciences_test_mc_5shot_fast": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "fast_mc": True,
        },
    ),
    "mmlu_other_val_rc_var": (MMLU, {"dataset_name": "other", "prompt_variations": 1}),
    "mmlu_other_val_rc_5shot": (MMLU, {"dataset_name": "other", "prompt_variations": 2}),
    "mmlu_other_val_bpb_var": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_val_bpb_5shot": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_val_mc_5shot": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_val_mc_5shot_fast": (
        MMLU,
        {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True, "fast_mc": True},
    ),
    "mmlu_other_test_rc_var": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 1},
    ),
    "mmlu_other_test_rc_5shot": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2},
    ),
    "mmlu_other_test_bpb_var": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_test_bpb_5shot": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_test_mc_5shot": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_test_mc_5shot_fast": (
        MMLU,
        {
            "dataset_name": "other",
            "split": "test",
            "prompt_variations": 2,
            "mc_labels": True,
            "fast_mc": True,
        },
    ),
}

# Expanded tasks for BPB on some generative tasks
LABEL_TO_TASK_MAP_EXPANDED = {
    "gsm8k_gold_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "gsm8k", "dataset_name": "gold_bpb_5shot", "metric_type": "bpb"},
    ),
    "codex_humaneval_gold_bpb_0shot": (
        OEEvalTask,
        {"dataset_path": "codex_humaneval", "dataset_name": "gold_bpb_0shot", "metric_type": "bpb"},
    ),
    "codex_humaneval_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "codex_humaneval", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "codex_mbpp_gold_bpb_0shot": (
        OEEvalTask,
        {"dataset_path": "codex_mbpp", "dataset_name": "gold_bpb_0shot", "metric_type": "bpb"},
    ),
    "codex_mbpp_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "codex_mbpp", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "minerva_math_algebra_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_algebra",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_counting_and_probability_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_counting_and_probability",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_geometry_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_geometry",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_intermediate_algebra_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_intermediate_algebra",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_number_theory_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_number_theory",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_prealgebra_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_prealgebra",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_precalculus_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_precalculus",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "minerva_math_500_gold_bpb_0shot": (
        OEEvalTask,
        {
            "dataset_path": "minerva_math_500",
            "dataset_name": "gold_bpb_0shot",
            "metric_type": "bpb",
        },
    ),
    "mt_mbpp_haskell_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_haskell", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_go_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_go", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_python_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_python", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_cpp_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_cpp", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_javascript_gold_bpb_3shot": (
        OEEvalTask,
        {
            "dataset_path": "mt_mbpp_javascript",
            "dataset_name": "gold_bpb_3shot",
            "metric_type": "bpb",
        },
    ),
    "mt_mbpp_swift_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_swift", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_scala_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_scala", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_bash_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_bash", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_typescript_gold_bpb_3shot": (
        OEEvalTask,
        {
            "dataset_path": "mt_mbpp_typescript",
            "dataset_name": "gold_bpb_3shot",
            "metric_type": "bpb",
        },
    ),
    "mt_mbpp_c_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_c", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_php_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_php", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_rust_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_rust", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_csharp_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_csharp", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_r_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_r", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_ruby_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_ruby", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_java_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_java", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    "mt_mbpp_matlab_gold_bpb_3shot": (
        OEEvalTask,
        {"dataset_path": "mt_mbpp_matlab", "dataset_name": "gold_bpb_3shot", "metric_type": "bpb"},
    ),
    # Gen tasks BPB (5 tasks)
    "coqa_bpb_0shot": (
        OEEvalTask,
        {"dataset_path": "coqa", "dataset_name": "bpb_0shot", "metric_type": "bpb"},
    ),
    "drop_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "drop", "dataset_name": "bpb_5shot", "metric_type": "bpb"},
    ),
    "jeopardy_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "jeopardy", "dataset_name": "bpb_5shot", "metric_type": "bpb"},
    ),
    "naturalqs_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "naturalqs_open", "dataset_name": "bpb_5shot", "metric_type": "bpb"},
    ),
    "squad_bpb_5shot": (
        OEEvalTask,
        {"dataset_path": "squad", "dataset_name": "bpb_5shot", "metric_type": "bpb"},
    ),
    # Science/medical RC (6 tasks)
    "lab_bench_dbqa_rc_3shot": (
        OEEvalTask,
        {"dataset_path": "lab_bench_dbqa", "dataset_name": "rc_3shot", "metric_type": "acc"},
    ),
    "lab_bench_protocolqa_rc_3shot": (
        OEEvalTask,
        {"dataset_path": "lab_bench_protocolqa", "dataset_name": "rc_3shot", "metric_type": "acc"},
    ),
    "medmcqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "medmcqa", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "medqa_en_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "medqa_en", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "qasper_yesno_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "qasper_yesno", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "sciriff_yesno_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "sciriff_yesno", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    # Lambada BPB (single gold answer, BPB only)
    "lambada_bpb_0shot": (
        OEEvalTask,
        {"dataset_path": "lambada", "dataset_name": "bpb_0shot", "metric_type": "bpb"},
    ),
}


LABEL_TO_TASK_MAP = {
    **LABEL_TO_TASK_MAP_ORIG,
    **LABEL_TO_TASK_MAP_LADDER,
    **LABEL_TO_TASK_MAP_EXPANDED,
}


def list_tasks() -> List[str]:
    return list(LABEL_TO_TASK_MAP.keys())


def build_task(
    label: str,
    tokenizer: Tokenizer,
    model_ctx_len: int = 2048,
    fixed_ctx_len: bool = False,
) -> ICLMultiChoiceTaskDataset:
    if label not in LABEL_TO_TASK_MAP.keys():
        raise KeyError(
            f'\
            Downstream evaluation config "{label}" not found in available configuraitons. \
            Please specify tasks in the available configurations: {LABEL_TO_TASK_MAP}\
        '
        )
    task_class = LABEL_TO_TASK_MAP[label]
    task_kwargs = {}
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
    return cast(Type[ICLMultiChoiceTaskDataset], task_class)(
        tokenizer=tokenizer, model_ctx_len=model_ctx_len, fixed_ctx_len=fixed_ctx_len, **task_kwargs
    )
