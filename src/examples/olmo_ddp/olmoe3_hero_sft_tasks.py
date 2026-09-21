"""Four bounded chat benchmarks using OLMo Think prompts and the frozen eval scorers."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer, Scorer
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.evals.tasks.humaneval import HumanEval
from olmo_eval.evals.tasks.ifeval_ood import IFEvalOOD
from olmo_eval.evals.tasks.minerva_math import Math500

SAMPLING = SamplingParams(
    max_tokens=32768,
    temperature=float(os.environ.get("HERO_SFT_TEMPERATURE", "0.6")),
    top_p=0.95,
    top_k=-1,
    num_samples=1,
    do_sample=True,
    stop_sequences=("<|im_end|>", "<|endoftext|>"),
)


def split_reasoning(raw):
    """Separate Think final answers; unfinished reasoning is not a final response."""
    if "</think>" not in raw:
        return raw.removeprefix("<think>"), "", False
    reasoning, answer = raw.rsplit("</think>", 1)
    reasoning = reasoning.removeprefix("<think>")
    answer = re.sub(r"^\s*<answer>\s*", "", answer)
    answer = re.sub(r"</answer>\s*$", "", answer).strip()
    return reasoning, answer, True


class ThinkAnswers:
    """Retain raw responses and score only final answers, including IFBench verifiers."""

    def _extract_answers(self, responses):
        audit = os.environ.get("HERO_SFT_RESPONSE_AUDIT")
        for response in responses:
            for index, output in enumerate(response.outputs):
                if "hero_sft_raw" not in output.metadata:
                    raw = output.text
                    reasoning, final, closed = split_reasoning(raw)
                    output.metadata.update(hero_sft_raw=raw, reasoning_closed=closed)
                    output.text = final
                    if audit:
                        record = {
                            "native_id": response.instance.metadata["id"],
                            "question": response.instance.question,
                            "raw_response": raw,
                            "reasoning": reasoning,
                            "final_response": final,
                            "reasoning_closed": closed,
                            "output_metadata": {
                                k: v for k, v in output.metadata.items() if k != "hero_sft_raw"
                            },
                            "reference": response.instance.metadata.get("reference"),
                        }
                        key = hashlib.sha256(str(record["native_id"]).encode()).hexdigest()
                        path = Path(audit) / self.config.name / f"{key}-{index}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if path.exists():
                            assert os.environ.get("HERO_SFT_RESUME") == "1"
                            assert (
                                json.loads(path.read_text()) == record
                            ), "Conflicting saved response"
                        else:
                            from olmoe3_lr_sweep_watch import atomic_json

                            atomic_json(path, record)
        super()._extract_answers(responses)


@register("hero_sft_math500")
class HeroMath500(ThinkAnswers, Math500):
    """MATH-500, zero-shot boxed-answer Think prompt, native math equivalence scorer."""

    num_fewshot = 0
    data_source = DataSource(
        path="HuggingFaceH4/MATH-500", revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )
    sampling_params = SAMPLING
    formatter = ChatFormatter(
        user_template="{question}\n\nPresent the answer in LaTex format: \\boxed{{Your answer}}"
    )


@register("hero_sft_ifbench")
class HeroIFBench(ThinkAnswers, IFEvalOOD):
    """The 300-prompt IFBench_test2 benchmark, not the broader multi-turn suite."""

    sampling_params = SAMPLING
    data_source = DataSource(
        path="allenai/IFBench_test2",
        split="train",
        revision="05477ce4dfe0627904ac78378eddb466e0f52900",
    )


@register("hero_sft_humaneval")
class HeroHumanEval(ThinkAnswers, HumanEval):
    """HumanEval zero-shot whole-function chat generation with isolated code execution."""

    data_source = DataSource(
        path="openai/openai_humaneval", revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
    )
    num_fewshot = 0
    sampling_params = SAMPLING
    formatter = ChatFormatter(
        user_template="Complete the following function:\n{question}\n"
        "Provide CONCISE reasoning on how to arrive at the answer, and make sure to finish "
        "the response with the following, where (CODE) is the code for the complete function:\n\n"
        "Here is the completed function:\n\n```python\n(CODE)\n```"
    )
    metrics = (PassAtKMetric(k=1, scorer=CodeExecutionScorer),)
    primary_metric = PassAtKMetric(k=1, scorer=CodeExecutionScorer)

    def _extract_answers(self, responses):
        # ThinkAnswers preprocesses final text; HumanEval's body-only helper must not
        # prepend the function signature to a complete function from this prompt.
        ThinkAnswers._extract_answers(self, responses)
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = extract_code(output.text) if output.text else None


@dataclass(frozen=True)
class GenerationRecorded(Scorer):
    """Bookkeeping only; AlpacaEval quality is scored by the official judge afterward."""

    name: str = "generation_recorded"

    def score(self, instance, output):
        return 1.0


@register("hero_sft_alpaca")
class HeroAlpaca(ThinkAnswers, Task):
    """Generate the 805 official prompts before the standard GPT-4.1 LC-win-rate judge."""

    # Explicit JSON avoids the retired dataset-script loader without changing the data.
    data_source = DataSource(
        path="json",
        split="train",
        data_files="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/2edc6fad8be6b14ea7230aabfd08188da6b8b814/alpaca_eval_gpt4_baseline.json",
    )
    split = Split.TRAIN
    sampling_params = SAMPLING
    metrics = (AccuracyMetric(name="generation_recorded", scorer=GenerationRecorded),)
    primary_metric = AccuracyMetric(name="generation_recorded", scorer=GenerationRecorded)

    @property
    def instances(self):
        yield from self._load_instances_cached()

    @property
    def request_type(self):
        return RequestType.CHAT

    def process_doc(self, doc, index=0):
        return Instance(
            question=doc["instruction"],
            gold_answer=None,
            metadata={"id": index, "reference": dict(doc)},
        )

    def format_request(self, instance):
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output):
        return output.text


TASKS = {
    "math500": ("hero_sft_math500", 500),
    "ifbench": ("hero_sft_ifbench", 300),
    "humaneval": ("hero_sft_humaneval", 164),
    "alpaca": ("hero_sft_alpaca", 805),
}
