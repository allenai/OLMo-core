"""SFT prompt formatting for Molmo2 stage-2 (uber_model_v2 + demo_or_style_v2)."""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grounding import (
    MULTI_IMAGE_POINT_COUNT_PROMPTS,
    MULTI_IMAGE_POINTING_PROMPTS,
    POINT_COUNT_PROMPTS,
    POINTING_PROMPTS,
    multi_image_pointing_answer,
    normalize_points,
    pointing_answer,
)
from .multiple_choice_templates import template_mc_question

__all__ = ["SftFormatter", "DEMO_STYLES", "IMAGE_MC_STYLES", "MULTI_IMAGE_POINTING_STYLES"]

# Subset of mm_olmo data_formatter.py DEMO_STYLES / IMAGE_MC_STYLES used by image-only-v9.
DEMO_STYLES = frozenset(
    {
        "point_count",
        "pointing",
        "cosyn_point",
        "user_qa",
        "long_caption",
        "short_caption",
        "text_sft",
        "synthetic_qa",
        "transcript",
        "correction_qa",
        # mm_olmo DEMO_STYLES also carries the multi-image pointing family
        # (data_formatter.py:958-961) — no "style:" prefix under demo_or_style_v2.
        "multi_image_pointing",
        "multi_image_counting",
        "multi_image_point_then_count",
        "multi_image_count_then_point",
    }
)
IMAGE_MC_STYLES = frozenset(
    {
        "a_okvqa_mc",
        "ai2_diagram",
        "ai2_diagram_no_letter",
        "science_qa",
        "eval_multiple_choice",
        "multi_image_mc",
        "mantis_instruct_mc",
    }
)
MULTI_IMAGE_POINTING_STYLES = frozenset(
    {
        "multi_image_pointing",
        "multi_image_point_then_count",
        "multi_image_counting",
        "multi_image_count_then_point",
    }
)

CAPTION_PROMPTS = (
    "Describe this image.",
    "Describe this image",
    "describe the image",
    "Write a long description of this image.",
    "caption the picture",
    "Caption",
    "caption",
    "Construct a long caption for this image",
    "Generate a caption",
    "Create a detailed caption",
    "Write a long caption",
    "Describe this image in detail",
    "Describe this",
    "describe this",
    "Caption this",
    "What can be seen in this image?",
    "What do you see in the image?",
    "Look at this photo carefully and then tell me about it in detail",
    "Write a long description of this image",
    "Tell me about this picture.",
    "Write a paragraph about this image.",
    "Look at this image carefully and then describe it in detail",
    "Generate a long caption about this image.",
)

CHAIN_OF_THOUGHT_PROMPTS = (
    "{question} Provide reasoning steps and then give the short answer.",
)


def _apply_chain_of_thought_prompt(question: str) -> str:
    return CHAIN_OF_THOUGHT_PROMPTS[0].format(question=question)


@dataclass
class SftFormatter:
    """Minimal port of mm_olmo DataFormatter for image-only-v9 SFT."""

    select_answer: str = "best"
    seed: int = 0
    p_multi_point_all_image: float = 0.5
    """Probability that a multi-image pointing question targets "all images" instead of
    a random image subset (mm_olmo stage-2 sets 0.5)."""

    def style_prefix(self, style: str) -> str:
        if style in DEMO_STYLES or style in IMAGE_MC_STYLES:
            return ""
        return f"{style}:"

    def select_vqa_answer(self, answers: Sequence[str], rng: np.random.RandomState) -> Optional[str]:
        if answers is None:
            return None
        if isinstance(answers, str):
            return answers if answers.strip() else None
        answers = [a for a in answers if a and str(a).strip()]
        if not answers:
            return None
        if self.select_answer == "first":
            return min(answers)
        counts = Counter(answers)
        m = max(counts.values())
        cands = [k for k, v in counts.items() if v == m]
        return cands[rng.randint(0, len(cands))]

    def template_options(
        self, example: Dict[str, Any], is_training: bool, rng: np.random.RandomState
    ) -> Tuple[str, str, Dict[str, Any]]:
        labelled = "options" in example
        allow_unlabelled = True
        if labelled and "answer_idx" in example:
            allow_unlabelled = bool(str(example["options"][example["answer_idx"]]).strip())

        if not is_training or rng.random() < 0.1:
            if labelled:
                prefixes = string.ascii_uppercase
                option_text = "\n".join(
                    f"{p}. {o}" for p, o in zip(prefixes, example["options"])
                )
                option_names = list(prefixes[: len(example["options"])])
                outputs = [f"{n}. {o}" for n, o in zip(option_names, example["options"])]
            else:
                option_text = "\n".join(example["unlabelled_options"])
                option_names = list(example["unlabelled_options"])
                outputs = list(example["unlabelled_options"])
            question = (
                example["question"]
                + "\nOnly return the correct answer option.\n"
                + option_text
            )
        else:
            question = example["question"]
            opts = example["options"] if labelled else example["unlabelled_options"]
            question, option_names, outputs = template_mc_question(
                question,
                opts,
                rng,
                unlabelled=not labelled,
                p_label_options=0.8 if allow_unlabelled else 1.0,
            )

        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            if not (0 <= ans_idx < len(outputs)):
                raise ValueError(f"Invalid answer idx in example: {example}")
            output = outputs[ans_idx]
        else:
            output = None
        return question, output, {"option_names": option_names}

    def format_points(self, example: Dict[str, Any]) -> str:
        import numpy as np

        pts = example.get("points")
        if pts is None:
            xy = []
        elif isinstance(pts, dict) and "x" in pts and "y" in pts:
            xy = np.asarray([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        elif isinstance(pts, np.ndarray):
            xy = pts
        elif pts and isinstance(pts[0], dict):
            xy = [[p["x"], p["y"]] for p in pts]
        else:
            xy = pts
        label = example.get("label", example.get("label_cased", "")).lower()
        point_scale = example["point_scale"] if "point_scale" in example else 100
        norm = normalize_points(
            xy, point_scale=point_scale, image_size=example.get("image_size")
        )
        style = example.get("style", "pointing")
        count = example.get("count", len(norm))
        if style == "point_count":
            return pointing_answer(norm, label, "point_count", count=count)
        return pointing_answer(norm, label, "pointing")

    def _format_multi_image_points(
        self, example: Dict[str, Any], rng: np.random.RandomState, is_training: bool
    ) -> Tuple[str, Optional[str]]:
        """Multi-image pointing / counting (port of mm_olmo ``format_multi_points``).

        The example carries parallel per-image lists: ``normalized_labels``, ``labels``,
        and ``points`` (each entry a list of ``{'x','y'}`` dicts or ``(x, y)`` pairs,
        at ``point_scale`` units). One label is drawn, then either "all images" or a
        random image subset is targeted (``p_multi_point_all_image``); the answer
        serializes the matching points of the targeted images with 1-based image
        indices and point ids continuing across images.

        Deviation from mm_olmo (documented): unique labels keep first-occurrence order
        (``dict.fromkeys``) instead of ``list(set(...))``, whose iteration order depends
        on ``PYTHONHASHSEED``; the rng draw count is unchanged.
        """
        style = example["style"]
        norm_labels = list(example["normalized_labels"])
        labels = list(example.get("labels", norm_labels))
        points = list(example["points"])

        unique_labels = list(dict.fromkeys(norm_labels))
        selected_norm = rng.choice(unique_labels)
        selected_label = next(
            (labels[i] for i, nl in enumerate(norm_labels) if nl == selected_norm),
            selected_norm,
        )

        all_images = list(range(len(norm_labels)))
        n_images = len(all_images)
        label_exists = any(
            nl == selected_norm and len(points[i]) > 0 for i, nl in enumerate(norm_labels)
        )

        if self.p_multi_point_all_image:
            if n_images == 1 or rng.random() < self.p_multi_point_all_image:
                selected_images = "all images"
            else:
                n_select = rng.randint(1, n_images)
                chosen = rng.choice(all_images, n_select, replace=False)
                selected_images = ", ".join(f"image_{i + 1}" for i in chosen)
        else:
            n_select = rng.randint(1, n_images) if n_images >= 1 else n_images
            chosen = rng.choice(all_images, size=n_select, replace=False)
            if n_select == n_images and rng.random() < 0.5:
                selected_images = "all images"
            else:
                selected_images = ", ".join(f"image_{i + 1}" for i in chosen)

        pool = (
            MULTI_IMAGE_POINTING_PROMPTS
            if style == "multi_image_pointing"
            else MULTI_IMAGE_POINT_COUNT_PROMPTS
        )
        if selected_images == "all images" and style == "multi_image_pointing":
            # 50% chance to phrase an all-images question with the single-image pool.
            if rng.random() < 0.5:
                question = POINTING_PROMPTS[rng.randint(len(POINTING_PROMPTS))].format(
                    label=selected_label
                )
            else:
                question = pool[rng.randint(len(pool))].format(
                    selected_images=selected_images, selected_label=selected_label
                )
        else:
            question = pool[rng.randint(len(pool))].format(
                selected_images=selected_images, selected_label=selected_label
            )

        if not is_training:
            return question, None
        if not label_exists:
            return question, "There are none."

        if selected_images == "all images":
            valid = [
                i
                for i in all_images
                if norm_labels[i] == selected_norm and len(points[i]) > 0
            ]
        else:
            chosen_idx = [int(s.split("_")[1]) - 1 for s in selected_images.split(", ")]
            valid = [
                i
                for i in chosen_idx
                if i < len(norm_labels)
                and norm_labels[i] == selected_norm
                and len(points[i]) > 0
            ]
        if not valid:
            return question, "There are none."

        scale = float(example.get("point_scale", 100))
        by_image = []
        for i in valid:
            xy = [
                (p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1])
                for p in points[i]
            ]
            # clip_points clamping is equivalent to the [0, 1] clamp inside
            # _scale_point after normalization.
            by_image.append((i + 1, [(x / scale, y / scale) for x, y in xy]))
        return question, multi_image_pointing_answer(by_image, selected_label, style)

    def _needs_formatted_message(self, msg: Dict[str, Any]) -> bool:
        style_str = str(msg.get("style", ""))
        return (
            msg.get("explanation") is not None
            or style_str.endswith("_exp")
            or style_str.startswith("cosyn_")
        )

    def format_turns(
        self,
        example: Dict[str, Any],
        *,
        is_training: bool = True,
        index: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[Tuple[str, str]]:
        """Return list of (user_text, assistant_text) turns for one image."""
        if rng is None:
            rng = np.random.RandomState(self.seed + index)
        style = example.get("style", "")

        if "message_list" in example:
            turns: List[Tuple[str, str]] = []
            for msg in example["message_list"]:
                if "messages" in msg:
                    messages = msg["messages"]
                    for u in range(0, len(messages) - 1, 2):
                        turns.append((messages[u], messages[u + 1]))
                elif "question" in msg and "answer" in msg and self._needs_formatted_message(msg):
                    sub = {k: v for k, v in example.items() if k != "message_list"}
                    sub.update(msg)
                    turns.extend(
                        self.format_turns(sub, is_training=is_training, index=index, rng=rng)
                    )
                elif "question" in msg and "answer" in msg:
                    sub = {k: v for k, v in example.items() if k != "message_list"}
                    sub.update(msg)
                    turns.extend(
                        self.format_turns(sub, is_training=is_training, index=index, rng=rng)
                    )
                elif "question" in msg and "answers" in msg:
                    sub = {k: v for k, v in example.items() if k != "message_list"}
                    sub.update(msg)
                    turns.extend(
                        self.format_turns(sub, is_training=is_training, index=index, rng=rng)
                    )
                else:
                    sub = {k: v for k, v in example.items() if k != "message_list"}
                    sub.update(msg)
                    turns.extend(
                        self.format_turns(sub, is_training=is_training, index=index, rng=rng)
                    )
            valid = [(q, a) for q, a in turns if a and str(a).strip()]
            if is_training and not valid:
                raise ValueError("No valid (question, answer) branches")
            return valid

        output: Optional[str] = None
        if style in MULTI_IMAGE_POINTING_STYLES:
            prompt, output = self._format_multi_image_points(example, rng, is_training)
            if is_training and (output is None or not str(output).strip()):
                raise ValueError("No valid output in example")
            return [(prompt, output)]
        elif style in ("long_caption", "short_caption") and "question" not in example:
            prompt = rng.choice(CAPTION_PROMPTS)
            output = example.get("text") or example.get("caption", "")
        elif style in ("pointing", "point_count", "cosyn_point"):
            label = example.get("label", example.get("label_cased", "")).lower()
            if "question" in example:
                prompt = example["question"]
            else:
                pool = POINT_COUNT_PROMPTS if style == "point_count" else POINTING_PROMPTS
                prompt = pool[rng.randint(len(pool))].format(label=label)
            if not is_training:
                output = None
            else:
                output = self.format_points(example)
        elif "question" in example and (
            "options" in example or "unlabelled_options" in example
        ):
            prompt, output, _ = self.template_options(example, is_training, rng)
        elif "question" in example:
            prompt = example["question"]
        elif "prompt" in example:
            prompt = example["prompt"]
        else:
            prompt = ""

        if "_exp" in style and prompt:
            prompt = _apply_chain_of_thought_prompt(prompt)

        if output is None and is_training:
            if "answers" in example:
                output = self.select_vqa_answer(example["answers"], rng)
            elif "answer" in example and "explanation" in example:
                output = f"{example['explanation']} Answer: {example['answer']}"
            elif "answer" in example:
                output = example["answer"]
            elif "text" in example:
                output = example["text"]

        if is_training and (output is None or not str(output).strip()):
            raise ValueError("No valid output in example")

        prefix = self.style_prefix(style)
        user_text = (f"{prefix} " if prefix and prompt else prefix) + prompt
        return [(user_text, output)]
