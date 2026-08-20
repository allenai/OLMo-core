"""Pinned task adapters for the SSMax downstream fast suite."""

from __future__ import annotations

import re

from olmo_core.eval.vision_alignment_ssmax_downstream import is_mathvista_geometry_mc


def blink_doc_to_visual(document):
    """Return BLINK images in validated numeric slot order, skipping only trailing nulls."""
    slots = {}
    for key, image in document.items():
        match = re.fullmatch(r"image_(\d+)", key)
        if match is None:
            continue
        index = int(match.group(1))
        if index <= 0 or index in slots:
            raise ValueError(f"BLINK has an invalid or duplicate numeric image slot {key!r}")
        slots[index] = image
    if not slots or sorted(slots) != list(range(1, max(slots) + 1)):
        raise ValueError(f"BLINK image slots must be contiguous from image_1, got {sorted(slots)}")

    images = []
    saw_null = False
    for index in sorted(slots):
        image = slots[index]
        if image is None:
            saw_null = True
        elif saw_null:
            raise ValueError("BLINK has a non-null image after an empty numbered slot")
        else:
            images.append(image.convert("RGB"))
    if not images:
        raise ValueError("BLINK document has no non-null images")
    return images


def blink_doc_to_text(document, lmms_eval_specific_kwargs=None):
    """Reproduce the pinned BLINK zero-shot prompt without importing task globals."""
    kwargs = lmms_eval_specific_kwargs or {}
    letters = ", ".join(chr(ord("A") + index) for index in range(len(document["choices"])))
    return kwargs.get("pre_prompt", "").format(letters) + document["prompt"]


def _blink_answer_letter(response):
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", response.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def blink_process_results(document, results):
    """Score BLINK locally and retain the valid-choice count for diagnostics."""
    target = document["answer"].strip("()")
    response = results[0]
    prediction = _blink_answer_letter(response)
    return {
        "blink_acc": {
            "id": document["idx"],
            "gt_content": target,
            "pred_parsed": prediction,
            "pred": response,
            "sub_task": document["sub_task"],
            "is_correct": prediction == target,
            "num_choices": len(document["choices"]),
        }
    }


def blink_aggregate_results(results):
    """Return exact BLINK accuracy from local boolean outcomes."""
    if not results:
        return 0.0
    return sum(int(result["is_correct"]) for result in results) / len(results)


def mathvista_geometry_mc_process_docs(dataset):
    """Materialize only the reviewed geometry problem-solving multiple-choice slice."""
    return dataset.filter(is_mathvista_geometry_mc)


def mathvista_doc_to_visual(document):
    """Return the one RGB image from a MathVista document."""
    return [document["decoded_image"].convert("RGB")]


def mathvista_doc_to_text(document, lmms_eval_specific_kwargs=None):
    """Build the pinned zero-shot solution prompt without initializing an LLM server."""
    kwargs = lmms_eval_specific_kwargs or {}
    expected = {"shot_type": "solution", "shot": 0, "use_caption": False, "use_ocr": False}
    if any(kwargs.get(name) != value for name, value in expected.items()):
        raise ValueError(f"MathVista prompt settings must remain {expected}, got {kwargs}")
    if document["question_type"] != "multi_choice" or document["answer_type"] != "text":
        raise ValueError("the MathVista fast task supports text multiple choice only")

    question = f"Question: {document['question']}"
    if document.get("unit"):
        question += f" (Unit: {document['unit']})"
    choices = ["Choices:"]
    choices.extend(
        f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(document["choices"])
    )
    hint = (
        "Hint: Please answer the question and provide the correct option letter, "
        "e.g., A, B, C, D, at the end."
    )
    return "\n".join((question, *choices, hint, "Solution:"))


def _safe_equal(left, right):
    return str(left).strip() == str(right).strip()


def mathvista_process_results(document, results):
    """Score a restricted option letter locally without MathVista's LLM extractor."""
    choices = document["choices"]
    raw_response = results[0].strip().upper()
    valid_letters = [chr(ord("A") + index) for index in range(len(choices))]
    if raw_response in valid_letters:
        extraction = raw_response
        prediction = choices[valid_letters.index(raw_response)]
    else:
        extraction = ""
        prediction = None
    answer = document.get("answer")
    result = {
        "question_id": document["pid"],
        "query": document["query"],
        "choices": choices,
        "answer": answer,
        "extraction": extraction,
        "prediction": prediction,
        "true_false": prediction is not None and _safe_equal(prediction, answer),
        "question_type": document["question_type"],
        "answer_type": document["answer_type"],
        "precision": document.get("precision", 0),
        "metadata": document["metadata"],
        "num_choices": len(choices),
        "raw_response": raw_response,
    }
    return {"mathvista_geometry_mc_acc": result, "submission": dict(result)}


def mathvista_geometry_mc_aggregate_results(results, *args, **kwargs):
    """Return exact accuracy on the already locally scored geometry-MC rows."""
    del args, kwargs
    return sum(int(result["true_false"]) for result in results) / len(results)
