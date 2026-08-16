import argparse
import copy
import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_external_academic.py"
    )
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_external_academic_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def academic():
    return _load_module()


def test_standard_task_metrics(academic):
    assert academic._normalize_vqa_answer("The red,blue cats?") == "red blue cats"
    assert academic._normalize_vqa_answer("One cat, don't go!") == "1 cat don't go"
    assert academic._normalize_vqa_answer("dont") == "don't"
    assert academic._normalize_vqa_answer("1,000.") == "1000"
    assert academic._normalize_vqa_answer("red,blue?") == "red blue"
    assert academic._normalize_textvqa_answer("red,blue?") == "redblue"
    assert academic._normalize_textvqa_answer("Girl's") == "girl 's"
    # This inert lowercase contraction matches the published EvalAI implementation's
    # capital-I dictionary keys after its lowercasing pass.
    assert academic._normalize_textvqa_answer("im") == "im"

    three_matches = ["yes"] * 3 + ["no"] * 7
    assert academic._vqa_accuracy("yes", three_matches) == pytest.approx(0.9)
    assert academic._vqa_accuracy("yes", ["yes"] * 4 + ["no"] * 6) == 1.0
    assert academic._anls("sitten", ["kitten"]) == pytest.approx(5 / 6)
    assert academic._anls("x", ["kitten"]) == 0.0
    assert academic._anls("ab", ["ac"]) == 0.0
    assert academic._chartqa_relaxed_accuracy("104", "100") == 1.0
    assert academic._chartqa_relaxed_accuracy("105", "100") == 1.0
    assert academic._chartqa_relaxed_accuracy("106", "100") == 0.0
    assert academic._chartqa_relaxed_accuracy("5%", "0.05") == 1.0
    assert academic._chartqa_relaxed_accuracy("0", "0") == 1.0
    assert academic._chartqa_relaxed_accuracy("0.0", "0") == 0.0
    assert academic._chartqa_relaxed_accuracy("Blue", "blue") == 1.0


def test_multiple_choice_scoring_and_empty_option_rendering(academic):
    ai2d = academic.AcademicExample(
        task="ai2d",
        example_id="4185.png-0",
        source_position="392",
        visual=None,
        image_reference=None,
        question="Which diagram label is correct?",
        options=("D", "B", "A", "C"),
        answer_index=2,
        stratum="standard",
    )
    # The predicted index is the outer prompt label C, not the raw diagram-letter text A.
    assert academic._score_prediction(ai2d, prediction="C", predicted_index=2) == 1.0
    assert academic._score_prediction(ai2d, prediction="A", predicted_index=0) == 0.0

    a_okvqa = academic.AcademicExample(
        task="a_okvqa_mc",
        example_id="duplicate-correct-text",
        source_position="0",
        visual=None,
        image_reference=None,
        question="Where?",
        options=("road", "water", "road", "air"),
        answer_index=0,
    )
    # The official A-OKVQA MC scorer compares chosen text, so a duplicate text is correct.
    assert academic._score_prediction(a_okvqa, prediction="C", predicted_index=2) == 1.0

    prompt = academic._build_mc_prompt("Choose.", ("s", "", "b", "f"))
    assert "A. s\nB. <empty>\nC. b\nD. f" in prompt


def _examples(academic, task="vqav2", count=5):
    return [
        academic.AcademicExample(
            task=task,
            example_id=f"id-{index}",
            source_position=str(index),
            visual=None,
            image_reference=f"image-{index}",
            question=f"Question {index}?",
            answers=("yes",) * 10,
        )
        for index in range(count)
    ]


def test_selection_and_shuffle_are_deterministic(academic):
    examples = _examples(academic)
    selected = academic._select_examples(examples, task="vqav2", seed=17, maximum=3)
    reversed_selected = academic._select_examples(
        list(reversed(examples)), task="vqav2", seed=17, maximum=3
    )
    assert [item.example_id for item in selected] == [item.example_id for item in reversed_selected]

    hashes = {
        item.example_id: hashlib.sha256(item.example_id.encode()).hexdigest() for item in selected
    }
    grids = {item.example_id: (14, 14, 14, 24) for item in selected}
    donors = academic._shuffle_donors(selected, hashes, grids)
    assert donors == academic._shuffle_donors(selected, hashes, grids)
    assert all(hashes[recipient] != hashes[donor] for recipient, donor in donors.items())


@pytest.mark.parametrize("size", [(378, 378), (256, 1024), (1024, 256), (80, 2400)])
def test_grid_signature_matches_native_preprocessor(academic, size):
    image = Image.new("RGB", size, color=(17, 31, 47))
    _, _, actual = academic.preprocess_image_molmo2(
        image,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_crops=academic.DEFAULT_MAX_CROPS,
        is_training=False,
    )
    assert academic._molmo2_grid_signature(image) == tuple(int(value) for value in actual)


def test_grid_selection_excludes_and_backfills_singleton_strata(academic):
    examples = _examples(academic, count=6)
    ranked = sorted(
        examples,
        key=lambda example: (
            hashlib.sha256(f"29\0vqav2\0{example.example_id}".encode()).digest(),
            example.example_id,
        ),
    )
    singleton = ranked[3].example_id
    first_backfill = ranked[4].example_id
    grid_by_id = {
        **{example.example_id: (14, 14, 24, 24) for example in examples},
        singleton: (14, 14, 14, 52),
    }

    def resolve(example):
        return (
            hashlib.sha256(example.example_id.encode()).hexdigest(),
            grid_by_id[example.example_id],
        )

    selected, hashes, grids, audit = academic._grid_compatible_selection(
        examples,
        task="vqav2",
        seed=29,
        maximum=4,
        resolve_image=resolve,
    )
    assert [example.example_id for example in selected] == [
        ranked[0].example_id,
        ranked[1].example_id,
        ranked[2].example_id,
        first_backfill,
    ]
    assert audit["excluded_nonviable_pairing_ids"] == [singleton]
    assert audit["backfilled_ids"] == [first_backfill]
    donors = academic._shuffle_donors(selected, hashes, grids)
    assert all(grids[recipient] == grids[donor] for recipient, donor in donors.items())


def test_ai2d_donors_match_stratum_and_change_base_diagram(academic):
    specifications = (
        ("941.png-3", "standard", "a" * 64),
        ("941.png-4", "standard", "b" * 64),
        ("632.png-0", "standard", "c" * 64),
        ("941.png-3_transparent", "transparent", "d" * 64),
        ("632.png-0_transparent", "transparent", "e" * 64),
    )
    examples = [
        academic.AcademicExample(
            task="ai2d",
            example_id=example_id,
            source_position=str(index),
            visual=None,
            image_reference=None,
            question="Choose.",
            options=("yes", "no"),
            answer_index=0,
            stratum=stratum,
        )
        for index, (example_id, stratum, _) in enumerate(specifications)
    ]
    hashes = {
        example.example_id: image_hash
        for example, (_, _, image_hash) in zip(examples, specifications)
    }
    grids = {example.example_id: (14, 14, 14, 24) for example in examples}
    donors = academic._shuffle_donors(examples, hashes, grids)
    by_id = {example.example_id: example for example in examples}
    for recipient_id, donor_id in donors.items():
        assert by_id[recipient_id].stratum == by_id[donor_id].stratum
        assert academic._ai2d_base_diagram_id(recipient_id) != academic._ai2d_base_diagram_id(
            donor_id
        )
    assert donors["941.png-3_transparent"] != "941.png-3"


def test_inventory_is_fail_closed_sorted_and_unique(academic, tmp_path):
    missing = tmp_path / "missing.sha256"
    with pytest.raises(ValueError, match="missing"):
        academic._load_train_inventory(missing)

    inventory = tmp_path / "train.sha256"
    inventory.write_text(f"{'b' * 64}\n{'a' * 64}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sorted and unique"):
        academic._load_train_inventory(inventory)

    inventory.write_text(f"{'a' * 64}\n{'b' * 64}\n", encoding="utf-8")
    values, identity = academic._load_train_inventory(inventory)
    assert values == {"a" * 64, "b" * 64}
    assert identity["count"] == 2


def test_canonical_writer_refuses_overwrite_and_content_tamper(academic, tmp_path):
    path = tmp_path / "receipt.json"
    payload = academic._attach_content_sha256({"b": 2, "a": 1})
    academic._write_json_no_overwrite(path, payload)
    assert path.read_bytes() == academic._canonical_bytes(payload) + b"\n"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        academic._write_json_no_overwrite(path, payload)

    tampered = dict(payload)
    tampered["a"] = 2
    with pytest.raises(ValueError, match="content SHA-256 differs"):
        academic._verify_content_sha256(tampered, name="test receipt")


def test_repo_file_identity_is_checkout_prefix_independent(academic, tmp_path):
    first_root = tmp_path / "checkout-a"
    second_root = tmp_path / "checkout-b"
    first = first_root / "src" / "module.py"
    second = second_root / "src" / "module.py"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"VALUE = 1\n")
    second.write_bytes(b"VALUE = 1\n")

    assert academic._repo_relative_file_identity(
        first, first_root
    ) == academic._repo_relative_file_identity(second, second_root)
    assert (
        academic._repo_relative_file_identity(first, first_root)["repo_relative_path"]
        == "src/module.py"
    )


def test_manifest_rederives_selected_ids_and_donors(academic, tmp_path, monkeypatch):
    images = []
    examples = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        image = tmp_path / f"image-{index}.png"
        Image.new("RGB", (64, 32), color=color).save(image)
        images.append(image)
        examples.append(
            academic.AcademicExample(
                task="vqav2",
                example_id=f"id-{index}",
                source_position=str(index),
                visual=str(image),
                image_reference=str(image),
                question=f"Question {index}?",
                answers=("yes",) * 10,
            )
        )
    inventory = tmp_path / "train.sha256"
    inventory.write_text(
        f"{hashlib.sha256(images[0].read_bytes()).hexdigest()}\n", encoding="utf-8"
    )
    implementation = {"files": {}, "files_sha256": "f" * 64}
    source = {
        "split": "validation",
        "examples": len(examples),
        "ordered_annotation_projection_sha256": academic._ordered_projection_sha256(examples),
        "files": [],
        "files_sha256": academic._canonical_sha256([]),
    }
    monkeypatch.setattr(academic, "_load_task_examples", lambda task: list(examples))
    monkeypatch.setattr(academic, "_source_identity", lambda task, rows: source)
    monkeypatch.setattr(academic, "_implementation_identity", lambda: implementation)
    monkeypatch.setattr(
        academic,
        "_git_revision",
        lambda: {"revision": "1" * 40, "dirty": False},
    )
    args = argparse.Namespace(
        tasks=["vqav2"],
        examples_per_task=3,
        selection_seed=19,
        train_image_inventory=str(inventory),
    )
    manifest = academic._build_manifest(args)
    path = tmp_path / "manifest.json"
    academic._write_json_no_overwrite(path, manifest)
    loaded_manifest, loaded, _ = academic._validate_manifest_and_load_examples(path)
    assert loaded_manifest == manifest
    assert list(loaded["vqav2"]) == [
        record["example_id"] for record in manifest["tasks"]["vqav2"]["records"]
    ]

    reordered = copy.deepcopy(manifest)
    reordered["tasks"]["vqav2"]["records"][:2] = reversed(
        reordered["tasks"]["vqav2"]["records"][:2]
    )
    reordered["tasks"]["vqav2"]["selection_sha256"] = academic._canonical_sha256(
        reordered["tasks"]["vqav2"]["records"]
    )
    reordered.pop("content_sha256")
    reordered = academic._attach_content_sha256(reordered)
    reordered_path = tmp_path / "manifest-reordered.json"
    academic._write_json_no_overwrite(reordered_path, reordered)
    with pytest.raises(ValueError, match="selected-ID order"):
        academic._validate_manifest_and_load_examples(reordered_path)

    wrong_donor = copy.deepcopy(manifest)
    records = wrong_donor["tasks"]["vqav2"]["records"]
    record_by_id = {record["example_id"]: record for record in records}
    recipient = records[0]
    alternative = next(
        record
        for record in records
        if record["example_id"] not in (recipient["example_id"], recipient["shuffled_donor_id"])
    )
    recipient["shuffled_donor_id"] = alternative["example_id"]
    recipient["shuffled_image_sha256"] = record_by_id[alternative["example_id"]]["image_sha256"]
    recipient["shuffled_image_grid_signature"] = record_by_id[alternative["example_id"]][
        "image_grid_signature"
    ]
    recipient["shuffled_alignment_train_image_overlap"] = record_by_id[alternative["example_id"]][
        "alignment_train_image_overlap"
    ]
    wrong_donor["tasks"]["vqav2"]["selection_sha256"] = academic._canonical_sha256(records)
    wrong_donor.pop("content_sha256")
    wrong_donor = academic._attach_content_sha256(wrong_donor)
    wrong_donor_path = tmp_path / "manifest-wrong-donor.json"
    academic._write_json_no_overwrite(wrong_donor_path, wrong_donor)
    with pytest.raises(ValueError, match="donor"):
        academic._validate_manifest_and_load_examples(wrong_donor_path)


def _task_example(academic, task):
    common = {
        "task": task,
        "example_id": f"{task}-0",
        "source_position": "0",
        "visual": None,
        "image_reference": None,
        "question": "What is the answer?",
    }
    if task in ("ai2d", "a_okvqa_mc"):
        return academic.AcademicExample(
            **common,
            options=("yes", "no"),
            answer_index=0,
            stratum="standard" if task == "ai2d" else None,
        )
    return academic.AcademicExample(
        **common,
        answers=(("100",) if task == "chartqa" else ("yes",) * 10),
        stratum="human" if task == "chartqa" else None,
    )


def _prediction(academic, example):
    if example.options:
        prediction = {
            "prediction": "A",
            "predicted_index": 0,
            "candidate_log_probabilities": {
                "A": float(np.log(0.8)),
                "B": float(np.log(0.2)),
            },
            "input_tokens": 10,
            "output_tokens": 1,
        }
    else:
        answer_token_id = 8 if example.task == "chartqa" else 7
        prediction = {
            "prediction": "100" if example.task == "chartqa" else "yes",
            "predicted_index": None,
            "generated_token_ids": [answer_token_id, academic.EXPECTED_EOS_TOKEN_ID],
            "stop_reason": "eos",
            "input_tokens": 10,
            "output_tokens": 2,
        }
    prediction.update(
        {
            "image_grid_signature": [14, 14, 14, 14],
            "image_token_count": 435,
            "image_token_ids_sha256": "c" * 64,
        }
    )
    prediction["score"] = academic._score_prediction(
        example,
        prediction=prediction["prediction"],
        predicted_index=prediction["predicted_index"],
    )
    return prediction


def _synthetic_tasks_and_manifest(academic):
    tasks = {}
    manifest_tasks = {}
    for task in academic.DEFAULT_TASKS:
        example = _task_example(academic, task)
        record = {
            "example_id": example.example_id,
            "source_position": example.source_position,
            "annotation_sha256": academic._canonical_sha256(example.annotation()),
            "image_sha256": "a" * 64,
            "image_grid_signature": [14, 14, 14, 14],
            "image_token_count": 435,
            "alignment_train_image_overlap": False,
            "shuffled_donor_id": "synthetic-donor",
            "shuffled_image_sha256": "b" * 64,
            "shuffled_image_grid_signature": [14, 14, 14, 14],
            "shuffled_alignment_train_image_overlap": False,
        }
        prediction = _prediction(academic, example)
        row = {
            **{field: record[field] for field in record},
            "question": example.question,
            "gold_answers": list(example.answers),
            "options": list(example.options),
            "gold_answer_index": example.answer_index,
            "stratum": example.stratum,
            "controls": {control: copy.deepcopy(prediction) for control in academic.CONTROLS},
        }
        aggregates = academic._aggregate_task_outputs(task, [row])
        source = {"synthetic": task}
        tasks[task] = {
            "source": source,
            "selection_count": 1,
            "selection_sha256": academic._canonical_sha256([record]),
            "alignment_train_image_overlap_count": 0,
            "generation_stop_counts": academic._generation_stop_counts(task, [row]),
            "elapsed_seconds": 1.0,
            **aggregates,
            "examples": [row],
        }
        manifest_tasks[task] = {
            "source": source,
            "grid_selection": {
                "rule": "synthetic",
                "initial_count": 1,
                "excluded_nonviable_pairing_count": 0,
                "excluded_nonviable_pairing_ids": [],
                "backfilled_count": 0,
                "backfilled_ids": [],
                "final_count": 1,
                "initial_stratum_counts": {"<none>": 1},
                "final_stratum_counts": {"<none>": 1},
            },
            "selection_count": 1,
            "selection_sha256": academic._canonical_sha256([record]),
            "records": [record],
        }
    manifest = {
        "schema_version": academic.SCHEMA_VERSION,
        "format": academic.MANIFEST_FORMAT,
        "protocol_name": academic.PROTOCOL_NAME,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git": {"revision": "1" * 40, "dirty": False},
        "implementation": academic._implementation_identity(),
        "selection": {
            "split": "validation",
            "tasks": list(academic.DEFAULT_TASKS),
            "seed": academic.DEFAULT_SELECTION_SEED,
            "examples_per_task_limit": academic.DEFAULT_EXAMPLES_PER_TASK,
            "partial": True,
            "panel_status": "confirmatory",
            "ranking": "sha256(seed\\0task\\0example_id), then example_id",
        },
        "controls": {
            "names": list(academic.CONTROLS),
            "shuffled": (
                "next lexicographically sorted unique image SHA-256 within exact task and "
                "Molmo2 pooled-grid/task-stratum pairing, skipping the recipient's donor "
                "content group (AI2D source base diagram)"
            ),
            "blank": "solid RGB(0,0,0) image at the recipient image dimensions",
        },
        "contamination": {
            "method": "exact encoded-image-byte SHA-256 intersection",
            "alignment_train_image_inventory": {},
            "reported_subset": "exact_byte_nonoverlap",
            "limitation": "exact-byte non-overlap is not semantic contamination cleanliness",
        },
        "tasks": manifest_tasks,
    }
    return tasks, academic._attach_content_sha256(manifest)


class _SyntheticReceiptTokenizer:
    eos_token_id = 100257

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        words = {7: "yes", 8: "100", 9: "no"}
        return " ".join(words[token_id] for token_id in token_ids if token_id != self.eos_token_id)


def test_receipt_rows_and_aggregates_are_rederived(academic):
    tasks, manifest = _synthetic_tasks_and_manifest(academic)
    tokenizer = _SyntheticReceiptTokenizer()
    validation_args = {
        "manifest": manifest,
        "loaded": None,
        "tokenizer": tokenizer,
        "text_vocab_size": 100278,
    }
    academic._validate_receipt_tasks(tasks, **validation_args)

    row_tamper = copy.deepcopy(tasks)
    row_tamper["vqav2"]["examples"][0]["controls"]["correct"]["score"] = 0.0
    with pytest.raises(ValueError, match="score was not rederived"):
        academic._validate_receipt_tasks(row_tamper, **validation_args)

    aggregate_tamper = copy.deepcopy(tasks)
    aggregate_tamper["docvqa"]["controls"]["correct"]["anls"] = 0.0
    with pytest.raises(ValueError, match="controls differs"):
        academic._validate_receipt_tasks(aggregate_tamper, **validation_args)


def test_receipt_rejects_generated_id_and_candidate_argmax_tampering(academic):
    tasks, manifest = _synthetic_tasks_and_manifest(academic)
    validation_args = {
        "manifest": manifest,
        "loaded": None,
        "tokenizer": _SyntheticReceiptTokenizer(),
        "text_vocab_size": 100278,
    }

    generated_tamper = copy.deepcopy(tasks)
    generated_tamper["vqav2"]["examples"][0]["controls"]["correct"]["generated_token_ids"][0] = 9
    with pytest.raises(ValueError, match="decoded generated tokens differ"):
        academic._validate_receipt_tasks(generated_tamper, **validation_args)

    vocabulary_tamper = copy.deepcopy(tasks)
    vocabulary_tamper["vqav2"]["examples"][0]["controls"]["correct"]["generated_token_ids"][
        0
    ] = 100278
    with pytest.raises(ValueError, match="generated-token identity differs"):
        academic._validate_receipt_tasks(vocabulary_tamper, **validation_args)

    argmax_tamper = copy.deepcopy(tasks)
    probabilities = argmax_tamper["ai2d"]["examples"][0]["controls"]["correct"][
        "candidate_log_probabilities"
    ]
    probabilities["A"], probabilities["B"] = probabilities["B"], probabilities["A"]
    with pytest.raises(ValueError, match="deterministic probability argmax"):
        academic._validate_receipt_tasks(argmax_tamper, **validation_args)


def test_ai2d_adapter_keeps_encoded_images_lazy_and_blank_positions(academic):
    encoded = io.BytesIO()
    Image.new("RGB", (8, 4), color=(1, 2, 3)).save(encoded, format="PNG")
    image_reference = {"bytes": encoded.getvalue(), "path": None}
    raw = [
        {
            "question_id": "4185.png-0",
            "image": image_reference,
            "question": "Choose.",
            "answer_texts": ["red", "blue", "", "green"],
            "correct_answer": 3,
            "has_transparent_box": True,
        }
    ]
    examples = academic._ai2d_examples_from_raw(raw)
    assert examples[0].visual is image_reference
    assert examples[0].image_reference is image_reference
    assert not isinstance(examples[0].visual, Image.Image)
    assert examples[0].options == ("red", "blue", "", "green")
    assert examples[0].answer_index == 3
    assert examples[0].stratum == "transparent"


def test_a_okvqa_adapter_preserves_known_blank_distractor(academic, monkeypatch):
    row = {
        "image": "unused.png",
        "question": "Which letter?",
        "options": ["s", "", "b", "f"],
        "answer_idx": 3,
        "metadata": {"example_id": "iTeGLgJNuQjkRqeTmKwWLs"},
    }
    monkeypatch.setitem(
        academic.academic_registry.ACADEMIC_REGISTRY,
        "a_okvqa_mc",
        SimpleNamespace(loader=lambda split: [row]),
    )
    examples = academic._load_task_examples("a_okvqa_mc")
    assert examples[0].example_id == "iTeGLgJNuQjkRqeTmKwWLs"
    assert examples[0].source_position == "0"
    assert examples[0].options == ("s", "", "b", "f")
    assert examples[0].answer_index == 3


def test_ai2d_strata_and_exact_nonoverlap_pair_deltas(academic):
    rows = []
    specifications = (
        ("standard", False, False, 1.0, 0.0, 0.0),
        ("transparent", False, True, 0.0, 1.0, 1.0),
        ("standard", True, False, 0.0, 0.0, 0.0),
    )
    for index, (stratum, recipient_overlap, donor_overlap, correct, shuffled, blank) in enumerate(
        specifications
    ):
        rows.append(
            {
                "example_id": str(index),
                "stratum": stratum,
                "alignment_train_image_overlap": recipient_overlap,
                "shuffled_alignment_train_image_overlap": donor_overlap,
                "controls": {
                    "correct": {"score": correct},
                    "shuffled": {"score": shuffled},
                    "blank": {"score": blank},
                },
            }
        )
    result = academic._aggregate_task_outputs("ai2d", rows)
    metrics = result["controls"]["correct"]
    assert metrics["multiple_choice_accuracy_standard"] == pytest.approx(0.5)
    assert metrics["multiple_choice_accuracy_transparent"] == 0.0
    assert result["controls"]["shuffled"]["exact_byte_nonoverlap_multiple_choice_accuracy"] == 0.0
    deltas = result["image_control_deltas"]
    assert deltas[
        "exact_byte_nonoverlap_multiple_choice_accuracy_correct_minus_shuffled"
    ] == pytest.approx(1.0)
    assert deltas["exact_byte_nonoverlap_correct_minus_shuffled_examples"] == 1
    assert deltas[
        "exact_byte_nonoverlap_multiple_choice_accuracy_correct_minus_blank"
    ] == pytest.approx(0.0)
    assert deltas["exact_byte_nonoverlap_correct_minus_blank_examples"] == 2


def test_generation_stop_counts_distinguish_eos_and_cap(academic):
    rows = []
    for stop_reason in ("eos", "max_tokens"):
        rows.append(
            {"controls": {control: {"stop_reason": stop_reason} for control in academic.CONTROLS}}
        )
    counts = academic._generation_stop_counts("docvqa", rows)
    assert counts == {control: {"eos": 1, "max_tokens": 1} for control in academic.CONTROLS}


def test_answer_token_coverage_and_frozen_docvqa_headroom(academic):
    class WhitespaceTokenizer:
        def encode(self, value, add_special_tokens=False):
            assert add_special_tokens is False
            return list(range(len(value.split())))

    loaded = {}
    for task in ("vqav2", "textvqa", "docvqa", "chartqa"):
        examples = []
        for index, answers in enumerate((("one two three",), ("one", "one two"))):
            examples.append(
                academic.AcademicExample(
                    task=task,
                    example_id=f"{task}-{index}",
                    source_position=str(index),
                    visual=None,
                    image_reference=None,
                    question="Question?",
                    answers=answers,
                )
            )
        loaded[task] = {example.example_id: example for example in examples}
    coverage = academic._answer_token_coverage(loaded, WhitespaceTokenizer())
    assert all(value["max_shortest_response_tokens"] == 3 for value in coverage.values())
    assert all(value["rows_exceeding_cap"] == 0 for value in coverage.values())
    assert academic.EXPECTED_ANSWER_TOKEN_COVERAGE["docvqa"] == {
        "selected": 512,
        "max_shortest_response_tokens": 19,
        "max_shortest_response_tokens_with_eos": 20,
        "rows_exceeding_cap": 0,
        "rows_without_eos_room": 0,
        "rows_over_8_response_tokens": 33,
        "ordered_rows_sha256": ("b3e918b62495e706490d7e180785476d88d0c055b92ab2e1f775ed386442bbf2"),
    }


def test_image_encoding_runs_under_inference_mode(academic, monkeypatch):
    class FakeModel:
        def encode_images(self, images, pooling):
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            return torch.zeros(1)

    class FakeTrainModule:
        device = torch.device("cpu")

        def __init__(self):
            self.model_parts = [FakeModel()]

        def model_forward_no_pipeline(self, *args, **kwargs):
            logits = torch.zeros((1, 1, 100))
            logits[0, 0, 10] = 2.0
            logits[0, 0, 11] = 1.0
            return logits

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

    inference = academic._NativeAcademicInference(
        FakeTrainModule(),
        FakeTokenizer(),
        SimpleNamespace(image_token_ids=frozenset({90})),
        max_sequence_length=academic.DEFAULT_MAX_SEQUENCE_LENGTH,
        max_crops=academic.DEFAULT_MAX_CROPS,
        max_new_tokens=academic.DEFAULT_MAX_NEW_TOKENS,
        sequence_bucket_size=academic.DEFAULT_SEQUENCE_BUCKET_SIZE,
    )
    inference._prepare_visual = lambda image: (
        torch.zeros(1),
        torch.zeros(1),
        [90],
        (14, 14, 14, 14),
    )
    monkeypatch.setattr(
        academic, "document_prompt_ids", lambda tokenizer, prompt, image_ids: [1, 90, 2]
    )
    monkeypatch.setattr(
        academic,
        "response_ids",
        lambda tokenizer, letter: [10 + "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(letter)],
    )
    example = academic.AcademicExample(
        task="ai2d",
        example_id="ai2d-0",
        source_position="0",
        visual=None,
        image_reference=None,
        question="Choose.",
        options=("first", "second"),
        answer_index=0,
    )
    output = inference.predict(example, Image.new("RGB", (2, 2)))
    assert output["predicted_index"] == 0
    assert output["image_grid_signature"] == [14, 14, 14, 14]


def test_parser_freezes_practical_defaults(academic):
    build = academic._parser().parse_args(
        ["build-manifest", "--output", "out.json", "--train-image-inventory", "train.sha256"]
    )
    assert build.examples_per_task == 512
    evaluate = academic._parser().parse_args(
        [
            "evaluate",
            "--manifest",
            "manifest.json",
            "--checkpoint",
            "step12000",
            "--output",
            "receipt.json",
        ]
    )
    assert evaluate.max_new_tokens == 24
    assert evaluate.max_crops == 8
    assert evaluate.max_sequence_length == 8192
