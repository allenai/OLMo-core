import argparse
from collections import Counter
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoModelForCausalLM

try:
    from generate_copylabels_dataset import generate_dataset, load_dataset, save_dataset
except ModuleNotFoundError:
    from .generate_copylabels_dataset import (
        generate_dataset,
        load_dataset,
        save_dataset,
    )


@dataclass
class FailureExample:
    example_id: int
    target_word: str
    gold_label: str
    pred_label: str
    prompt: str
    top_scores: list[tuple[str, float]]


def normalize_torch_device(device_like) -> torch.device | None:
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")
    if isinstance(device_like, str):
        if device_like in {"cpu", "disk"}:
            return None
        return torch.device(device_like)
    return None


def get_model_input_device(model, fallback_device: torch.device) -> torch.device:
    try:
        embedding_device = model.get_input_embeddings().weight.device
        if embedding_device.type != "meta":
            return embedding_device
    except Exception:
        pass

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for mapped_device in hf_device_map.values():
            torch_device = normalize_torch_device(mapped_device)
            if torch_device is not None:
                return torch_device

    return fallback_device


def load_model_for_inference(
    *,
    load_path: str,
    model_dtype: torch.dtype,
    fallback_device: torch.device,
    device_map_mode: str,
):
    load_kwargs = {
        "trust_remote_code": True,
        "dtype": model_dtype,
    }
    if device_map_mode == "auto":
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    if device_map_mode == "none":
        model = model.to(fallback_device)
    model.eval()
    input_device = get_model_input_device(model, fallback_device)
    return model, input_device


def score_completion_logprobs(
    model,
    prompt_ids: list[int],
    completion_token_ids: dict[str, list[int]],
    device: torch.device,
    pad_token_id: int,
) -> dict[str, float]:
    labels = list(completion_token_ids)
    full_sequences = [prompt_ids + completion_token_ids[label] for label in labels]
    max_length = max(len(seq) for seq in full_sequences)

    input_ids = torch.full(
        (len(full_sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(full_sequences), max_length),
        dtype=torch.long,
        device=device,
    )

    for row_idx, seq in enumerate(full_sequences):
        seq_len = len(seq)
        input_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[row_idx, :seq_len] = 1

    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attention_mask).logits.float()
        logprobs = logits.log_softmax(dim=-1)

    scores: dict[str, float] = {}
    prompt_len = len(prompt_ids)
    for row_idx, label in enumerate(labels):
        score = 0.0
        for step_idx, token_id in enumerate(completion_token_ids[label]):
            token_pos = prompt_len + step_idx
            score += float(logprobs[row_idx, token_pos - 1, token_id])
        scores[label] = score

    return scores


def sort_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def build_label_completion_ids(
    tokenizer,
    prompt: str,
    allowed_labels: list[str],
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_completion_ids: dict[str, list[int]] = {}

    for label in allowed_labels:
        completion_ids = tokenizer(prompt + " " + label, add_special_tokens=False).input_ids
        suffix_ids = completion_ids[len(prompt_ids) :]
        if not suffix_ids:
            raise ValueError(
                f"Empty completion for label {label!r} after prompt ending {prompt[-40:]!r}"
            )
        target_completion_ids[label] = suffix_ids

    return target_completion_ids


def get_option_label(option: dict) -> str:
    if "label" in option:
        return str(option["label"])
    return str(option["letter"])


def get_answer_label(example: dict) -> str:
    if "answer_label" in example:
        return str(example["answer_label"])
    return str(example["answer_letter"])


def render_completed_example(prompt: str, answer_label: str) -> str:
    return f"{prompt} {answer_label}"


def build_few_shot_prompt(
    *,
    prompt_key: str,
    shot_examples: list[dict],
    query_example: dict,
) -> str:
    blocks = [
        render_completed_example(ex[prompt_key], get_answer_label(ex)) for ex in shot_examples
    ]
    blocks.append(query_example[prompt_key])
    return "\n\n".join(blocks)


def evaluate_prompt_family(
    *,
    model,
    tokenizer,
    examples: list[dict],
    shot_examples: list[dict],
    prompt_key: str,
    device: torch.device,
    max_failures: int,
) -> dict:
    # num_options = examples[0]["num_options"]
    allowed_labels = [get_option_label(option) for option in examples[0]["options"]]

    correct = 0
    gold_dist: Counter = Counter()
    pred_dist: Counter = Counter()
    correct_by_label: Counter = Counter()
    total_by_label: Counter = Counter()
    failures: list[FailureExample] = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for ex in examples:
        prompt = build_few_shot_prompt(
            prompt_key=prompt_key,
            shot_examples=shot_examples,
            query_example=ex,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_completion_ids = build_label_completion_ids(tokenizer, prompt, allowed_labels)
        scores = score_completion_logprobs(
            model,
            prompt_ids,
            target_completion_ids,
            device,
            pad_token_id,
        )
        ranked_scores = sort_scores(scores)

        gold_label = get_answer_label(ex)
        pred_label = ranked_scores[0][0]

        correct += int(pred_label == gold_label)
        gold_dist[gold_label] += 1
        pred_dist[pred_label] += 1
        total_by_label[gold_label] += 1
        correct_by_label[gold_label] += int(pred_label == gold_label)

        if pred_label != gold_label and len(failures) < max_failures:
            failures.append(
                FailureExample(
                    example_id=ex["example_id"],
                    target_word=ex["target_word"],
                    gold_label=gold_label,
                    pred_label=pred_label,
                    prompt=prompt,
                    top_scores=ranked_scores[:5],
                )
            )

    accuracy_by_label = {
        label: (correct_by_label[label] / total_by_label[label])
        for label in allowed_labels
        if total_by_label[label] > 0
    }

    return {
        "accuracy": correct / len(examples),
        "gold_dist": gold_dist,
        "pred_dist": pred_dist,
        "accuracy_by_label": accuracy_by_label,
        "failures": failures,
        "allowed_labels": allowed_labels,
        "num_shots": len(shot_examples),
    }


def print_results(
    *,
    title: str,
    results: dict,
) -> None:
    allowed_labels = results["allowed_labels"]
    print(title)
    print(f"Shots: {results['num_shots']}")
    print(f"Chance accuracy: {1 / len(allowed_labels):.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(
        "Prediction distribution:",
        {label: results["pred_dist"].get(label, 0) for label in allowed_labels},
    )
    print(
        "Ground-truth distribution:",
        {label: results["gold_dist"].get(label, 0) for label in allowed_labels},
    )
    print(
        "Accuracy by gold label:",
        {label: round(results["accuracy_by_label"].get(label, 0.0), 4) for label in allowed_labels},
    )
    print()
    print("Representative failures:")
    if not results["failures"]:
        print("None")
    for failure in results["failures"]:
        print("---" * 10)
        print(f"Example {failure.example_id}, target word: {failure.target_word}")
        print(f"Gold label: {failure.gold_label}")
        print(f"Pred label: {failure.pred_label}")
        print("Prompt:")
        print(failure.prompt)
        print(f"Top label scores: {failure.top_scores}")
    print()


def main() -> None:
    default_device_map = (
        "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "none"
    )
    parser = argparse.ArgumentParser(
        description="Probe label-binding on synthetic copylabels datasets."
    )
    parser.add_argument("--load-path", type=str, default="/workspace/tmp/step96085-hf")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional JSON dataset file produced by generate_copylabels_dataset.py.",
    )
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--num-options", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-style", type=str, default="letters", choices=("letters", "numbers")
    )
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument(
        "--device-map", type=str, default=default_device_map, choices=("none", "auto")
    )
    parser.add_argument(
        "--save-generated-dataset",
        type=str,
        default=None,
        help="Optional path to save the generated dataset JSON.",
    )
    parser.add_argument("--max-failures", type=int, default=8)
    args = parser.parse_args()

    if args.dataset is not None:
        dataset = load_dataset(args.dataset)
    else:
        total_examples = args.num_examples + args.num_shots
        dataset = generate_dataset(
            num_examples=total_examples,
            num_options=args.num_options,
            seed=args.seed,
            label_style=args.label_style,
        )
        if args.save_generated_dataset is not None:
            save_dataset(dataset, args.save_generated_dataset)

    examples = dataset["examples"]
    meta = dataset["meta"]
    if not examples:
        raise ValueError("Dataset has no examples")
    if args.num_shots < 0:
        raise ValueError(f"num_shots must be >= 0 (got {args.num_shots})")
    if len(examples) <= args.num_shots:
        raise ValueError(
            f"Need more than num_shots examples, got {len(examples)} examples and num_shots={args.num_shots}"
        )

    shot_examples = examples[: args.num_shots]
    eval_examples = examples[args.num_shots :]
    eval_examples = eval_examples[: args.num_examples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path, trust_remote_code=True)
    model, input_device = load_model_for_inference(
        load_path=args.load_path,
        model_dtype=model_dtype,
        fallback_device=device,
        device_map_mode=args.device_map,
    )

    prefix_results = evaluate_prompt_family(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        shot_examples=shot_examples,
        prompt_key="prefix_prompt",
        device=input_device,
        max_failures=args.max_failures,
    )
    suffix_results = evaluate_prompt_family(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        shot_examples=shot_examples,
        prompt_key="suffix_prompt",
        device=input_device,
        max_failures=args.max_failures,
    )

    print(
        f"Dataset: {meta['task']}, stored_examples={meta['num_examples']}, "
        f"eval_examples={len(eval_examples)}, shots={args.num_shots}, "
        f"options/example={meta['num_options']}, seed={meta['seed']}, "
        f"label_style={meta.get('label_style', 'letters')}"
    )
    print(f"Checkpoint: {args.load_path}")
    print(f"Device map: {args.device_map}")
    print(f"Input device: {input_device}")
    print()
    print("Prefix few-shot prompt example:")
    print(
        build_few_shot_prompt(
            prompt_key="prefix_prompt",
            shot_examples=shot_examples,
            query_example=eval_examples[0],
        )
    )
    print(f"Gold: {get_answer_label(eval_examples[0])}")
    print()
    print("Suffix few-shot prompt example:")
    print(
        build_few_shot_prompt(
            prompt_key="suffix_prompt",
            shot_examples=shot_examples,
            query_example=eval_examples[0],
        )
    )
    print(f"Gold: {get_answer_label(eval_examples[0])}")
    print()

    label_style = meta.get("label_style", "letters")
    prefix_example = "A. green" if label_style == "letters" else "1. green"
    suffix_example = "green (A)" if label_style == "letters" else "green (1)"
    print_results(
        title=f"Prefix format: '{prefix_example}' then 'green is option:'", results=prefix_results
    )
    print_results(
        title=f"Suffix format: '{suffix_example}' then 'green is option:'", results=suffix_results
    )


if __name__ == "__main__":
    main()
