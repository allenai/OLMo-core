from collections import Counter
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoModelForCausalLM

COLOR_VOCAB = [
    "yellow",
    "pink",
    "brown",
    "orange",
    "blue",
    "purple",
    "red",
    "white",
    "green",
    "grey",
    "black",
]


@dataclass
class ExampleDiagnostics:
    question: str
    gold_letter: str
    pred_letter: str
    gold_color: str
    pred_mc_color: str
    pred_qonly_color: str
    letter_scores: list[tuple[str, float]]
    mc_color_scores: list[tuple[str, float]]
    qonly_color_scores: list[tuple[str, float]]


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


def decode_letter(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).strip()


def split_last_block(ctx_text: str) -> str:
    return ctx_text.strip().split("\n\n")[-1]


def parse_last_block(last_block: str) -> tuple[str, dict[str, str]]:
    lines = [line for line in last_block.split("\n") if line]
    question = lines[0]
    option_map: dict[str, str] = {}
    for line in lines[1:]:
        if ". " not in line:
            continue
        prefix, color = line.split(". ", 1)
        prefix = prefix.strip()
        if len(prefix) == 1 and "A" <= prefix <= "J":
            option_map[prefix] = color.strip()
    return question, option_map


def score_next_token_logprobs(
    model,
    input_ids: list[int],
    target_token_ids: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    x = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logprobs = model(x).logits[0, -1].float().log_softmax(dim=-1)
    return {label: float(logprobs[token_id]) for label, token_id in target_token_ids.items()}


def sort_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = torch.load("/workspace/OLMo-core/cc_dbg.pt")

    load_path = "/workspace/tmp/step96085-hf"
    device_map_mode = (
        "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "none"
    )
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    model, input_device = load_model_for_inference(
        load_path=load_path,
        model_dtype=model_dtype,
        fallback_device=device,
        device_map_mode=device_map_mode,
    )
    print(f"Device map: {device_map_mode}")
    print(f"Input device: {input_device}")

    choice_ids = examples[0]["choices"]
    letter_token_ids = {decode_letter(tokenizer, token_id): token_id for token_id in choice_ids}
    qonly_color_token_ids = {
        color: tokenizer.encode(" " + color, add_special_tokens=False)[0] for color in COLOR_VOCAB
    }

    letter_correct = 0
    mc_color_correct = 0
    qonly_color_correct = 0
    explicit_color_letter_correct = 0
    explicit_color_options_only_letter_correct = 0
    mc_color_correct_letter_wrong = 0
    qonly_color_correct_letter_wrong = 0

    pred_letter_dist: Counter = Counter()
    gold_letter_dist: Counter = Counter()
    pred_mc_color_dist: Counter = Counter()
    pred_qonly_color_dist: Counter = Counter()

    failure_examples: list[ExampleDiagnostics] = []

    for ex in examples:
        ctx_text = tokenizer.decode(ex["ctx"])
        last_block = split_last_block(ctx_text)
        question, option_map = parse_last_block(last_block)

        gold_letter = decode_letter(tokenizer, ex["choices"][ex["label_id"]])
        gold_color = option_map[gold_letter]

        mc_letter_scores = score_next_token_logprobs(
            model, ex["ctx"], letter_token_ids, input_device
        )
        pred_letter = sort_scores(mc_letter_scores)[0][0]

        mc_color_token_ids = {
            color: tokenizer.encode(" " + color, add_special_tokens=False)[0]
            for color in option_map.values()
        }
        mc_color_scores = score_next_token_logprobs(
            model, ex["ctx"], mc_color_token_ids, input_device
        )
        pred_mc_color = sort_scores(mc_color_scores)[0][0]

        qonly_prompt = question + "\nAnswer:"
        qonly_input_ids = tokenizer(qonly_prompt, return_tensors="pt").input_ids[0].tolist()
        qonly_color_scores = score_next_token_logprobs(
            model, qonly_input_ids, qonly_color_token_ids, input_device
        )
        pred_qonly_color = sort_scores(qonly_color_scores)[0][0]

        option_lines = [f" {letter}. {color}" for letter, color in option_map.items()]
        explicit_color_prompt = question + "\n" + "\n".join(option_lines)
        explicit_color_prompt += f"\nThe correct color is {gold_color}.\nAnswer:"
        explicit_color_letter_scores = score_next_token_logprobs(
            model,
            tokenizer(explicit_color_prompt, return_tensors="pt").input_ids[0].tolist(),
            letter_token_ids,
            input_device,
        )
        pred_explicit_color_letter = sort_scores(explicit_color_letter_scores)[0][0]

        explicit_color_options_only_prompt = "\n".join(option_lines)
        explicit_color_options_only_prompt += f"\nThe correct color is {gold_color}.\nAnswer:"
        explicit_color_options_only_letter_scores = score_next_token_logprobs(
            model,
            tokenizer(explicit_color_options_only_prompt, return_tensors="pt")
            .input_ids[0]
            .tolist(),
            letter_token_ids,
            input_device,
        )
        pred_explicit_color_options_only_letter = sort_scores(
            explicit_color_options_only_letter_scores
        )[0][0]

        letter_correct += int(pred_letter == gold_letter)
        mc_color_correct += int(pred_mc_color == gold_color)
        qonly_color_correct += int(pred_qonly_color == gold_color)
        explicit_color_letter_correct += int(pred_explicit_color_letter == gold_letter)
        explicit_color_options_only_letter_correct += int(
            pred_explicit_color_options_only_letter == gold_letter
        )
        mc_color_correct_letter_wrong += int(
            pred_mc_color == gold_color and pred_letter != gold_letter
        )
        qonly_color_correct_letter_wrong += int(
            pred_qonly_color == gold_color and pred_letter != gold_letter
        )

        pred_letter_dist[pred_letter] += 1
        gold_letter_dist[gold_letter] += 1
        pred_mc_color_dist[pred_mc_color] += 1
        pred_qonly_color_dist[pred_qonly_color] += 1

        if pred_letter != gold_letter and len(failure_examples) < 12:
            failure_examples.append(
                ExampleDiagnostics(
                    question=question,
                    gold_letter=gold_letter,
                    pred_letter=pred_letter,
                    gold_color=gold_color,
                    pred_mc_color=pred_mc_color,
                    pred_qonly_color=pred_qonly_color,
                    letter_scores=sort_scores(mc_letter_scores)[:5],
                    mc_color_scores=sort_scores(mc_color_scores)[:5],
                    qonly_color_scores=sort_scores(qonly_color_scores)[:5],
                )
            )

    n = len(examples)
    labels = [chr(ord("A") + i) for i in range(10)]

    print(f"Examples: {n}")
    print(f"MC letter accuracy (direct logits over A-J): {letter_correct / n:.4f}")
    print(f"MC color accuracy (direct logits over option color words): {mc_color_correct / n:.4f}")
    print(
        f"Question-only color accuracy (direct logits over color words): {qonly_color_correct / n:.4f}"
    )
    print(
        "Explicit-color letter accuracy (question + options + "
        f"'The correct color is X'): {explicit_color_letter_correct / n:.4f}"
    )
    print(
        "Explicit-color letter accuracy (options only + "
        f"'The correct color is X'): {explicit_color_options_only_letter_correct / n:.4f}"
    )
    print()

    print(
        "Cases where the full MC prompt color is correct but the letter is wrong: "
        f"{mc_color_correct_letter_wrong}/{n}"
    )
    print(
        "Cases where the question-only color is correct but the letter is wrong: "
        f"{qonly_color_correct_letter_wrong}/{n}"
    )
    print()

    print(
        "Prediction distribution over letters:",
        {label: pred_letter_dist.get(label, 0) for label in labels},
    )
    print(
        "Ground-truth distribution over letters:",
        {label: gold_letter_dist.get(label, 0) for label in labels},
    )
    print()
    print("Predicted MC colors:", dict(pred_mc_color_dist))
    print("Predicted question-only colors:", dict(pred_qonly_color_dist))
    print()

    print("Representative letter failures:")
    for diag in failure_examples:
        print("---" * 10)
        print(diag.question)
        print(f"Gold letter/color: {diag.gold_letter} / {diag.gold_color}")
        print(f"Pred letter: {diag.pred_letter}")
        print(f"Pred MC color: {diag.pred_mc_color}")
        print(f"Pred question-only color: {diag.pred_qonly_color}")
        print(f"Top letter scores: {diag.letter_scores}")
        print(f"Top MC color scores: {diag.mc_color_scores}")
        print(f"Top question-only color scores: {diag.qonly_color_scores}")
