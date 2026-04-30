import argparse
import json
import random
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_STYLES = ("letters", "numbers")

# Chosen to give a reasonably large pool while staying simple and visually distinct.
WORD_POOL = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "grey",
    "silver",
    "gold",
    "cyan",
    "lime",
    "teal",
    "navy",
    "beige",
    "coral",
    "peach",
    "olive",
    "maroon",
    "indigo",
    "violet",
    "amber",
    "ivory",
]


def build_labels(*, num_options: int, label_style: str) -> list[str]:
    if label_style not in LABEL_STYLES:
        raise ValueError(f"label_style must be one of {LABEL_STYLES} (got {label_style!r})")
    if label_style == "letters":
        if num_options > len(LETTERS):
            raise ValueError(
                f"num_options={num_options} exceeds supported letter label count={len(LETTERS)}"
            )
        return list(LETTERS[:num_options])
    return [str(idx) for idx in range(1, num_options + 1)]


def render_prefix_prompt(options: list[dict[str, str]], target_word: str) -> str:
    lines = [f"{item['label']}. {item['word']}" for item in options]
    lines.append(f"{target_word} is option:")
    return "\n".join(lines)


def render_suffix_prompt(options: list[dict[str, str]], target_word: str) -> str:
    lines = [f"{item['word']} ({item['label']})" for item in options]
    lines.append(f"{target_word} is option:")
    return "\n".join(lines)


def generate_dataset(
    *,
    num_examples: int,
    num_options: int,
    seed: int,
    label_style: str = "letters",
) -> dict:
    if num_options < 2:
        raise ValueError(f"num_options must be >= 2 (got {num_options})")
    if num_options > len(WORD_POOL):
        raise ValueError(f"num_options={num_options} exceeds WORD_POOL size={len(WORD_POOL)}")
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1 (got {num_examples})")

    rng = random.Random(seed)
    labels = build_labels(num_options=num_options, label_style=label_style)
    examples = []
    for example_idx in range(num_examples):
        sampled_words = rng.sample(WORD_POOL, num_options)
        options = [
            {"label": labels[idx], "letter": labels[idx], "word": word}
            for idx, word in enumerate(sampled_words)
        ]
        answer_index = rng.randrange(num_options)
        target_word = options[answer_index]["word"]
        answer_label = options[answer_index]["label"]

        examples.append(
            {
                "example_id": example_idx,
                "num_options": num_options,
                "target_word": target_word,
                "answer_label": answer_label,
                "answer_letter": answer_label,
                "options": options,
                "prefix_prompt": render_prefix_prompt(options, target_word),
                "suffix_prompt": render_suffix_prompt(options, target_word),
            }
        )

    return {
        "meta": {
            "task": "copylabels",
            "num_examples": num_examples,
            "num_options": num_options,
            "seed": seed,
            "label_style": label_style,
            "labels": labels,
            "word_pool": WORD_POOL,
        },
        "examples": examples,
    }


def save_dataset(dataset: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")


def load_dataset(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paired copylabels dataset.")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--num-options", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-style", type=str, default="letters", choices=LABEL_STYLES)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the JSON dataset file to write.",
    )
    args = parser.parse_args()

    dataset = generate_dataset(
        num_examples=args.num_examples,
        num_options=args.num_options,
        seed=args.seed,
        label_style=args.label_style,
    )
    save_dataset(dataset, args.output)

    first = dataset["examples"][0]
    print(f"Wrote dataset to {args.output}")
    print(
        f"Examples: {dataset['meta']['num_examples']}, "
        f"options/example: {dataset['meta']['num_options']}, seed: {dataset['meta']['seed']}, "
        f"label_style: {dataset['meta']['label_style']}"
    )
    print("First prefix example:")
    print(first["prefix_prompt"])
    print(f"Gold: {first['answer_label']}")
    print()
    print("First suffix example:")
    print(first["suffix_prompt"])
    print(f"Gold: {first['answer_label']}")


if __name__ == "__main__":
    main()
