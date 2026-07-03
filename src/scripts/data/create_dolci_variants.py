"""
Creates two filtered variants of Dolci-Think-SFT-32B-75pct:

  allenai/Dolci-Think-SFT-32B-math-reduced
    - OT3 Math capped at 150K (down from ~564K)
    - All other sources unchanged
    - Used in Mix A (low-math + chat)

  allenai/Dolci-Think-SFT-32B-75pct-scaled-65
    - All sources scaled to 65% of their baseline count
    - Used in Mix B (agentic)

Token budget math (with avg tokens/instance from sampling):
  math-reduced Dolci:   ~9.80B tokens
  75pct-scaled-65:      ~11.95B tokens
"""

import random
from collections import defaultdict

from datasets import Dataset, load_dataset

SEED = 42
MATH_SOURCE = "Dolci Think OpenThoughts3 Math"
MATH_CAP = 150_000
SCALE_FACTOR = 0.65


def load_by_source(dataset_name: str) -> dict[str, list]:
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    by_source: dict[str, list] = defaultdict(list)
    for ex in ds:
        by_source[ex["source"]].append(ex)
    for src, pool in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(pool):,}")
    return by_source


def main():
    rng = random.Random(SEED)
    by_source = load_by_source("allenai/Dolci-Think-SFT-32B-75pct")

    # --- Math-reduced variant ---
    print("\nBuilding math-reduced variant...")
    examples = []
    for source, pool in by_source.items():
        if source == MATH_SOURCE:
            sampled = rng.sample(pool, min(MATH_CAP, len(pool)))
        else:
            sampled = pool
        examples.extend(sampled)
        print(f"  {source}: {len(sampled):,}")

    rng.shuffle(examples)
    print(f"\nTotal: {len(examples):,} examples")
    print("Pushing to allenai/Dolci-Think-SFT-32B-math-reduced...")
    Dataset.from_list(examples).push_to_hub("allenai/Dolci-Think-SFT-32B-math-reduced", private=True)
    print("Done.")

    # --- Scaled-65 variant ---
    print("\nBuilding 75pct-scaled-65 variant...")
    examples = []
    for source, pool in by_source.items():
        n = max(1, int(len(pool) * SCALE_FACTOR))
        sampled = rng.sample(pool, n)
        examples.extend(sampled)
        print(f"  {source}: {len(pool):,} → {n:,}")

    rng.shuffle(examples)
    print(f"\nTotal: {len(examples):,} examples")
    print("Pushing to allenai/Dolci-Think-SFT-32B-75pct-scaled-65...")
    Dataset.from_list(examples).push_to_hub("allenai/Dolci-Think-SFT-32B-75pct-scaled-65", private=True)
    print("Done.")


if __name__ == "__main__":
    main()
