"""
Creates allenai/Dolci-Think-SFT-32B-low-math-chat

Mix A: Dramatically reduces math (150K vs 564K in baseline), replaces with
Nemotron General Chat (2M) and Nemotron Instruction Following (1.5M).

Expected total: ~18.6B tokens
"""

import random
from collections import defaultdict

from datasets import Dataset, load_dataset

HF_REPO = "allenai/Dolci-Think-SFT-32B-low-math-chat"
SEED = 42

# How many instances to keep per source from Dolci-Think-SFT-32B-75pct.
# None = keep all instances of that source.
SOURCE_CAPS = {
    "Dolci Think OpenThoughts3 Math": 150_000,
    "Dolci Think Python Algorithms": None,
    "Persona Precise IF": None,
    "Dolci Think Precise IF": None,
    "Nemotron Code": None,
    "SYNTHETIC-2-SFT-Verified": None,
    "Dolci Think OpenThoughts3 STEM": None,
    "Aya": None,
    "Dolci Think OpenThoughts3 Code": None,
    "Wildchat": None,
    "WildJailbreak": None,
    "WildGuardMix": None,
    "CoCoNot": None,
    "OpenAssistant": None,
    "TableGPT": None,
    "Olmo Identity Hardcoded Data": None,
}

# IF subset only has ~820K total — use all of it
# Bump chat to 2.35M to compensate and keep total ~18.6B tokens
NEMOTRON_CHAT_N = 2_350_000
NEMOTRON_IF_N = 820_000  # use full IF subset


def main():
    rng = random.Random(SEED)

    print("Loading Dolci-Think-SFT-32B-75pct...")
    base = load_dataset("allenai/Dolci-Think-SFT-32B-75pct", split="train")

    # Group by source
    by_source = defaultdict(list)
    for ex in base:
        by_source[ex["source"]].append(ex)

    # Sample per source
    examples = []
    for source, cap in SOURCE_CAPS.items():
        pool = by_source.get(source, [])
        if cap is not None and len(pool) > cap:
            pool = rng.sample(pool, cap)
        examples.extend(pool)
        print(f"  {source}: {len(pool):,}")

    print(f"\nDolci-Think total: {len(examples):,}")

    # Add Nemotron General Chat
    print(f"\nStreaming {NEMOTRON_CHAT_N:,} Nemotron Chat examples...")
    chat_ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "chat", split="train", streaming=True)
    n_chat = 0
    for ex in chat_ds:
        if "messages" not in ex:
            continue
        examples.append({"messages": ex["messages"], "source": "Nemotron Chat"})
        n_chat += 1
        if n_chat >= NEMOTRON_CHAT_N:
            break
    print(f"  Added {n_chat:,} Nemotron Chat examples")

    # Add Nemotron Instruction Following
    print(f"\nStreaming {NEMOTRON_IF_N:,} Nemotron IF examples...")
    if_ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "instruction_following", split="train", streaming=True)
    n_if = 0
    for ex in if_ds:
        if "messages" not in ex:
            continue
        examples.append({"messages": ex["messages"], "source": "Nemotron Instruction Following"})
        n_if += 1
        if n_if >= NEMOTRON_IF_N:
            break
    print(f"  Added {n_if:,} Nemotron IF examples")

    print(f"\nTotal examples: {len(examples):,}")

    rng.shuffle(examples)

    dataset = Dataset.from_list(examples)
    print(f"\nPushing to {HF_REPO}...")
    dataset.push_to_hub(HF_REPO, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
