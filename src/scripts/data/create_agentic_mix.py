"""
Creates allenai/Dolci-Think-SFT-32B-agentic

Mix B: Scales all existing Dolci-Think sources to 65%, adds Nemotron
Conversational Agent (600K) and Nemotron Terminal Agent (170K).

Expected total: ~18.4B tokens
"""

import random
from collections import defaultdict

from datasets import Dataset, load_dataset

HF_REPO = "allenai/Dolci-Think-SFT-32B-agentic"
SEED = 42
EXISTING_SCALE = 0.65

NEMOTRON_CONV_AGENT_N = 600_000
NEMOTRON_TERMINAL_AGENT_N = 170_000


def main():
    rng = random.Random(SEED)

    print("Loading Dolci-Think-SFT-32B-75pct...")
    base = load_dataset("allenai/Dolci-Think-SFT-32B-75pct", split="train")

    # Group by source and sample 65%
    by_source = defaultdict(list)
    for ex in base:
        by_source[ex["source"]].append(ex)

    examples = []
    for source, pool in by_source.items():
        n = max(1, int(len(pool) * EXISTING_SCALE))
        sampled = rng.sample(pool, n)
        examples.extend(sampled)
        print(f"  {source}: {len(pool):,} → {n:,} (65%)")

    print(f"\nDolci-Think total (scaled): {len(examples):,}")

    # Add Nemotron Conversational Agent
    print(f"\nStreaming {NEMOTRON_CONV_AGENT_N:,} Nemotron Conversational Agent examples...")
    conv_ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "conversational_agent", split="train", streaming=True)
    n_conv = 0
    for ex in conv_ds:
        if "messages" not in ex:
            continue
        examples.append({"messages": ex["messages"], "source": "Nemotron Conversational Agent"})
        n_conv += 1
        if n_conv >= NEMOTRON_CONV_AGENT_N:
            break
    print(f"  Added {n_conv:,} examples")

    # Add Nemotron Terminal Agent
    print(f"\nStreaming {NEMOTRON_TERMINAL_AGENT_N:,} Nemotron Terminal Agent examples...")
    term_ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", "terminal_agent", split="train", streaming=True)
    n_term = 0
    for ex in term_ds:
        if "messages" not in ex:
            continue
        examples.append({"messages": ex["messages"], "source": "Nemotron Terminal Agent"})
        n_term += 1
        if n_term >= NEMOTRON_TERMINAL_AGENT_N:
            break
    print(f"  Added {n_term:,} examples")

    print(f"\nTotal examples: {len(examples):,}")

    rng.shuffle(examples)

    dataset = Dataset.from_list(examples)
    print(f"\nPushing to {HF_REPO}...")
    dataset.push_to_hub(HF_REPO, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
