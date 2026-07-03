"""
Creates four Nemotron-Cascade-2 subset datasets on HuggingFace:
  - allenai/Nemotron-Cascade-2-Chat           (~14M examples)
  - allenai/Nemotron-Cascade-2-IF             (~820K examples)
  - allenai/Nemotron-Cascade-2-ConvAgent      (~822K examples)
  - allenai/Nemotron-Cascade-2-TerminalAgent  (~822K examples)

These are reusable components for Mix A and Mix B tokenization jobs.
Run once; the tokenization scripts reference these by ratio.
"""

from datasets import Dataset, load_dataset

SUBSETS = {
    "chat": "allenai/Nemotron-Cascade-2-Chat",
    "instruction_following": "allenai/Nemotron-Cascade-2-IF",
    "conversational_agent": "allenai/Nemotron-Cascade-2-ConvAgent",
    "terminal_agent": "allenai/Nemotron-Cascade-2-TerminalAgent",
}


def main():
    for subset, hf_repo in SUBSETS.items():
        print(f"\nProcessing {subset} → {hf_repo}")
        ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", subset, split="train")
        # Keep only messages and add source field
        ds = ds.map(
            lambda ex: {"messages": ex["messages"], "source": hf_repo.split("/")[-1]},
            remove_columns=[c for c in ds.column_names if c not in ("messages",)],
        )
        print(f"  {len(ds):,} examples, pushing to {hf_repo}...")
        ds.push_to_hub(hf_repo, private=True)
        print(f"  Done.")


if __name__ == "__main__":
    main()
