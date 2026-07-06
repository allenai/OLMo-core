#!/usr/bin/env python3
"""
Script to create a local dataset of ~100 long documents (32k+ OLMo tokens)
from 5 HuggingFace datasets: pg19, govreport, finepdfs, pile-of-law, blbooks.

Takes 20 random documents from each dataset.
"""

import random
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Minimum token count required
MIN_TOKENS = 32_000
DOCS_PER_DATASET = 25
OUTPUT_DIR = Path(__file__).parent.parent / "long_docs"

# OLMo tokenizer
TOKENIZER_NAME = "allenai/dolma2-tokenizer"


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the OLMo tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_long_docs(
    dataset_name: str,
    text_field: str,
    tokenizer,
    n_docs: int = DOCS_PER_DATASET,
    subset: Optional[str] = None,
    split: str = "train",
    shuffle_seed: int = 42,
    shuffle_buffer: int = 10000,
) -> list[dict]:
    """
    Sample n_docs documents with at least MIN_TOKENS tokens from a dataset.

    Shuffles the dataset first, then iteratively checks documents until
    n_docs valid documents are found.

    Args:
        dataset_name: HuggingFace dataset name
        text_field: Field containing the document text
        tokenizer: Tokenizer to use for counting tokens
        n_docs: Number of documents to sample
        subset: Dataset subset/config name if applicable
        split: Dataset split to use
        shuffle_seed: Seed for shuffling
        shuffle_buffer: Buffer size for streaming shuffle

    Returns:
        List of dicts with 'text', 'source', 'token_count' fields
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"{'='*60}")

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    # Shuffle the dataset
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    long_docs = []
    candidates_checked = 0

    print(f"Searching for {n_docs} documents with {MIN_TOKENS}+ tokens...")

    for example in tqdm(ds, desc="Scanning"):
        candidates_checked += 1

        # Handle nested text fields (e.g., "text.text")
        text = example
        for key in text_field.split("."):
            text = text[key]

        if not text or not isinstance(text, str):
            continue

        # Quick length check before tokenizing (rough estimate: 4 chars per token)
        if len(text) < MIN_TOKENS * 3:
            continue

        token_count = count_tokens(text, tokenizer)

        if token_count >= MIN_TOKENS:
            long_docs.append({
                "text": text,
                "source": f"{dataset_name}" + (f"/{subset}" if subset else ""),
                "token_count": token_count,
            })
            print(f"  Found doc #{len(long_docs)} with {token_count:,} tokens")

            if len(long_docs) >= n_docs:
                break

    print(f"  Checked {candidates_checked} candidates, found {len(long_docs)} valid docs")

    if len(long_docs) < n_docs:
        print(f"  Warning: Only found {len(long_docs)} docs with {MIN_TOKENS}+ tokens")

    return long_docs


def main():
    random.seed(42)

    print("Loading OLMo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    all_docs = []

    # Dataset configurations: (name, text_field, subset, split)
    datasets_config = [
        # PG-19: Full books from Project Gutenberg
        ("manu/project_gutenberg", "text", None, "en"),

        # GovReport: US government reports
        ("ccdv/govreport-summarization", "report", None, "train"),

        # FinePDFs: Large PDF corpus (use a specific subset for manageability)
        ("HuggingFaceFW/finepdfs", "text", "eng_Latn", "train"),

        # Pile of Law: Legal documents (use nlrb_decisions subset - court opinions)
        ("pile-of-law/pile-of-law", "text", "nlrb_decisions", "train"),

        # British Library Books: Historical books
        #("TheBritishLibrary/blbooks", "text", None, "train"),
    ]

    for dataset_name, text_field, subset, split in datasets_config:
        docs = sample_long_docs(
            dataset_name=dataset_name,
            text_field=text_field,
            tokenizer=tokenizer,
            subset=subset,
            split=split,
        )
        all_docs.extend(docs)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents collected: {len(all_docs)}")

    if not all_docs:
        print("No documents found! Exiting.")
        return

    # Create dataset
    dataset = Dataset.from_list(all_docs)

    # Print stats
    token_counts = [d["token_count"] for d in all_docs]
    print(f"Token count stats:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Mean: {sum(token_counts) / len(token_counts):,.0f}")

    # Count by source
    from collections import Counter
    source_counts = Counter(d["source"] for d in all_docs)
    print(f"\nDocuments by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    # Save dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"\nDataset saved to: {OUTPUT_DIR}")

    # Also save as JSON for easy inspection
    json_path = OUTPUT_DIR / "long_docs.json"
    dataset.to_json(str(json_path))
    print(f"JSON backup saved to: {json_path}")


if __name__ == "__main__":
    main()
