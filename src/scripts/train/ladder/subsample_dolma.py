#!/usr/bin/env python3
"""
Dolma Dataset Subsampling Script

Analyzes token distribution across topics in the all-dressed-snazzy2 dataset and
generates subsampled YAML configs for the ladder training pipeline.

Usage:
    # Analyze the dataset (print token distribution)
    python src/scripts/train/ladder/subsample_dolma.py analyze

    # Generate subsampled YAML (default 300B target)
    python src/scripts/train/ladder/subsample_dolma.py generate \
        --output=src/scripts/train/ladder/dolma-300B-mix.yaml

    # Custom target
    python src/scripts/train/ladder/subsample_dolma.py generate \
        --target-tokens=500B \
        --output=src/scripts/train/ladder/dolma-500B-mix.yaml
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Dataset constants
S3_BASE_PATH = (
    "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2"
)

# All 24 topics in the dataset
TOPICS = [
    "adult_content",
    "art_and_design",
    "crime_and_law",
    "education_and_jobs",
    "electronics_and_hardware",
    "entertainment",
    "fashion_and_beauty",
    "finance_and_business",
    "food_and_dining",
    "games",
    "health",
    "history_and_geography",
    "home_and_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_math_and_technology",
    "social_life",
    "software",
    "software_development",
    "sports_and_fitness",
    "transportation",
    "travel_and_tourism",
]

# Code fresh validation data paths (relative to mix-base-dir)
CODE_FRESH_PATHS = [
    "s3://ai2-llm/eval-data/perplexity/code_fresh_0825_1225_dolma2-tokenizer/python/val/part-0-00000.npy",
    "s3://ai2-llm/eval-data/perplexity/code_fresh_0825_1225_dolma2-tokenizer/javascript/val/part-0-00000.npy",
]

# Bytes per token for dolma2-tokenizer (uint16)
BYTES_PER_TOKEN = 2


@dataclass
class FileInfo:
    """Information about a single .npy file."""

    path: str
    size_bytes: int

    @property
    def num_tokens(self) -> int:
        return self.size_bytes // BYTES_PER_TOKEN


@dataclass
class TopicStats:
    """Statistics for a single topic."""

    name: str
    files: List[FileInfo] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def total_tokens(self) -> int:
        return sum(f.num_tokens for f in self.files)

    @property
    def num_files(self) -> int:
        return len(self.files)


def parse_token_count(s: str) -> int:
    """Parse a token count string like '300B' or '500M' to an integer."""
    s = s.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMBT]?)$", s)
    if not match:
        raise ValueError(f"Invalid token count format: {s}")

    value = float(match.group(1))
    suffix = match.group(2)

    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    return int(value * multipliers[suffix])


def format_tokens(tokens: int) -> str:
    """Format a token count for display."""
    if tokens >= 1_000_000_000_000:
        return f"{tokens / 1_000_000_000_000:.1f}T"
    elif tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.1f}B"
    elif tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def format_bytes(size_bytes: int) -> str:
    """Format bytes for display."""
    if size_bytes >= 1_000_000_000_000:
        return f"{size_bytes / 1_000_000_000_000:.1f} TB"
    elif size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


def list_s3_files(s3_path: str) -> List[Tuple[str, int]]:
    """
    List all files in an S3 path with their sizes.

    Returns list of (full_path, size_bytes) tuples.
    """
    cmd = ["aws", "s3", "ls", "--recursive", s3_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error listing S3 path {s3_path}: {e.stderr}", file=sys.stderr)
        return []

    files = []
    # Parse aws s3 ls output format: "2024-01-01 12:00:00  12345678 path/to/file.npy"
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            size_bytes = int(parts[2])
            # Reconstruct the path (may contain spaces, though unlikely)
            relative_path = " ".join(parts[3:])
            if relative_path.endswith(".npy"):
                # Build full S3 path
                # The base path we provided ends with the bucket prefix,
                # and the ls output gives us the relative path from bucket root
                bucket_and_prefix = s3_path.replace("s3://", "")
                bucket = bucket_and_prefix.split("/")[0]
                full_path = f"s3://{bucket}/{relative_path}"
                files.append((full_path, size_bytes))

    return files


def analyze_dataset() -> Dict[str, TopicStats]:
    """Analyze the full dataset and return statistics per topic."""
    print(f"Analyzing dataset at {S3_BASE_PATH}...")
    print("This may take a moment as we query S3 for file sizes.\n")

    topic_stats: Dict[str, TopicStats] = {}

    for topic in TOPICS:
        topic_path = f"{S3_BASE_PATH}/{topic}/"
        files = list_s3_files(topic_path)

        stats = TopicStats(name=topic)
        for path, size_bytes in sorted(files):  # Sort by path for reproducibility
            stats.files.append(FileInfo(path=path, size_bytes=size_bytes))

        topic_stats[topic] = stats
        print(f"  {topic}: {stats.num_files} files, {format_tokens(stats.total_tokens)} tokens")

    return topic_stats


def print_analysis(topic_stats: Dict[str, TopicStats]) -> None:
    """Print a formatted analysis table."""
    total_bytes = sum(s.total_bytes for s in topic_stats.values())
    total_tokens = sum(s.total_tokens for s in topic_stats.values())
    total_files = sum(s.num_files for s in topic_stats.values())

    print("\n" + "=" * 85)
    print(f"{'Topic':<35} {'Files':>8} {'Size':>12} {'Tokens':>12} {'Pct':>8}")
    print("-" * 85)

    # Sort by tokens descending
    for topic in sorted(
        topic_stats.keys(), key=lambda t: topic_stats[t].total_tokens, reverse=True
    ):
        stats = topic_stats[topic]
        pct = (stats.total_tokens / total_tokens * 100) if total_tokens > 0 else 0
        print(
            f"{stats.name:<35} {stats.num_files:>8} {format_bytes(stats.total_bytes):>12} "
            f"{format_tokens(stats.total_tokens):>12} {pct:>7.2f}%"
        )

    print("-" * 85)
    print(
        f"{'TOTAL':<35} {total_files:>8} {format_bytes(total_bytes):>12} "
        f"{format_tokens(total_tokens):>12} {'100.00%':>8}"
    )
    print("=" * 85)


def subsample_topics(
    topic_stats: Dict[str, TopicStats],
    target_tokens: int,
    overshoot_factor: float = 1.5,
) -> Tuple[Dict[str, List[FileInfo]], List[str]]:
    """
    Subsample files from each topic using strict proportional selection.

    Returns:
        - Dictionary mapping topic names to selected files
        - List of excluded topic names
    """
    total_tokens = sum(s.total_tokens for s in topic_stats.values())

    selected_files: Dict[str, List[FileInfo]] = {}
    excluded_topics: List[str] = []

    for topic, stats in topic_stats.items():
        if stats.num_files == 0:
            excluded_topics.append(topic)
            continue

        # Calculate proportional target for this topic
        topic_proportion = stats.total_tokens / total_tokens
        topic_target = int(target_tokens * topic_proportion)
        topic_max = int(topic_target * overshoot_factor)

        # Check if first file exceeds the allowed overshoot
        first_file_tokens = stats.files[0].num_tokens
        if first_file_tokens > topic_max:
            excluded_topics.append(topic)
            continue

        # Select files sequentially until we reach the target
        # (allowing up to overshoot_factor overshoot)
        selected: List[FileInfo] = []
        current_tokens = 0

        for file_info in stats.files:
            if current_tokens >= topic_target:
                break
            # Check if adding this file would exceed overshoot limit
            if current_tokens + file_info.num_tokens > topic_max and current_tokens > 0:
                break
            selected.append(file_info)
            current_tokens += file_info.num_tokens

        if selected:
            selected_files[topic] = selected

    return selected_files, excluded_topics


def generate_yaml(
    selected_files: Dict[str, List[FileInfo]],
    excluded_topics: List[str],
    target_tokens: int,
    output_path: Optional[str] = None,
    web_weight: float = 0.9,
    code_weight: float = 0.1,
) -> str:
    """Generate YAML config for the selected files."""
    lines = ["mix:"]

    # Web data source
    lines.append("  - name: web")
    lines.append(f"    weight: {web_weight}")
    lines.append("    paths:")

    # Add files grouped by topic with comments
    for topic in sorted(selected_files.keys()):
        files = selected_files[topic]
        topic_tokens = sum(f.num_tokens for f in files)
        lines.append(f"      # {topic} ({len(files)} files, {format_tokens(topic_tokens)} tokens)")
        for file_info in files:
            lines.append(f"      - {file_info.path}")

    lines.append("    repetition_factor: -1.0")

    # Code fresh source
    lines.append("  - name: code_fresh")
    lines.append(f"    weight: {code_weight}")
    lines.append("    paths:")
    for path in CODE_FRESH_PATHS:
        lines.append(f"      - {path}")
    lines.append("    repetition_factor: -1.0")

    yaml_content = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).write_text(yaml_content)
        print(f"YAML config written to: {output_path}")

    return yaml_content


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the analyze command."""
    topic_stats = analyze_dataset()
    print_analysis(topic_stats)


def cmd_generate(args: argparse.Namespace) -> None:
    """Run the generate command."""
    target_tokens = parse_token_count(args.target_tokens)
    print(f"Target tokens: {format_tokens(target_tokens)}")

    # Analyze dataset
    topic_stats = analyze_dataset()
    print_analysis(topic_stats)

    # Subsample
    print(f"\nSubsampling with target {format_tokens(target_tokens)} tokens...")
    selected_files, excluded_topics = subsample_topics(
        topic_stats, target_tokens, overshoot_factor=args.overshoot_factor
    )

    # Print summary
    total_selected_files = sum(len(files) for files in selected_files.values())
    total_selected_tokens = sum(
        sum(f.num_tokens for f in files) for files in selected_files.values()
    )

    print("\nSubsampling Results:")
    print(f"  Topics included: {len(selected_files)}/{len(TOPICS)}")
    print(f"  Topics excluded: {excluded_topics if excluded_topics else 'None'}")
    print(f"  Files selected: {total_selected_files}")
    print(f"  Tokens selected: {format_tokens(total_selected_tokens)}")
    print(f"  Target tokens: {format_tokens(target_tokens)}")
    print(f"  Ratio: {total_selected_tokens / target_tokens:.2%} of target")

    # Print per-topic breakdown
    print("\nPer-topic selection:")
    print("-" * 70)
    for topic in sorted(selected_files.keys()):
        files = selected_files[topic]
        topic_tokens = sum(f.num_tokens for f in files)
        original_tokens = topic_stats[topic].total_tokens
        print(
            f"  {topic:<35} {len(files):>3} files, "
            f"{format_tokens(topic_tokens):>10} tokens "
            f"({topic_tokens / original_tokens:.1%} of topic)"
        )

    # Generate YAML
    print()
    yaml_content = generate_yaml(
        selected_files,
        excluded_topics,
        target_tokens,
        output_path=args.output,
        web_weight=args.web_weight,
        code_weight=args.code_weight,
    )

    if not args.output:
        print("\nGenerated YAML (use --output to save to file):")
        print("-" * 70)
        # Print first 50 lines
        lines = yaml_content.split("\n")
        for line in lines[:50]:
            print(line)
        if len(lines) > 50:
            print(f"... ({len(lines) - 50} more lines)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dolma Dataset Subsampling Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze the dataset and print token distribution"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a subsampled YAML config")
    generate_parser.add_argument(
        "--target-tokens",
        type=str,
        default="300B",
        help="Target number of tokens (e.g., '300B', '500M'). Default: 300B",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for the YAML config file",
    )
    generate_parser.add_argument(
        "--overshoot-factor",
        type=float,
        default=1.5,
        help="Maximum overshoot factor for topic targets. Default: 1.5",
    )
    generate_parser.add_argument(
        "--web-weight",
        type=float,
        default=0.9,
        help="Weight for web data in the mix. Default: 0.9",
    )
    generate_parser.add_argument(
        "--code-weight",
        type=float,
        default=0.1,
        help="Weight for code data in the mix. Default: 0.1",
    )
    generate_parser.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
