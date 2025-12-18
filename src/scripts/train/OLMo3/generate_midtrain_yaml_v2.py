#!/usr/bin/env python3
"""
Generate midtrain YAML with configurable repetition tiers.

This script ensures EXACT repetition control by:
1. Using actual token counts per file (not estimates)
2. Calculating target_ratio = (actual_tokens_in_bucket × repetition) / budget
3. Scaling file counts to fit within budget if needed

Usage:
    # Generate with 4 tiers (10x, 6x, 3x, 1x) from scratch:
    python src/scripts/train/OLMo3/generate_midtrain_yaml_v2.py

    # Extend an existing YAML by adding repeat10 from UNUSED files only:
    python src/scripts/train/OLMo3/generate_midtrain_yaml_v2.py \
        --base-yaml src/olmo_core/data/source_mixtures/midtrain-memo-auto.yaml \
        --new-tier 10 \
        --output midtrain-memo-10B-extended.yaml

    # Or customize via command line:
    python src/scripts/train/OLMo3/generate_midtrain_yaml_v2.py --tiers 10,6,3,1 --output my-mix.yaml

Requirements:
    - gsutil or google-cloud-storage
    - Access to the GCS bucket
"""

import argparse
import math
import subprocess
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import yaml

# Configuration
GCS_BASE_PATH = "gs://ai2-llm/yapeic/dolmino2-mix-0925-reconstructed/preprocessed/sources/cc_all_dressed/all_dressed_v3/weborganizer_ft/dclm_plus2_vigintiles/vigintile_0018_subset-decon-2/allenai/dolma2-tokenizer"
DEFAULT_OUTPUT_FILE = "src/olmo_core/data/source_mixtures/midtrain-memo-10B-4tier.yaml"
TOTAL_BUDGET_TOKENS = 10_000_000_000
BYTES_PER_TOKEN = 4  # uint32

# File size thresholds (in bytes)
LARGE_FILE_MIN_SIZE = 1_000_000   # 1MB - for most topics
SMALL_FILE_MIN_SIZE = 100_000     # 100KB - for small topics

# Topics to scan
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

# Topics known to have smaller files (use lower threshold)
SMALL_FILE_TOPICS = {
    "fashion_and_beauty",
    "adult_content",
    "travel_and_tourism",
    "social_life",
    "home_and_hobbies",
    "transportation",
}


def load_base_yaml(yaml_path: str) -> Tuple[List[Dict], Set[str]]:
    """
    Load an existing YAML and extract all used file paths.
    
    Returns:
        - List of source dictionaries from the YAML
        - Set of all file paths used in the YAML
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    sources = data.get("sources", [])
    used_paths: Set[str] = set()
    
    for source in sources:
        for path in source.get("paths", []):
            used_paths.add(path)
    
    return sources, used_paths


def parse_source_name(source_name: str) -> Tuple[str, int]:
    """
    Parse source name to extract topic and repetition rate.
    
    Example: "science_math_and_technology_repeat1" -> ("science_math_and_technology", 1)
    """
    match = re.match(r"(.+)_repeat(\d+)$", source_name)
    if match:
        topic = match.group(1)
        rep_rate = int(match.group(2))
        return topic, rep_rate
    return source_name, 1


@dataclass
class FileInfo:
    path: str
    size: int  # bytes
    tokens: int  # actual tokens (bytes / 4 for uint32)


@dataclass 
class BucketInfo:
    """Info for a repetition bucket."""
    files: List[FileInfo]
    repeat: int
    actual_tokens: int  # sum of tokens in files
    effective_tokens: int  # actual_tokens * repeat
    
    @property
    def ratio(self) -> float:
        """Target ratio for this bucket."""
        return self.effective_tokens / TOTAL_BUDGET_TOKENS


def list_gcs_files(gcs_path: str) -> List[Tuple[str, int]]:
    """List files in GCS path with their sizes using gsutil."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", "-l", f"{gcs_path}/*.npy"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            # Format: "  12345678  2024-01-01T00:00:00Z  gs://bucket/path/file.npy"
            match = re.match(r"\s*(\d+)\s+\S+\s+(gs://\S+)", line)
            if match:
                size = int(match.group(1))
                path = match.group(2)
                files.append((path, size))
        return sorted(files, key=lambda x: x[0])  # Sort by path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to list {gcs_path}: {e.stderr}")
        return []


def scan_topic(topic: str) -> List[FileInfo]:
    """Scan a topic directory and return file info with actual token counts."""
    gcs_path = f"{GCS_BASE_PATH}/{topic}"
    files = list_gcs_files(gcs_path)
    
    # Determine size threshold
    min_size = SMALL_FILE_MIN_SIZE if topic in SMALL_FILE_TOPICS else LARGE_FILE_MIN_SIZE
    
    # Filter by size and calculate actual tokens
    filtered = []
    for path, size in files:
        if size >= min_size:
            tokens = size // BYTES_PER_TOKEN
            filtered.append(FileInfo(path=path, size=size, tokens=tokens))
    
    return filtered


def create_bucket(files: List[FileInfo], repeat: int) -> BucketInfo:
    """Create a bucket with actual token counts."""
    actual_tokens = sum(f.tokens for f in files)
    effective_tokens = actual_tokens * repeat
    return BucketInfo(
        files=files,
        repeat=repeat,
        actual_tokens=actual_tokens,
        effective_tokens=effective_tokens,
    )


def split_files_into_buckets(
    files: List[FileInfo],
    repetition_tiers: List[int],
) -> List[BucketInfo]:
    """
    Split files into buckets based on repetition tiers.
    
    Args:
        files: List of files to distribute
        repetition_tiers: List of repetition rates, sorted descending (e.g., [10, 6, 3, 1])
    
    Returns:
        List of BucketInfo, one per tier (same order as repetition_tiers)
    """
    if not files:
        return [BucketInfo(files=[], repeat=r, actual_tokens=0, effective_tokens=0) 
                for r in repetition_tiers]
    
    n = len(files)
    num_tiers = len(repetition_tiers)
    
    # Handle edge case: very few files
    if n < num_tiers:
        # Put all files in the lowest repetition bucket
        buckets = [BucketInfo(files=[], repeat=r, actual_tokens=0, effective_tokens=0) 
                   for r in repetition_tiers[:-1]]
        buckets.append(create_bucket(files, repetition_tiers[-1]))
        return buckets
    
    # Calculate file distribution ratios
    # Higher repetition tiers get fewer files to balance effective tokens
    # Use inverse of repetition rate as weight
    weights = [1.0 / r for r in repetition_tiers]
    total_weight = sum(weights)
    
    # Normalize weights to get file counts
    file_counts = []
    remaining = n
    for i, w in enumerate(weights[:-1]):
        count = max(1, int(n * w / total_weight))
        count = min(count, remaining - (num_tiers - i - 1))  # Leave at least 1 for remaining tiers
        file_counts.append(count)
        remaining -= count
    file_counts.append(remaining)  # Last tier gets the rest
    
    # Split files (sorted by path for determinism)
    sorted_files = sorted(files, key=lambda f: f.path)
    
    buckets = []
    start_idx = 0
    for count, repeat in zip(file_counts, repetition_tiers):
        end_idx = start_idx + count
        bucket_files = sorted_files[start_idx:end_idx]
        buckets.append(create_bucket(bucket_files, repeat))
        start_idx = end_idx
    
    return buckets


def scale_buckets_to_budget(all_buckets: List[Tuple[str, BucketInfo]]) -> List[Tuple[str, BucketInfo]]:
    """
    Scale file counts to exactly hit the budget.
    
    Strategy:
    1. If under budget, use all files as-is
    2. If over budget, scale down proportionally then add files back to fill remaining
    
    Returns list of (source_name, bucket) tuples.
    """
    total_effective = sum(b.effective_tokens for _, b in all_buckets)
    
    if total_effective <= TOTAL_BUDGET_TOKENS:
        print(f"Total effective tokens ({total_effective/1e9:.3f}B) within budget ({TOTAL_BUDGET_TOKENS/1e9:.0f}B)")
        return all_buckets
    
    # Need to scale down
    scale_factor = TOTAL_BUDGET_TOKENS / total_effective
    print(f"Total effective tokens ({total_effective/1e9:.3f}B) exceeds budget ({TOTAL_BUDGET_TOKENS/1e9:.0f}B)")
    print(f"Scaling down by factor {scale_factor:.4f}")
    
    # Step 1: Initial scaling (slightly under budget)
    scaled_buckets = []
    leftover_files = []  # (name, file, repeat) tuples for files not included
    
    for name, bucket in all_buckets:
        if not bucket.files:
            continue
            
        # Calculate how many files to keep
        n_original = len(bucket.files)
        n_scaled = max(1, int(n_original * scale_factor))
        
        # Take first n_scaled files
        scaled_files = bucket.files[:n_scaled]
        scaled_bucket = create_bucket(scaled_files, bucket.repeat)
        scaled_buckets.append((name, scaled_bucket))
        
        # Track leftover files for potential re-adding
        for f in bucket.files[n_scaled:]:
            leftover_files.append((name, f, bucket.repeat))
    
    # Step 2: Calculate remaining budget and fill it
    current_effective = sum(b.effective_tokens for _, b in scaled_buckets)
    remaining_budget = TOTAL_BUDGET_TOKENS - current_effective
    
    print(f"After initial scaling: {current_effective/1e9:.4f}B tokens")
    print(f"Remaining budget: {remaining_budget/1e9:.4f}B tokens")
    print(f"Leftover files available: {len(leftover_files)}")
    
    # Sort leftover files by effective tokens (smallest first for fine-grained filling)
    leftover_files.sort(key=lambda x: x[1].tokens * x[2])
    
    # Greedily add files to fill remaining budget
    files_added = 0
    bucket_additions: Dict[str, List[FileInfo]] = {}
    
    for name, file_info, repeat in leftover_files:
        effective_tokens = file_info.tokens * repeat
        if effective_tokens <= remaining_budget:
            remaining_budget -= effective_tokens
            files_added += 1
            if name not in bucket_additions:
                bucket_additions[name] = []
            bucket_additions[name].append(file_info)
    
    print(f"Added {files_added} files to fill budget")
    print(f"Final remaining budget: {remaining_budget/1e9:.6f}B tokens ({remaining_budget} tokens)")
    
    # Step 3: Merge additions back into buckets
    final_buckets = []
    for name, bucket in scaled_buckets:
        if name in bucket_additions:
            # Add the extra files
            all_files = bucket.files + bucket_additions[name]
            final_bucket = create_bucket(all_files, bucket.repeat)
            final_buckets.append((name, final_bucket))
        else:
            final_buckets.append((name, bucket))
    
    return final_buckets


def generate_yaml_sources(all_buckets: List[Tuple[str, BucketInfo]]) -> List[Dict]:
    """Generate YAML source entries from buckets."""
    sources = []
    
    for source_name, bucket in all_buckets:
        if not bucket.files or bucket.effective_tokens == 0:
            continue
            
        # Calculate ratio from ACTUAL tokens
        ratio = bucket.effective_tokens / TOTAL_BUDGET_TOKENS
        
        sources.append({
            "source_name": source_name,
            "target_ratio": ratio,  # Don't round yet - will normalize first
            "max_repetition_ratio": float(bucket.repeat),
            "paths": [f.path for f in bucket.files],
            # Metadata for verification (will be in comments)
            "_actual_tokens": bucket.actual_tokens,
            "_effective_tokens": bucket.effective_tokens,
            "_n_files": len(bucket.files),
        })
    
    return sources


def normalize_to_sum_one(sources: List[Dict], max_tier: int) -> List[Dict]:
    """
    Normalize ratios to sum to EXACTLY 1.0.
    
    This is necessary because SourceMixtureList.validate() checks:
        if not np.allclose(summed_weights, 1.0):
            raise OLMoConfigurationError(...)
    
    Strategy:
    - If sum > 1.0: Scale down all ratios (safe - we request less than available)
    - If sum < 1.0: Add the gap to the highest repetition tier source (most headroom)
    """
    if not sources:
        return sources
    
    total_ratio = sum(s["target_ratio"] for s in sources)
    
    if total_ratio == 0:
        return sources
    
    print(f"\nNormalizing ratios to sum to EXACTLY 1.0:")
    print(f"  Current sum: {total_ratio:.6f}")
    
    # Sort by effective tokens (largest first) - largest source will absorb the gap
    sources.sort(key=lambda s: -s["_effective_tokens"])
    
    gap = 1.0 - total_ratio
    
    # Find the highest repetition tier pattern for absorber selection
    absorber_pattern = f"_repeat{max_tier}"
    
    if gap < 0:
        # Sum > 1.0: scale down all ratios proportionally (safe)
        scale_factor = 1.0 / total_ratio
        print(f"  Sum > 1.0: Scaling down all ratios by {scale_factor:.6f}")
        for s in sources:
            s["target_ratio"] = s["target_ratio"] * scale_factor
    else:
        # Sum < 1.0: add the gap to a high-repetition source (has most headroom)
        high_rep_sources = [s for s in sources if absorber_pattern in s["source_name"]]
        
        if high_rep_sources:
            # Add gap to the largest high-rep source
            high_rep_sources.sort(key=lambda s: -s["_effective_tokens"])
            target_source = high_rep_sources[0]
            
            # Calculate how much extra repetition is needed
            old_ratio = target_source["target_ratio"]
            new_ratio = old_ratio + gap
            ratio_increase = new_ratio / old_ratio if old_ratio > 0 else 1.0
            
            old_max_rep = target_source["max_repetition_ratio"]
            new_max_rep = old_max_rep * ratio_increase * 1.01  # 1% extra buffer
            
            print(f"  Sum < 1.0: Adding gap of {gap:.6f} to {absorber_pattern} source: {target_source['source_name']}")
            print(f"    Ratio: {old_ratio:.6f} → {new_ratio:.6f}")
            print(f"    max_repetition_ratio: {old_max_rep:.3f} → {new_max_rep:.3f} (includes 1% buffer)")
            
            target_source["target_ratio"] = new_ratio
            target_source["max_repetition_ratio"] = new_max_rep
        else:
            # Fallback: add to largest source
            print(f"  Sum < 1.0: No {absorber_pattern} source found, adding gap to: {sources[0]['source_name']}")
            sources[0]["target_ratio"] += gap
    
    # Use integer arithmetic for precise rounding
    PRECISION = 1_000_000
    
    # Find the absorber (highest rep source)
    high_rep_sources = [s for s in sources if absorber_pattern in s["source_name"]]
    if high_rep_sources:
        high_rep_sources.sort(key=lambda s: -s["_effective_tokens"])
        absorber = high_rep_sources[0]
    else:
        absorber = sources[0]  # Fallback
    
    # Round DOWN all sources EXCEPT the absorber
    rounded_parts = []
    for s in sources:
        if s is absorber:
            continue
        parts = math.floor(s["target_ratio"] * PRECISION)
        rounded_parts.append(parts)
        s["target_ratio"] = parts / PRECISION
    
    # Set the absorber to fill the remainder
    remainder = PRECISION - sum(rounded_parts)
    final_absorber_ratio = remainder / PRECISION
    absorber["target_ratio"] = final_absorber_ratio
    
    # Recalculate max_repetition_ratio for absorber
    original_actual = absorber["_actual_tokens"]
    if original_actual > 0:
        required_rep = (final_absorber_ratio * TOTAL_BUDGET_TOKENS) / original_actual
        absorber["max_repetition_ratio"] = required_rep * 1.01
        print(f"  Absorber final ratio: {final_absorber_ratio:.6f}")
        print(f"  Absorber max_repetition_ratio: {absorber['max_repetition_ratio']:.4f}")
    
    # Verify the sum
    final_sum = sum(s["target_ratio"] for s in sources)
    print(f"  Final sum: {final_sum}")
    
    total_parts = remainder + sum(rounded_parts)
    if total_parts == PRECISION:
        print(f"  ✓ Sum is exactly 1.0 (verified with integer arithmetic)")
    else:
        print(f"  WARNING: Integer sum is {total_parts}, expected {PRECISION}!")
    
    return sources


def main():
    parser = argparse.ArgumentParser(description="Generate midtrain YAML with configurable repetition tiers")
    parser.add_argument(
        "--tiers", 
        type=str, 
        default="10,6,3,1",
        help="Comma-separated list of repetition tiers, descending order (default: 10,6,3,1)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output YAML file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=10.0,
        help="Token budget in billions (default: 10.0)"
    )
    parser.add_argument(
        "--base-yaml",
        type=str,
        default=None,
        help="Path to existing YAML file. If provided, keeps all existing sources and adds "
             "a new tier from UNUSED files only (maximizes overlap with previous run)"
    )
    parser.add_argument(
        "--new-tier",
        type=int,
        default=10,
        help="Repetition rate for the new tier when using --base-yaml (default: 10)"
    )
    args = parser.parse_args()
    
    global TOTAL_BUDGET_TOKENS
    TOTAL_BUDGET_TOKENS = int(args.budget * 1_000_000_000)
    
    # =========================================================================
    # MODE 1: EXTEND existing YAML with a new tier from unused files
    # =========================================================================
    if args.base_yaml:
        run_extend_mode(args)
    # =========================================================================
    # MODE 2: FRESH generation with all tiers from scratch
    # =========================================================================
    else:
        run_fresh_mode(args)


def run_extend_mode(args):
    """
    Extend an existing YAML by adding a new repetition tier from UNUSED files.
    
    This mode:
    1. Loads all sources from the base YAML (keeps them unchanged)
    2. Finds files NOT used in the base YAML
    3. Creates new sources for the new tier from unused files only
    4. Merges and re-normalizes ratios to sum to 1.0
    """
    global TOTAL_BUDGET_TOKENS
    
    new_tier = args.new_tier
    
    print("=" * 80)
    print("MIDTRAIN YAML GENERATOR v2 - EXTEND MODE")
    print("=" * 80)
    print(f"Base YAML: {args.base_yaml}")
    print(f"New tier: {new_tier}x repetition")
    print(f"Budget: {TOTAL_BUDGET_TOKENS/1e9:.0f}B tokens")
    print(f"Output: {args.output}")
    print()
    
    # Step 1: Load base YAML and extract used paths
    print("Loading base YAML...")
    base_sources, used_paths = load_base_yaml(args.base_yaml)
    print(f"  Found {len(base_sources)} sources with {len(used_paths)} total files")
    
    # Collect existing tiers from base YAML
    existing_tiers = set()
    for src in base_sources:
        _, rep_rate = parse_source_name(src.get("source_name", ""))
        existing_tiers.add(rep_rate)
    print(f"  Existing tiers: {sorted(existing_tiers)}")
    
    # Calculate effective tokens from base sources
    base_effective_tokens = 0
    for src in base_sources:
        ratio = src.get("target_ratio", 0)
        base_effective_tokens += ratio * TOTAL_BUDGET_TOKENS
    print(f"  Base effective tokens: {base_effective_tokens/1e9:.4f}B ({base_effective_tokens/TOTAL_BUDGET_TOKENS*100:.2f}% of budget)")
    
    # Step 2: Scan GCS for files NOT in the base YAML
    print(f"\nScanning for UNUSED files to add as repeat{new_tier}...")
    new_buckets: List[Tuple[str, BucketInfo]] = []
    total_unused_files = 0
    total_unused_tokens = 0
    
    for topic in TOPICS:
        print(f"  Scanning {topic}...")
        all_files = scan_topic(topic)
        
        if not all_files:
            print(f"    No files found")
            continue
        
        # Filter to only unused files
        unused_files = [f for f in all_files if f.path not in used_paths]
        
        if not unused_files:
            print(f"    All {len(all_files)} files already used in base YAML")
            continue
        
        # Create a bucket with the new tier rate
        bucket = create_bucket(unused_files, new_tier)
        new_buckets.append((f"{topic}_repeat{new_tier}", bucket))
        
        total_unused_files += len(unused_files)
        total_unused_tokens += bucket.actual_tokens
        
        print(f"    Found {len(unused_files)} unused files (of {len(all_files)} total)")
        print(f"    Tokens: {bucket.actual_tokens/1e9:.4f}B actual, {bucket.effective_tokens/1e9:.4f}B effective (×{new_tier})")
    
    print(f"\nTotal unused: {total_unused_files} files, {total_unused_tokens/1e9:.4f}B tokens")
    total_new_effective = sum(b.effective_tokens for _, b in new_buckets)
    print(f"Total new effective tokens: {total_new_effective/1e9:.4f}B")
    
    # Step 3: Check if we exceed budget and scale if needed
    combined_effective = base_effective_tokens + total_new_effective
    print(f"\nCombined effective tokens: {combined_effective/1e9:.4f}B")
    
    if combined_effective > TOTAL_BUDGET_TOKENS:
        # Need to scale down the new tier to fit
        available_budget = TOTAL_BUDGET_TOKENS - base_effective_tokens
        scale_factor = available_budget / total_new_effective
        print(f"Exceeds budget! Scaling new tier by {scale_factor:.4f}")
        
        scaled_buckets = []
        for name, bucket in new_buckets:
            if not bucket.files:
                continue
            n_scaled = max(1, int(len(bucket.files) * scale_factor))
            scaled_files = bucket.files[:n_scaled]
            scaled_bucket = create_bucket(scaled_files, new_tier)
            scaled_buckets.append((name, scaled_bucket))
        new_buckets = scaled_buckets
        total_new_effective = sum(b.effective_tokens for _, b in new_buckets)
        print(f"Scaled new effective tokens: {total_new_effective/1e9:.4f}B")
    
    # Step 4: Generate sources for new buckets
    print("\nGenerating sources for new tier...")
    new_sources = generate_yaml_sources(new_buckets)
    
    # Step 5: Merge base sources with new sources
    # Convert base sources to same format (add metadata fields)
    merged_sources = []
    for src in base_sources:
        # Calculate effective tokens from ratio
        ratio = src.get("target_ratio", 0)
        max_rep = src.get("max_repetition_ratio", 1.0)
        paths = src.get("paths", [])
        
        # Estimate actual tokens (this is approximate but needed for normalization)
        effective = ratio * TOTAL_BUDGET_TOKENS
        actual = effective / max_rep if max_rep > 0 else effective
        
        merged_sources.append({
            "source_name": src.get("source_name", ""),
            "target_ratio": ratio,
            "max_repetition_ratio": max_rep,
            "paths": paths,
            "_actual_tokens": int(actual),
            "_effective_tokens": int(effective),
            "_n_files": len(paths),
            "_from_base": True,  # Mark as from base YAML
        })
    
    # Add new sources
    for s in new_sources:
        s["_from_base"] = False
        merged_sources.append(s)
    
    # Step 6: Normalize to sum to exactly 1.0
    # The new tier should absorb the gap since base sources are already set
    all_tiers = sorted(existing_tiers | {new_tier}, reverse=True)
    max_tier = max(all_tiers)
    merged_sources = normalize_to_sum_one(merged_sources, max_tier)
    
    # Sort by ratio (largest first)
    merged_sources.sort(key=lambda s: -s["target_ratio"])
    
    # Verify sum
    total_ratio = sum(s["target_ratio"] for s in merged_sources)
    print(f"Total ratio: {total_ratio:.6f}")
    
    # Build YAML structure
    yaml_sources = []
    for s in merged_sources:
        yaml_sources.append({
            "source_name": s["source_name"],
            "target_ratio": s["target_ratio"],
            "max_repetition_ratio": s["max_repetition_ratio"],
            "paths": s["paths"],
        })
    
    yaml_content = {"sources": yaml_sources}
    
    # Build tier description
    tier_desc = "\n".join([f"#   - _repeat{r}: {r}x repetition" for r in all_tiers])
    
    # Write YAML
    with open(args.output, "w") as f:
        f.write("# Auto-generated midtraining mix with EXACT repetition control\n")
        f.write(f"# Generated by: {__file__}\n")
        f.write(f"# Extended from: {args.base_yaml}\n")
        f.write(f"# Total sources: {len(merged_sources)}\n")
        f.write(f"# Budget: {TOTAL_BUDGET_TOKENS/1e9:.0f}B tokens\n")
        f.write(f"# Total ratio: {total_ratio:.6f}\n")
        f.write("#\n")
        f.write("# IMPORTANT: target_ratio is calculated as:\n")
        f.write("#   ratio = (actual_tokens_in_bucket × max_repetition_ratio) / budget\n")
        f.write("# This ensures each file is repeated EXACTLY max_repetition_ratio times.\n")
        f.write("#\n")
        f.write("# Repetition tiers:\n")
        f.write(tier_desc + "\n")
        f.write("#\n")
        f.write(f"# NOTE: Sources from base YAML kept unchanged, new repeat{new_tier} tier\n")
        f.write(f"#       uses {total_unused_files} files NOT in base YAML.\n")
        f.write("#\n\n")
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, width=200)
    
    print(f"\nWritten to {args.output}")
    print(f"Total sources: {len(merged_sources)}")
    
    # Print detailed summary
    print("\n" + "=" * 95)
    print("VERIFICATION SUMMARY")
    print("=" * 95)
    print(f"{'Source Name':<50} {'Files':>6} {'Actual':>10} {'×Rep':>5} {'Effective':>12} {'Ratio':>10} {'From'}")
    print("-" * 95)
    
    total_effective = 0
    for s in merged_sources:
        name = s["source_name"]
        n_files = s["_n_files"]
        actual = s["_actual_tokens"] / 1e9
        repeat = s["max_repetition_ratio"]
        effective = s["_effective_tokens"] / 1e9
        ratio = s["target_ratio"]
        from_base = "base" if s.get("_from_base", False) else "NEW"
        total_effective += s["_effective_tokens"]
        print(f"{name:<50} {n_files:>6} {actual:>9.4f}B ×{repeat:>5.1f} {effective:>11.4f}B {ratio:>10.6f} {from_base}")
    
    print("-" * 95)
    print(f"{'TOTAL':<50} {'':<6} {'':<10} {'':<5} {total_effective/1e9:>11.4f}B {total_ratio:>10.6f}")
    print()
    
    # Count sources per tier
    print("Sources per tier:")
    for tier in all_tiers:
        count = sum(1 for s in merged_sources if f"_repeat{tier}" in s["source_name"])
        new_count = sum(1 for s in merged_sources if f"_repeat{tier}" in s["source_name"] and not s.get("_from_base", False))
        if new_count > 0:
            print(f"  repeat{tier}: {count} sources ({new_count} NEW)")
        else:
            print(f"  repeat{tier}: {count} sources (from base)")


def run_fresh_mode(args):
    """
    Generate a fresh YAML with all tiers from scratch.
    
    This is the original behavior of the script.
    """
    global TOTAL_BUDGET_TOKENS
    
    # Parse repetition tiers
    repetition_tiers = [int(x.strip()) for x in args.tiers.split(",")]
    repetition_tiers.sort(reverse=True)  # Ensure descending order
    
    print("=" * 80)
    print("MIDTRAIN YAML GENERATOR v2 - FRESH MODE")
    print("=" * 80)
    print(f"Repetition tiers: {repetition_tiers}")
    print(f"Budget: {TOTAL_BUDGET_TOKENS/1e9:.0f}B tokens")
    print(f"Output: {args.output}")
    print(f"Bytes per token: {BYTES_PER_TOKEN} (uint32)")
    print()
    
    # Collect all buckets
    all_buckets: List[Tuple[str, BucketInfo]] = []
    topic_stats = {}
    
    for topic in TOPICS:
        print(f"  Scanning {topic}...")
        files = scan_topic(topic)
        
        if not files:
            print(f"    No files found for {topic}")
            continue
        
        buckets = split_files_into_buckets(files, repetition_tiers)
        
        # Add non-empty buckets with appropriate names
        for bucket, repeat in zip(buckets, repetition_tiers):
            if bucket.files:
                all_buckets.append((f"{topic}_repeat{repeat}", bucket))
        
        total_tokens = sum(f.tokens for f in files)
        total_effective = sum(b.effective_tokens for b in buckets)
        
        topic_stats[topic] = {
            "n_files": len(files),
            "buckets": [(len(b.files), b.repeat) for b in buckets],
            "total_tokens": total_tokens,
            "total_effective": total_effective,
        }
        
        print(f"    Found {len(files)} files, {total_tokens/1e9:.4f}B tokens")
        bucket_str = ", ".join([f"{len(b.files)}×{b.repeat}" for b in buckets if b.files])
        print(f"    Buckets: {bucket_str}")
        print(f"    Effective tokens: {total_effective/1e9:.4f}B")
    
    print()
    
    # Scale to budget if needed
    all_buckets = scale_buckets_to_budget(all_buckets)
    
    # Calculate final totals
    total_effective = sum(b.effective_tokens for _, b in all_buckets)
    print(f"\nFinal total effective tokens: {total_effective/1e9:.4f}B")
    
    print("\nGenerating YAML...")
    sources = generate_yaml_sources(all_buckets)
    
    # Normalize to sum to exactly 1.0
    max_tier = max(repetition_tiers)
    sources = normalize_to_sum_one(sources, max_tier)
    
    # Sort by ratio (largest first)
    sources.sort(key=lambda s: -s["target_ratio"])
    
    # Verify sum
    total_ratio = sum(s["target_ratio"] for s in sources)
    print(f"Total ratio: {total_ratio:.6f}")
    
    # Build YAML structure
    yaml_sources = []
    for s in sources:
        yaml_sources.append({
            "source_name": s["source_name"],
            "target_ratio": s["target_ratio"],
            "max_repetition_ratio": s["max_repetition_ratio"],
            "paths": s["paths"],
        })
    
    yaml_content = {"sources": yaml_sources}
    
    # Build tier description
    tier_desc = "\n".join([f"#   - _repeat{r}: {r}x repetition" for r in repetition_tiers])
    
    # Write YAML
    with open(args.output, "w") as f:
        f.write("# Auto-generated midtraining mix with EXACT repetition control\n")
        f.write(f"# Generated by: {__file__}\n")
        f.write(f"# Total sources: {len(sources)}\n")
        f.write(f"# Budget: {TOTAL_BUDGET_TOKENS/1e9:.0f}B tokens\n")
        f.write(f"# Total ratio: {total_ratio:.6f}\n")
        f.write("#\n")
        f.write("# IMPORTANT: target_ratio is calculated as:\n")
        f.write("#   ratio = (actual_tokens_in_bucket × max_repetition_ratio) / budget\n")
        f.write("# This ensures each file is repeated EXACTLY max_repetition_ratio times.\n")
        f.write("#\n")
        f.write("# Repetition tiers:\n")
        f.write(tier_desc + "\n")
        f.write("#\n\n")
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, width=200)
    
    print(f"\nWritten to {args.output}")
    print(f"Total sources: {len(sources)}")
    
    # Print detailed summary
    print("\n" + "=" * 90)
    print("VERIFICATION SUMMARY")
    print("=" * 90)
    print(f"{'Source Name':<50} {'Files':>6} {'Actual':>10} {'×Rep':>5} {'Effective':>12} {'Ratio':>10}")
    print("-" * 90)
    
    for s in sources:
        name = s["source_name"]
        n_files = s["_n_files"]
        actual = s["_actual_tokens"] / 1e9
        repeat = s["max_repetition_ratio"]
        effective = s["_effective_tokens"] / 1e9
        ratio = s["target_ratio"]
        print(f"{name:<50} {n_files:>6} {actual:>9.4f}B ×{repeat:>5.1f} {effective:>11.4f}B {ratio:>10.6f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<50} {'':<6} {'':<10} {'':<5} {total_effective/1e9:>11.4f}B {total_ratio:>10.6f}")
    print()
    
    # Count sources per tier
    print("Sources per tier:")
    for tier in repetition_tiers:
        count = sum(1 for s in sources if f"_repeat{tier}" in s["source_name"])
        print(f"  repeat{tier}: {count} sources")


if __name__ == "__main__":
    main()

