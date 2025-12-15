#!/usr/bin/env python3
"""
Generate midtrain-memo.yaml by scanning GCS for .npy files.

This script ensures EXACT repetition control by:
1. Using actual token counts per file (not estimates)
2. Calculating target_ratio = (actual_tokens_in_bucket × repetition) / budget
3. Scaling file counts to fit within budget if needed

Usage:
    python src/scripts/train/OLMo3/generate_midtrain_yaml.py

Requirements:
    - gsutil or google-cloud-storage
    - Access to the GCS bucket
"""

import subprocess
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import yaml

# Configuration
GCS_BASE_PATH = "gs://ai2-llm/yapeic/dolmino2-mix-0925-reconstructed/preprocessed/sources/cc_all_dressed/all_dressed_v3/weborganizer_ft/dclm_plus2_vigintiles/vigintile_0018_subset-decon-2/allenai/dolma2-tokenizer"
OUTPUT_FILE = "src/olmo_core/data/source_mixtures/midtrain-memo-auto.yaml"
TOTAL_BUDGET_TOKENS = 10_000_000_000  # 10B tokens
BYTES_PER_TOKEN = 4  # uint32

# No safety margin needed - we'll handle the gap in normalization by adding
# the difference to the largest source only (which has the most tokens to spare)

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
) -> Tuple[BucketInfo, BucketInfo, BucketInfo]:
    """
    Split files into heavy/medium/light buckets.
    
    Simple approach: split by file count (1:2:6 ratio).
    The EXACT token counts are calculated per bucket, and target_ratio
    is set accordingly to guarantee exact repetition.
    
    Returns (heavy_bucket, medium_bucket, light_bucket).
    """
    if not files:
        empty = BucketInfo(files=[], repeat=1, actual_tokens=0, effective_tokens=0)
        return empty, empty, empty
    
    n = len(files)
    
    # Handle edge case: very few files
    if n < 3:
        # Put all in light bucket (no repetition)
        return (
            BucketInfo(files=[], repeat=6, actual_tokens=0, effective_tokens=0),
            BucketInfo(files=[], repeat=3, actual_tokens=0, effective_tokens=0),
            create_bucket(files, repeat=1),
        )
    
    # Simple split by file count (1:2:6 ratio)
    # The exact distribution doesn't matter - what matters is:
    # target_ratio = (actual_tokens_in_bucket × repetition) / budget
    total_parts = 9
    n_heavy = max(1, n // total_parts)
    n_medium = max(1, 2 * n // total_parts)
    # Light gets the rest
    
    # Split files (sorted by path for determinism)
    sorted_files = sorted(files, key=lambda f: f.path)
    heavy_files = sorted_files[:n_heavy]
    medium_files = sorted_files[n_heavy:n_heavy + n_medium]
    light_files = sorted_files[n_heavy + n_medium:]
    
    # Create buckets - the key is that actual_tokens is computed EXACTLY
    # from the sum of each file's tokens
    heavy = create_bucket(heavy_files, repeat=6)
    medium = create_bucket(medium_files, repeat=3)
    light = create_bucket(light_files, repeat=1)
    
    return heavy, medium, light


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


def normalize_to_sum_one(sources: List[Dict]) -> List[Dict]:
    """
    Normalize ratios to sum to EXACTLY 1.0.
    
    This is necessary because SourceMixtureList.validate() checks:
        if not np.allclose(summed_weights, 1.0):
            raise OLMoConfigurationError(...)
    
    Strategy:
    - If sum > 1.0: Scale down all ratios (safe - we request less than available)
    - If sum < 1.0: DON'T scale up all ratios (would over-request from every source)
      Instead, add the gap to the LARGEST source only (it has the most tokens to spare)
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
    
    if gap < 0:
        # Sum > 1.0: scale down all ratios proportionally (safe)
        scale_factor = 1.0 / total_ratio
        print(f"  Sum > 1.0: Scaling down all ratios by {scale_factor:.6f}")
        for s in sources:
            s["target_ratio"] = s["target_ratio"] * scale_factor
    else:
        # Sum < 1.0: add the gap to a _repeat6 source (has most headroom for extra tokens)
        # _repeat1 sources can't absorb extra tokens (max_repetition_ratio=1.0 means no repeat allowed)
        # _repeat6 sources can absorb a small gap - we also increase max_repetition_ratio accordingly
        repeat6_sources = [s for s in sources if "_repeat6" in s["source_name"]]
        
        if repeat6_sources:
            # Add gap to the largest _repeat6 source
            repeat6_sources.sort(key=lambda s: -s["_effective_tokens"])
            target_source = repeat6_sources[0]
            
            # Calculate how much extra repetition is needed
            old_ratio = target_source["target_ratio"]
            new_ratio = old_ratio + gap
            ratio_increase = new_ratio / old_ratio if old_ratio > 0 else 1.0
            
            # Increase max_repetition_ratio proportionally to allow the extra tokens
            old_max_rep = target_source["max_repetition_ratio"]
            new_max_rep = old_max_rep * ratio_increase
            
            print(f"  Sum < 1.0: Adding gap of {gap:.6f} to _repeat6 source: {target_source['source_name']}")
            print(f"    Ratio: {old_ratio:.6f} → {new_ratio:.6f}")
            print(f"    max_repetition_ratio: {old_max_rep:.3f} → {new_max_rep:.3f}")
            
            target_source["target_ratio"] = new_ratio
            target_source["max_repetition_ratio"] = new_max_rep
        else:
            # Fallback: add to largest source (shouldn't happen with our bucket setup)
            print(f"  Sum < 1.0: No _repeat6 source found, adding gap to: {sources[0]['source_name']}")
            sources[0]["target_ratio"] += gap
    
    # Use integer arithmetic for precise rounding
    PRECISION = 1_000_000
    
    # Round all except the first (largest) to 6 decimal places
    rounded_parts = []
    for s in sources[1:]:
        parts = round(s["target_ratio"] * PRECISION)
        rounded_parts.append(parts)
        s["target_ratio"] = parts / PRECISION
    
    # Set the largest to exactly fill the remainder (absorbs rounding error)
    remainder = PRECISION - sum(rounded_parts)
    sources[0]["target_ratio"] = remainder / PRECISION
    
    # Verify the sum
    final_sum = sum(s["target_ratio"] for s in sources)
    print(f"  Final sum: {final_sum}")
    
    # Double-check with integer arithmetic
    total_parts = remainder + sum(rounded_parts)
    if total_parts == PRECISION:
        print(f"  ✓ Sum is exactly 1.0 (verified with integer arithmetic)")
    else:
        print(f"  WARNING: Integer sum is {total_parts}, expected {PRECISION}!")
    
    return sources


def main():
    print("Scanning GCS for topic files...")
    print(f"Budget: {TOTAL_BUDGET_TOKENS/1e9:.0f}B tokens")
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
        
        heavy, medium, light = split_files_into_buckets(files)
        
        # Add non-empty buckets
        if heavy.files:
            all_buckets.append((f"{topic}_repeat6", heavy))
        if medium.files:
            all_buckets.append((f"{topic}_repeat3", medium))
        if light.files:
            all_buckets.append((f"{topic}_repeat1", light))
        
        total_tokens = sum(f.tokens for f in files)
        total_effective = heavy.effective_tokens + medium.effective_tokens + light.effective_tokens
        
        topic_stats[topic] = {
            "n_files": len(files),
            "heavy": len(heavy.files),
            "medium": len(medium.files),
            "light": len(light.files),
            "total_tokens": total_tokens,
            "total_effective": total_effective,
        }
        
        print(f"    Found {len(files)} files, {total_tokens/1e9:.4f}B tokens")
        print(f"    Buckets: heavy={len(heavy.files)} files ({heavy.actual_tokens/1e9:.4f}B × 6), "
              f"medium={len(medium.files)} files ({medium.actual_tokens/1e9:.4f}B × 3), "
              f"light={len(light.files)} files ({light.actual_tokens/1e9:.4f}B × 1)")
        print(f"    Effective tokens: {total_effective/1e9:.4f}B")
    
    print()
    
    # Scale to budget if needed
    all_buckets = scale_buckets_to_budget(all_buckets)
    
    # Calculate final totals
    total_effective = sum(b.effective_tokens for _, b in all_buckets)
    print(f"\nFinal total effective tokens: {total_effective/1e9:.4f}B")
    
    print("\nGenerating YAML...")
    sources = generate_yaml_sources(all_buckets)
    
    # Normalize to sum to exactly 1.0 (required by SourceMixtureList.validate())
    sources = normalize_to_sum_one(sources)
    
    # Sort by ratio (largest first)
    sources.sort(key=lambda s: -s["target_ratio"])
    
    # Verify sum
    total_ratio = sum(s["target_ratio"] for s in sources)
    print(f"Total ratio: {total_ratio:.6f}")
    
    if abs(total_ratio - 1.0) > 0.01:
        print(f"WARNING: Total ratio is {total_ratio:.4f}, expected ~1.0")
        print("This means not all of the 10B budget will be used, which is fine.")
    
    # Build YAML structure (remove metadata fields for output)
    yaml_sources = []
    for s in sources:
        yaml_sources.append({
            "source_name": s["source_name"],
            "target_ratio": s["target_ratio"],
            "max_repetition_ratio": s["max_repetition_ratio"],
            "paths": s["paths"],
        })
    
    yaml_content = {"sources": yaml_sources}
    
    # Write YAML with nice formatting
    with open(OUTPUT_FILE, "w") as f:
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
        f.write("#   - _repeat6: Heavy repetition (6x)\n")
        f.write("#   - _repeat3: Medium repetition (3x)\n")
        f.write("#   - _repeat1: No repetition (1x)\n")
        f.write("#\n\n")
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, width=200)
    
    print(f"\nWritten to {OUTPUT_FILE}")
    print(f"Total sources: {len(sources)}")
    
    # Print detailed summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"{'Source Name':<45} {'Files':>6} {'Actual':>10} {'×Rep':>5} {'Effective':>12} {'Ratio':>10}")
    print("-"*80)
    
    for s in sources:
        name = s["source_name"]
        n_files = s["_n_files"]
        actual = s["_actual_tokens"] / 1e9
        repeat = int(s["max_repetition_ratio"])
        effective = s["_effective_tokens"] / 1e9
        ratio = s["target_ratio"]
        print(f"{name:<45} {n_files:>6} {actual:>9.4f}B ×{repeat:>3} {effective:>11.4f}B {ratio:>10.6f}")
    
    print("-"*80)
    print(f"{'TOTAL':<45} {'':<6} {'':<10} {'':<5} {total_effective/1e9:>11.4f}B {total_ratio:>10.6f}")
    print()
    
    # Verify exact repetition guarantee
    print("\nEXACT REPETITION GUARANTEE CHECK:")
    all_pass = True
    for s in sources:
        expected_ratio = s["_effective_tokens"] / TOTAL_BUDGET_TOKENS
        actual_ratio = s["target_ratio"]
        diff = abs(expected_ratio - actual_ratio)
        if diff > 0.000001:
            print(f"  MISMATCH: {s['source_name']}: expected {expected_ratio:.6f}, got {actual_ratio:.6f}")
            all_pass = False
    
    if all_pass:
        print("  ✓ All ratios match expected values for exact repetition")
    
    print("\nFormula verification:")
    print("  For exact repetition, the training framework will request:")
    print(f"    tokens_requested = target_ratio × budget")
    print(f"  For each source, tokens_requested should equal:")
    print(f"    actual_tokens × max_repetition_ratio")
    print("  This ensures each file is repeated exactly max_repetition_ratio times.")


if __name__ == "__main__":
    main()
