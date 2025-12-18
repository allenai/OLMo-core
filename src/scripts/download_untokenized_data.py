#!/usr/bin/env python3
"""
Download untokenized data from CSV metadata files.

For each .npy tokenized file, there's a corresponding .csv.gz metadata file
that maps back to the original source documents.

CSV.gz columns:
- start (int): Start index in the .npy tokenized file (0-indexed)
- end (int): End index in the .npy tokenized file (0-indexed, exclusive)
- id (str): Unique document identifier
- src (str): Path to the original source file (e.g., s3://.../*.json.gz)
- loc (int): Line number in the source file (1-indexed)

Output Structure (when using --yaml-path):
    output/
    ├── metadata.json                    # Global index with stats
    ├── art_and_design/
    │   ├── repeat1/documents.jsonl.gz
    │   ├── repeat3/documents.jsonl.gz
    │   ├── repeat6/documents.jsonl.gz
    │   └── repeat10/documents.jsonl.gz
    ├── science_math_and_technology/
    │   └── repeat1/documents.jsonl.gz
    └── ...

Usage:
    # Download all source files referenced in the CSV files for a given YAML mix
    python src/scripts/download_untokenized_data.py \
        --yaml-path src/olmo_core/data/source_mixtures/midtrain-memo-auto.yaml \
        --output-dir /path/to/output

    # Preview what would be downloaded (dry run)
    python src/scripts/download_untokenized_data.py \
        --yaml-path src/olmo_core/data/source_mixtures/midtrain-memo-auto.yaml \
        --dry-run

    # Download from a single NPY file's corresponding CSV
    python src/scripts/download_untokenized_data.py \
        --npy-path "gs://ai2-llm/.../part-0-00000.npy" \
        --output-dir /path/to/output

    # With path remapping (CSV has local paths, but data is on GCS)
    python src/scripts/download_untokenized_data.py \
        --yaml-path src/olmo_core/data/source_mixtures/midtrain-memo-auto.yaml \
        --path-remap "/mnt/raid0/ai2-llm/pretraining-data/sources/dolmino2-mix-0925-reconstructed=gs://ai2-llm/yapeic/dolmino2-mix-0925-reconstructed" \
        --output-dir /path/to/output
"""

import argparse
import csv
import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import smart_open
import yaml


# Default path remappings for common cases
# Format: (old_prefix, new_prefix)
DEFAULT_PATH_REMAPS = [
    # Local mount paths to GCS paths
    (
        "/mnt/raid0/ai2-llm/pretraining-data/sources/dolmino2-mix-0925-reconstructed",
        "gs://ai2-llm/yapeic/dolmino2-mix-0925-reconstructed"
    ),
]


@dataclass
class DocumentMetadata:
    """Metadata for a single tokenized document."""
    start: int
    end: int
    doc_id: str
    src: str
    loc: int


@dataclass
class SourceInfo:
    """Information about a source from the YAML file."""
    source_name: str
    topic: str
    repetition_rate: int  # 1, 3, 6, or 10
    target_ratio: float
    max_repetition_ratio: float
    paths: list[str] = field(default_factory=list)


def parse_source_name(source_name: str) -> tuple[str, int]:
    """
    Parse source name to extract topic and repetition rate.
    
    Example: "science_math_and_technology_repeat1" -> ("science_math_and_technology", 1)
    """
    match = re.match(r"(.+)_repeat(\d+)$", source_name)
    if match:
        topic = match.group(1)
        rep_rate = int(match.group(2))
        return topic, rep_rate
    # Fallback: no repetition suffix found
    return source_name, 1


def load_sources_from_yaml(yaml_path: str) -> list[SourceInfo]:
    """Load source information from YAML file with topic and repetition metadata."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    sources = []
    for source in data.get("sources", []):
        source_name = source.get("source_name", "")
        topic, rep_rate = parse_source_name(source_name)
        sources.append(SourceInfo(
            source_name=source_name,
            topic=topic,
            repetition_rate=rep_rate,
            target_ratio=source.get("target_ratio", 0.0),
            max_repetition_ratio=source.get("max_repetition_ratio", 1.0),
            paths=source.get("paths", [])
        ))
    return sources


class PathRemapper:
    """Handles path remapping from CSV paths to actual data locations."""
    
    def __init__(self, remaps: Optional[list[tuple[str, str]]] = None):
        """
        Initialize with a list of (old_prefix, new_prefix) tuples.
        """
        self.remaps = remaps or []
    
    @classmethod
    def from_args(cls, remap_args: Optional[list[str]]) -> "PathRemapper":
        """
        Create from command-line arguments.
        
        Args:
            remap_args: List of strings in format "old_prefix=new_prefix"
        """
        remaps = []
        if remap_args:
            for remap in remap_args:
                if "=" not in remap:
                    raise ValueError(f"Invalid remap format: {remap}. Expected 'old=new'")
                old, new = remap.split("=", 1)
                remaps.append((old, new))
        return cls(remaps)
    
    def remap(self, path: str) -> str:
        """Remap a path using configured remappings."""
        for old_prefix, new_prefix in self.remaps:
            if path.startswith(old_prefix):
                return new_prefix + path[len(old_prefix):]
        return path


def get_csv_path(npy_path: str) -> str:
    """Convert .npy path to corresponding .csv.gz path."""
    if npy_path.endswith(".npy"):
        return npy_path[:-4] + ".csv.gz"
    return npy_path + ".csv.gz"


def read_csv_metadata(csv_path: str, path_remapper: Optional[PathRemapper] = None) -> Iterator[DocumentMetadata]:
    """Read CSV.gz metadata file and yield document metadata."""
    try:
        with smart_open.open(csv_path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                src_path = row[3]
                if path_remapper:
                    src_path = path_remapper.remap(src_path)
                yield DocumentMetadata(
                    start=int(row[0]),
                    end=int(row[1]),
                    doc_id=row[2],
                    src=src_path,
                    loc=int(row[4])
                )
    except Exception as e:
        print(f"Warning: Could not read {csv_path}: {e}")


def extract_npy_paths_from_yaml(yaml_path: str) -> list[str]:
    """Extract all .npy paths from a source mixture YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    paths = []
    for source in data.get("sources", []):
        paths.extend(source.get("paths", []))
    return paths


def collect_source_files(
    npy_paths: list[str],
    path_remapper: Optional[PathRemapper] = None,
    verbose: bool = True
) -> dict[str, list[tuple[str, int]]]:
    """
    Collect all source files referenced in CSV metadata files.
    
    Returns:
        Dict mapping source file paths to list of (doc_id, loc) tuples
    """
    source_files: dict[str, list[tuple[str, int]]] = defaultdict(list)
    
    for i, npy_path in enumerate(npy_paths):
        csv_path = get_csv_path(npy_path)
        if verbose:
            print(f"[{i+1}/{len(npy_paths)}] Scanning {csv_path}")
        
        for meta in read_csv_metadata(csv_path, path_remapper):
            source_files[meta.src].append((meta.doc_id, meta.loc))
    
    return source_files


def collect_source_files_by_topic(
    sources: list[SourceInfo],
    path_remapper: Optional[PathRemapper] = None,
    max_sources: Optional[int] = None,
    verbose: bool = True
) -> dict[str, dict[int, dict[str, list[tuple[str, int]]]]]:
    """
    Collect source files organized by topic and repetition rate.
    
    Returns:
        Nested dict: topic -> repetition_rate -> source_path -> [(doc_id, loc), ...]
    """
    # Structure: topic -> rep_rate -> src_path -> [(doc_id, loc), ...]
    organized: dict[str, dict[int, dict[str, list[tuple[str, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    
    sources_to_process = sources[:max_sources] if max_sources else sources
    total_paths = sum(len(s.paths) for s in sources_to_process)
    path_idx = 0
    
    for source in sources_to_process:
        topic = source.topic
        rep_rate = source.repetition_rate
        
        for npy_path in source.paths:
            path_idx += 1
            csv_path = get_csv_path(npy_path)
            if verbose:
                print(f"[{path_idx}/{total_paths}] [{topic}/repeat{rep_rate}] Scanning {os.path.basename(csv_path)}")
            
            for meta in read_csv_metadata(csv_path, path_remapper):
                organized[topic][rep_rate][meta.src].append((meta.doc_id, meta.loc))
    
    return organized


def read_document_from_source(src_path: str, loc: int) -> Optional[dict]:
    """
    Read a single document from a source file.
    
    Args:
        src_path: Path to the source file (typically .json.gz or .jsonl.gz)
        loc: 1-indexed line number
    
    Returns:
        The document as a dict, or None if not found
    """
    try:
        with smart_open.open(src_path, "rt") as f:
            for line_num, line in enumerate(f, start=1):
                if line_num == loc:
                    return json.loads(line)
    except Exception as e:
        print(f"Error reading {src_path} line {loc}: {e}")
    return None


def download_source_documents(
    source_files: dict[str, list[tuple[str, int]]],
    output_dir: str,
    max_files: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Download documents from source files and save them.
    
    Args:
        source_files: Dict mapping source paths to list of (doc_id, loc) tuples
        output_dir: Directory to save output files
        max_files: Maximum number of source files to process (for testing)
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group documents by source file
    source_list = list(source_files.items())
    if max_files:
        source_list = source_list[:max_files]
    
    all_documents = []
    
    for i, (src_path, doc_refs) in enumerate(source_list):
        if verbose:
            print(f"[{i+1}/{len(source_list)}] Reading {len(doc_refs)} docs from {src_path}")
        
        # Sort by line number for efficient sequential reading
        sorted_refs = sorted(doc_refs, key=lambda x: x[1])
        
        try:
            with smart_open.open(src_path, "rt") as f:
                needed_locs = {loc for _, loc in sorted_refs}
                loc_to_docs = {}
                
                for line_num, line in enumerate(f, start=1):
                    if line_num in needed_locs:
                        doc = json.loads(line)
                        loc_to_docs[line_num] = doc
                    if line_num > max(needed_locs):
                        break
                
                for doc_id, loc in sorted_refs:
                    if loc in loc_to_docs:
                        doc = loc_to_docs[loc]
                        doc["_metadata"] = {
                            "doc_id": doc_id,
                            "src": src_path,
                            "loc": loc
                        }
                        all_documents.append(doc)
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
    
    # Save all documents
    output_path = os.path.join(output_dir, "documents.jsonl.gz")
    with gzip.open(output_path, "wt") as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + "\n")
    
    print(f"\nSaved {len(all_documents)} documents to {output_path}")


def fetch_documents_from_sources(
    source_files: dict[str, list[tuple[str, int]]],
    topic: str,
    rep_rate: int,
    verbose: bool = True
) -> list[dict]:
    """Fetch documents from source files, adding topic/repetition metadata."""
    documents = []
    source_list = list(source_files.items())
    
    for i, (src_path, doc_refs) in enumerate(source_list):
        if verbose:
            print(f"  [{i+1}/{len(source_list)}] Reading {len(doc_refs)} docs from {os.path.basename(src_path)}")
        
        sorted_refs = sorted(doc_refs, key=lambda x: x[1])
        
        try:
            with smart_open.open(src_path, "rt") as f:
                needed_locs = {loc for _, loc in sorted_refs}
                loc_to_docs = {}
                
                for line_num, line in enumerate(f, start=1):
                    if line_num in needed_locs:
                        doc = json.loads(line)
                        loc_to_docs[line_num] = doc
                    if line_num > max(needed_locs):
                        break
                
                for doc_id, loc in sorted_refs:
                    if loc in loc_to_docs:
                        doc = loc_to_docs[loc]
                        doc["_metadata"] = {
                            "doc_id": doc_id,
                            "src": src_path,
                            "loc": loc,
                            "topic": topic,
                            "repetition_rate": rep_rate
                        }
                        documents.append(doc)
        except Exception as e:
            print(f"  Error processing {src_path}: {e}")
    
    return documents


def download_by_topic_and_repetition(
    organized_sources: dict[str, dict[int, dict[str, list[tuple[str, int]]]]],
    output_dir: str,
    verbose: bool = True
) -> dict:
    """
    Download documents organized by topic and repetition rate.
    
    Creates structure:
        output_dir/
        ├── metadata.json
        ├── topic1/
        │   ├── repeat1/documents.jsonl.gz
        │   └── repeat3/documents.jsonl.gz
        └── topic2/
            └── ...
    
    Returns:
        Metadata dict with statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "topics": {},
        "total_documents": 0,
        "total_source_files": 0
    }
    
    for topic, rep_rates in organized_sources.items():
        metadata["topics"][topic] = {"repetition_rates": {}}
        
        for rep_rate, source_files in rep_rates.items():
            if verbose:
                total_docs = sum(len(refs) for refs in source_files.values())
                print(f"\n[{topic}/repeat{rep_rate}] Downloading {total_docs} docs from {len(source_files)} files...")
            
            # Fetch documents
            documents = fetch_documents_from_sources(source_files, topic, rep_rate, verbose)
            
            # Create output directory
            topic_dir = os.path.join(output_dir, topic, f"repeat{rep_rate}")
            os.makedirs(topic_dir, exist_ok=True)
            
            # Save documents
            output_path = os.path.join(topic_dir, "documents.jsonl.gz")
            with gzip.open(output_path, "wt") as f:
                for doc in documents:
                    f.write(json.dumps(doc) + "\n")
            
            # Update metadata
            metadata["topics"][topic]["repetition_rates"][rep_rate] = {
                "document_count": len(documents),
                "source_file_count": len(source_files),
                "output_path": output_path
            }
            metadata["total_documents"] += len(documents)
            metadata["total_source_files"] += len(source_files)
            
            if verbose:
                print(f"  Saved {len(documents)} documents to {output_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Total documents: {metadata['total_documents']}")
        print(f"Total source files: {metadata['total_source_files']}")
        print(f"Metadata saved to: {metadata_path}")
    
    return metadata


def print_summary(source_files: dict[str, list[tuple[str, int]]]) -> None:
    """Print a summary of source files to be downloaded."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_docs = sum(len(refs) for refs in source_files.values())
    print(f"Total source files: {len(source_files)}")
    print(f"Total documents: {total_docs}")
    
    # Group by prefix for easier understanding
    prefixes = defaultdict(int)
    for src_path in source_files.keys():
        # Get first 3 path components after protocol
        parts = src_path.replace("gs://", "").replace("s3://", "").split("/")
        prefix = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
        prefixes[prefix] += 1
    
    print("\nSource file locations:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:20]:
        print(f"  {prefix}: {count} files")
    if len(prefixes) > 20:
        print(f"  ... and {len(prefixes) - 20} more locations")


def print_organized_summary(
    organized_sources: dict[str, dict[int, dict[str, list[tuple[str, int]]]]]
) -> None:
    """Print a summary organized by topic and repetition rate."""
    print("\n" + "=" * 60)
    print("SUMMARY BY TOPIC AND REPETITION RATE")
    print("=" * 60)
    
    total_docs = 0
    total_files = 0
    
    for topic in sorted(organized_sources.keys()):
        rep_rates = organized_sources[topic]
        print(f"\n{topic}:")
        
        for rep_rate in sorted(rep_rates.keys()):
            source_files = rep_rates[rep_rate]
            doc_count = sum(len(refs) for refs in source_files.values())
            file_count = len(source_files)
            total_docs += doc_count
            total_files += file_count
            print(f"  repeat{rep_rate}: {doc_count:,} documents from {file_count:,} source files")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_docs:,} documents from {total_files:,} source files")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Download untokenized data from CSV metadata files")
    parser.add_argument(
        "--yaml-path",
        type=str,
        help="Path to source mixture YAML file"
    )
    parser.add_argument(
        "--npy-path",
        type=str,
        help="Path to a single .npy file (will read its corresponding .csv.gz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./untokenized_data",
        help="Output directory for downloaded documents"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--max-npy-files",
        type=int,
        default=None,
        help="Maximum total number of NPY files to scan across all sources (for testing)"
    )
    parser.add_argument(
        "--max-npy-per-source",
        type=int,
        default=None,
        help="Maximum NPY files per source (topic+repeat combination)"
    )
    parser.add_argument(
        "--max-npy-per-topic",
        type=int,
        default=None,
        help="Maximum NPY files per topic (across all repeat rates)"
    )
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=None,
        help="Maximum number of source files to download (for testing)"
    )
    parser.add_argument(
        "--path-remap",
        type=str,
        action="append",
        dest="path_remaps",
        help="Path remapping in format 'old_prefix=new_prefix'. Can be specified multiple times. "
             "Example: --path-remap '/mnt/raid0/ai2-llm/...=gs://ai2-llm/yapeic/...'"
    )
    parser.add_argument(
        "--use-default-remaps",
        action="store_true",
        help="Use default path remappings (local mount paths to GCS)"
    )
    
    args = parser.parse_args()
    
    if not args.yaml_path and not args.npy_path:
        parser.error("Must specify either --yaml-path or --npy-path")
    
    # Set up path remapper
    path_remapper = PathRemapper.from_args(args.path_remaps)
    if args.use_default_remaps:
        path_remapper.remaps.extend(DEFAULT_PATH_REMAPS)
    
    if path_remapper.remaps:
        print("\nPath remappings:")
        for old, new in path_remapper.remaps:
            print(f"  {old}")
            print(f"    -> {new}")
    
    # Use different workflows for YAML (organized by topic) vs single NPY file
    if args.yaml_path:
        # YAML mode: organize by topic and repetition rate
        print(f"\nReading YAML: {args.yaml_path}")
        sources = load_sources_from_yaml(args.yaml_path)
        print(f"Found {len(sources)} sources with {sum(len(s.paths) for s in sources)} total NPY files")
        
        # Apply limits
        if args.max_npy_per_source or args.max_npy_per_topic or args.max_npy_files:
            limited_sources = []
            global_count = 0
            topic_counts: dict[str, int] = defaultdict(int)
            
            for source in sources:
                # Check global limit
                if args.max_npy_files and global_count >= args.max_npy_files:
                    break
                
                # Check per-topic limit
                if args.max_npy_per_topic:
                    topic_remaining = args.max_npy_per_topic - topic_counts[source.topic]
                    if topic_remaining <= 0:
                        continue
                else:
                    topic_remaining = float('inf')
                
                # Check per-source limit
                source_limit = args.max_npy_per_source if args.max_npy_per_source else float('inf')
                
                # Check global remaining
                if args.max_npy_files:
                    global_remaining = args.max_npy_files - global_count
                else:
                    global_remaining = float('inf')
                
                # Take the minimum of all limits
                max_paths = int(min(len(source.paths), source_limit, topic_remaining, global_remaining))
                
                if max_paths > 0:
                    if max_paths < len(source.paths):
                        # Partially include this source
                        partial = SourceInfo(
                            source_name=source.source_name,
                            topic=source.topic,
                            repetition_rate=source.repetition_rate,
                            target_ratio=source.target_ratio,
                            max_repetition_ratio=source.max_repetition_ratio,
                            paths=source.paths[:max_paths]
                        )
                        limited_sources.append(partial)
                    else:
                        limited_sources.append(source)
                    
                    global_count += max_paths
                    topic_counts[source.topic] += max_paths
            
            sources = limited_sources
            print(f"After applying limits: {sum(len(s.paths) for s in sources)} NPY files")
            if args.max_npy_per_topic:
                print(f"  Per-topic counts: {dict(topic_counts)}")
        
        # Collect source files organized by topic/repetition
        print("\nScanning CSV metadata files...")
        organized_sources = collect_source_files_by_topic(sources, path_remapper)
        
        # Print organized summary
        print_organized_summary(organized_sources)
        
        if args.dry_run:
            print("\nDry run complete. Remove --dry-run to download documents.")
            return
        
        # Download organized by topic and repetition rate
        print(f"\nDownloading documents to {args.output_dir}...")
        download_by_topic_and_repetition(organized_sources, args.output_dir)
    
    else:
        # Single NPY file mode: simple flat download
        npy_paths = [args.npy_path]
        print(f"\nScanning CSV metadata for: {args.npy_path}")
        source_files = collect_source_files(npy_paths, path_remapper)
        
        # Print summary
        print_summary(source_files)
        
        if args.dry_run:
            print("\nDry run complete. Remove --dry-run to download documents.")
            return
        
        # Download documents
        print(f"\nDownloading documents to {args.output_dir}...")
        download_source_documents(
            source_files,
            args.output_dir,
            max_files=args.max_source_files
        )


if __name__ == "__main__":
    main()

