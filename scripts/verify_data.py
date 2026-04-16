#!/usr/bin/env python3
"""
Diagnostic script to verify preprocessed data and show corruption examples.

Usage:
    # Show 5 examples from val split (default):
    python scripts/verify_data.py

    # Show 10 examples from train split:
    python scripts/verify_data.py --file data/processed/train.jsonl --num-samples 10

    # Run integrity check only (no sample display):
    python scripts/verify_data.py --integrity-only

    # Show corruption log stats:
    python scripts/verify_data.py --log-stats

Two modes depending on JSONL schema:
  • Pre-baked (new):  record has 'corrupted_summary' + 'true_summary'
    → reads baked values directly and shows the real corruption
  • Legacy (old):     record has only 'input'/'target'
    → re-runs synthesizer on-the-fly (for development use)
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Required keys for the new static schema ──────────────────────────────────
REQUIRED_KEYS = {"note_id", "input", "true_summary", "corrupted_summary"}


# ── Terminal colours ──────────────────────────────────────────────────────────
RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def print_colored_diff(original: str, corrupted: str) -> None:
    """Word-level coloured diff (red = removed, green = added)."""
    import difflib

    words_orig = original.split()
    words_corr = corrupted.split()
    matcher = difflib.SequenceMatcher(None, words_orig, words_corr)
    output = []

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(" ".join(words_orig[a0:a1]))
        elif opcode == 'insert':
            output.append(f"{GREEN}{' '.join(words_corr[b0:b1])}{RESET}")
        elif opcode == 'delete':
            output.append(f"{RED}{' '.join(words_orig[a0:a1])}{RESET}")
        elif opcode == 'replace':
            output.append(
                f"{RED}{' '.join(words_orig[a0:a1])}{RESET}"
                f" → {GREEN}{' '.join(words_corr[b0:b1])}{RESET}"
            )

    print(f"\n{BOLD}[Visual Diff]:{RESET} {' '.join(output)}")


def integrity_check(path: str) -> bool:
    """
    Verify every record in the JSONL has the required static-schema keys.
    Prints a summary and returns True if the file is clean.
    """
    total = 0
    missing_key_counts: Counter = Counter()
    bad_record_count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  {RED}Line {lineno}: JSON decode error — {e}{RESET}")
                bad_record_count += 1
                continue

            missing = REQUIRED_KEYS - set(record.keys())
            if missing:
                for k in missing:
                    missing_key_counts[k] += 1
                bad_record_count += 1

    print(f"\n{BOLD}[Integrity: {os.path.basename(path)}]{RESET}")
    print(f"  Total records : {total:,}")
    print(f"  Bad records   : {bad_record_count}")

    if missing_key_counts:
        print(f"  {RED}Missing keys:{RESET}")
        for k, n in missing_key_counts.most_common():
            print(f"    '{k}' missing in {n:,} records")
        return False
    else:
        print(f"  {GREEN}All records have required keys ✓{RESET}")
        return True


def log_stats(log_path: str) -> None:
    """Print corruption statistics from the corruption_log.jsonl."""
    if not os.path.exists(log_path):
        print(f"{YELLOW}[warn] Corruption log not found: {log_path}{RESET}")
        return

    total = 0
    type_counter: Counter = Counter()
    split_counter: Counter = Counter()

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            entry = json.loads(line)
            split_counter[entry.get("split", "unknown")] += 1
            for detail in entry.get("applied_errors", []):
                type_counter[detail.get("type", "UNKNOWN")] += 1

    print(f"\n{BOLD}[Corruption Log: {os.path.basename(log_path)}]{RESET}")
    print(f"  Total corrupted events : {total:,}")
    print(f"\n  By split:")
    for split, count in sorted(split_counter.items()):
        print(f"    {split:<10} {count:>8,}")
    print(f"\n  By error type:")
    for etype, count in type_counter.most_common():
        print(f"    {etype:<14} {count:>8,}")


def show_samples(path: str, num_samples: int) -> None:
    """Display corruption examples from a JSONL file."""
    samples_shown = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if samples_shown >= num_samples:
                break
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            is_pre_baked = "corrupted_summary" in record and "true_summary" in record

            if is_pre_baked:
                true_summary = record['true_summary']
                corrupted_summary = record['corrupted_summary']
                source_note = record.get('input', '')
                # Skip pass-through (uncorrupted) records
                if true_summary == corrupted_summary:
                    continue
            else:
                # Legacy fallback: re-run synth on-the-fly
                # (only happens for old-schema JSONL during development)
                from src.data.drug_dictionary import DrugDictionary
                from src.data.error_synthesizer import ClinicalErrorSynthesizer
                source_note = record.get('input', '')
                true_summary = record.get('target', '')

                if not hasattr(show_samples, '_synth'):
                    print("Loading drug dictionary for legacy on-the-fly mode...")
                    dd = DrugDictionary("data/raw/drug-dictionary/heh.parquet")
                    show_samples._synth = ClinicalErrorSynthesizer(dd, corruption_rate=1.0, seed=42)

                result = show_samples._synth.corrupt(true_summary, source_note)
                if not result.is_corrupted:
                    continue
                corrupted_summary = result.corrupted_summary

            samples_shown += 1
            schema_tag = "(pre-baked)" if is_pre_baked else "(on-the-fly)"
            print(f"\n{'='*80}")
            print(f"{BOLD}SAMPLE #{samples_shown}{RESET} {YELLOW}{schema_tag}{RESET}")
            print(f"{'='*80}")
            print(f"\n{BOLD}[NOTE ID]:{RESET} {record.get('note_id', 'N/A')}")
            print(f"\n{BOLD}[TRUE SUMMARY]:{RESET}\n{true_summary[:500]}"
                  f"{'...' if len(true_summary) > 500 else ''}")
            print(f"\n{BOLD}[CORRUPTED SUMMARY]:{RESET}\n{corrupted_summary[:500]}"
                  f"{'...' if len(corrupted_summary) > 500 else ''}")

            print_colored_diff(true_summary, corrupted_summary)

            # Show LED input prefix
            input_str = corrupted_summary + " </s> " + source_note
            print(f"\n{BOLD}[LED INPUT PREFIX (first 200 chars)]:{RESET}")
            print(f"{input_str[:200]}...")

    if samples_shown == 0:
        print(f"{YELLOW}[warn] No corrupted samples found in first records. "
              f"All may be pass-throughs.{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify preprocessed JSONL data for LED error correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", type=str,
                        default="data/processed/val.jsonl",
                        help="JSONL file to inspect")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of corrupted examples to display")
    parser.add_argument("--integrity-only", action="store_true",
                        help="Only run the integrity check; skip sample display")
    parser.add_argument("--log-stats", action="store_true",
                        help="Print corruption_log.jsonl statistics")
    parser.add_argument("--log-path", type=str,
                        default="data/processed/corruption_log.jsonl",
                        help="Path to corruption log for --log-stats")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"{RED}Error: {args.file} not found. "
              f"Have you run scripts/preprocess.py?{RESET}")
        sys.exit(1)

    # Always run integrity check
    ok = integrity_check(args.file)

    if args.log_stats:
        log_stats(args.log_path)

    if not args.integrity_only and ok:
        print(f"\n{BOLD}--- Showing up to {args.num_samples} corrupted samples "
              f"from {args.file} ---{RESET}")
        show_samples(args.file, args.num_samples)


if __name__ == "__main__":
    main()
