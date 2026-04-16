#!/usr/bin/env python3
"""
CLI script for data preprocessing.

Usage:
    # Full run (writes train/val/test.jsonl + corruption_log.jsonl):
    python scripts/preprocess.py

    # Dry run — process first 1000 records, print corruption stats, no files written:
    python scripts/preprocess.py --dry-run

    # Dry run with custom slice size:
    python scripts/preprocess.py --dry-run --dry-run-n 500

    # Force overwrite even if output files already exist:
    python scripts/preprocess.py --no-skip
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-IV-BHC data for LED error correction training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ───────────────────────────────────────────────────────────
    parser.add_argument("--csv-path", type=str,
                        default="data/raw/mimic-iv-bhc.csv",
                        help="Path to MIMIC-IV-BHC CSV file")
    parser.add_argument("--held-out-path", type=str,
                        default="data/held_out/held_out_ids.txt",
                        help="Path to held-out note IDs file")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed",
                        help="Output directory for train/val/test JSONL files")
    parser.add_argument("--parquet-path", type=str,
                        default="data/raw/drug-dictionary/heh.parquet",
                        help="Path to heh.parquet drug dictionary")

    # ── Filtering ────────────────────────────────────────────────────────────
    parser.add_argument("--min-target-tokens", type=int, default=50,
                        help="Minimum target token count")
    parser.add_argument("--max-target-tokens", type=int, default=2000,
                        help="Maximum target token count")

    # ── Corruption ───────────────────────────────────────────────────────────
    parser.add_argument("--corruption-rate", type=float, default=0.3,
                        help="Fraction of samples to corrupt (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # ── Run-mode flags ───────────────────────────────────────────────────────
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only the first --dry-run-n records; "
                             "print corruption stats but write NO files")
    parser.add_argument("--dry-run-n", type=int, default=1000,
                        help="Number of records to sample in dry-run mode")
    parser.add_argument("--no-skip", action="store_true",
                        help="Overwrite existing output files (default: skip them)")

    args = parser.parse_args()

    preprocess(
        csv_path=args.csv_path,
        held_out_path=args.held_out_path,
        output_dir=args.output_dir,
        min_target_tokens=args.min_target_tokens,
        max_target_tokens=args.max_target_tokens,
        parquet_path=args.parquet_path,
        corruption_rate=args.corruption_rate,
        seed=args.seed,
        dry_run=args.dry_run,
        dry_run_n=args.dry_run_n,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
