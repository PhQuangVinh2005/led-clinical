#!/usr/bin/env python3
"""
CLI script for data preprocessing.

Usage:
    python scripts/preprocess.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import preprocess


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV-BHC data")
    parser.add_argument("--csv-path", type=str,
                        default="data/raw/mimic-iv-bhc.csv",
                        help="Path to MIMIC-IV-BHC CSV file")
    parser.add_argument("--held-out-path", type=str,
                        default="data/held_out/held_out_ids.txt",
                        help="Path to held-out note IDs file")
    parser.add_argument("--output-dir", type=str,
                        default="data/processed",
                        help="Output directory for train/val/test JSONL files")
    parser.add_argument("--min-target-tokens", type=int, default=50,
                        help="Minimum target token count")
    parser.add_argument("--max-target-tokens", type=int, default=2000,
                        help="Maximum target token count")
    parser.add_argument("--parquet-path", type=str,
                        default="data/raw/drug-dictionary/heh.parquet",
                        help="Path to heh.parquet dictionary")
    parser.add_argument("--corruption-rate", type=float, default=0.3,
                        help="Rate of corruption for pre-applied errors")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

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
    )


if __name__ == "__main__":
    main()
