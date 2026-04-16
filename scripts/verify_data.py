#!/usr/bin/env python3
"""
Diagnostic script to verify on-the-fly data corruption and LED input formatting.

Loads a JSONL file, applies the error synthesizer, and prints examples
showing:
1. Original Summary
2. Corrupted Summary
3. Error Details (type, from -> to)
4. Final LED input string
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.drug_dictionary import DrugDictionary
from src.data.error_synthesizer import ClinicalErrorSynthesizer


def print_colored_diff(original: str, corrupted: str):
    """Simple word-level diff for visualization."""
    import difflib
    
    # Simple highlighting: Red for removal, Green for addition
    # Since we are in terminal, we use ANSI codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    
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
            output.append(f"{RED}{' '.join(words_orig[a0:a1])}{RESET} -> {GREEN}{' '.join(words_corr[b0:b1])}{RESET}")
            
    print(f"\n[Visual Diff]: {' '.join(output)}")


def main():
    parser = argparse.ArgumentParser(description="Verify clinical error synthesis")
    parser.add_argument("--file", type=str, default="data/processed/val.jsonl",
                        help="Path to JSONL file to check")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to display")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Force corruption rate to 1.0 for testing")
    parser.add_argument("--model", type=str, default="allenai/led-base-16384",
                        help="Tokenizer to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found. Have you run scripts/preprocess.py?")
        return

    # 1. Load Dictionary
    print("Loading drug dictionary...")
    drug_dict = DrugDictionary(parquet_path="data/raw/drug-dictionary/heh.parquet")
    
    # 2. Setup Synthesizer
    synth = ClinicalErrorSynthesizer(
        drug_dictionary=drug_dict,
        corruption_rate=args.rate,
        seed=42
    )

    # 3. Setup Tokenizer for input string check
    print(f"Loading tokenizer: {args.model}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model) # Unused for now

    # 4. Process samples
    print(f"\n--- Showing {args.num_samples} samples from {args.file} ---")
    
    samples_shown = 0
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in f:
            if samples_shown >= args.num_samples:
                break
                
            record = json.loads(line)
            
            # Check if this is a pre-corrupted file or a raw one
            is_pre_corrupted = 'corrupted_summary' in record and 'true_summary' in record
            
            if is_pre_corrupted:
                source_note = record['input']
                true_summary = record['true_summary']
                # corrupted_summary = record['corrupted_summary'] # Unused for now
                # We can't see details for pre-baked errors unless we check the log file,
                # but for this diagnostic we can re-run synth on the true summary to see logic
                result = synth.corrupt(true_summary, source_note)
            else:
                source_note = record['input']
                true_summary = record['target']
                result = synth.corrupt(true_summary, source_note)

            if not result.is_corrupted:
                continue
                
            print(f"\n{'='*80}")
            print(f"SAMPLE #{samples_shown + 1} | Error Types: {', '.join(result.error_types)} "
                  f"{'(Pre-baked)' if is_pre_corrupted else '(On-the-fly)'}")
            print(f"{'='*80}")
            
            print(f"\n[TRUE SUMMARY]:\n{true_summary}")
            print(f"\n[CORRUPTED SUMMARY]:\n{result.corrupted_summary}")
            
            print_colored_diff(true_summary, result.corrupted_summary)
            
            print("\n[ERROR DETAILS]:")
            for detail in result.error_details:
                e_type = detail.get('type')
                orig = detail.get('original')
                corr = detail.get('corrupted')
                e_class = detail.get('entity_class', 'N/A')
                print(f"  - {e_type}: '{orig}' -> '{corr}' (Class: {e_class})")
            
            # Show LED input string (first 200 chars)
            input_str = result.corrupted_summary + " </s> " + source_note
            print("\n[LED INPUT PREFIX (first 200 chars)]:")
            print(f"{input_str[:200]}...")
            
            samples_shown += 1


if __name__ == "__main__":
    main()
