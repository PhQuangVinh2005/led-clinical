"""
Preprocessor — Load MIMIC-IV-BHC data, exclude held-out samples,
apply target filters, and create stratified train/val/test splits.

Improvements over v1:
  - Skip-if-exists: avoids re-running on already-processed splits.
  - Per-split corruption statistics (hit-rate, error-type breakdown).
  - --dry-run support (first N records only, no files written).
"""

import csv
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from src.data.drug_dictionary import DrugDictionary
from src.data.error_synthesizer import ClinicalErrorSynthesizer


def load_held_out_ids(held_out_path: str) -> Set[str]:
    """Load the set of held-out note IDs to exclude from training."""
    ids = set()
    with open(held_out_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    print(f"[Preprocessor] Loaded {len(ids)} held-out IDs")
    return ids


def load_mimic_csv(csv_path: str, held_out_ids: Set[str]) -> List[Dict]:
    """
    Load the full MIMIC-IV-BHC CSV, excluding held-out samples.

    Returns list of dicts with keys: note_id, input, target, input_tokens, target_tokens
    """
    records = []
    excluded = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading CSV"):
            note_id = row['note_id']
            if note_id in held_out_ids:
                excluded += 1
                continue

            records.append({
                'note_id': note_id,
                'input': row['input'],
                'target': row['target'],
                'input_tokens': int(row['input_tokens']),
                'target_tokens': int(row['target_tokens']),
            })

    print(f"[Preprocessor] Loaded {len(records)} records "
          f"(excluded {excluded} held-out samples)")
    return records


def apply_target_filter(
    records: List[Dict],
    min_target_tokens: int = 50,
    max_target_tokens: int = 2000,
) -> List[Dict]:
    """Filter records by target (summary) token length."""
    filtered = [
        r for r in records
        if min_target_tokens <= r['target_tokens'] <= max_target_tokens
    ]
    print(f"[Preprocessor] After target filter [{min_target_tokens}, "
          f"{max_target_tokens}]: {len(filtered)} records "
          f"(removed {len(records) - len(filtered)})")
    return filtered


def assign_range(input_tokens: int) -> str:
    """Assign a token range bucket based on input token count."""
    if input_tokens < 1024:
        return 'range_0_1k'
    elif input_tokens < 2048:
        return 'range_1k_2k'
    else:
        return 'range_2k_4k'


def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split across the 3 token ranges.
    Each range contributes proportionally to train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    # Group by range
    range_groups: Dict[str, List[Dict]] = {}
    for r in records:
        bucket = assign_range(r['input_tokens'])
        r['range'] = bucket
        if bucket not in range_groups:
            range_groups[bucket] = []
        range_groups[bucket].append(r)

    train, val, test = [], [], []

    for bucket, group in sorted(range_groups.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

        print(f"  [{bucket}] total={n}, train={n_train}, "
              f"val={n_val}, test={n - n_train - n_val}")

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"[Preprocessor] Split: train={len(train)}, "
          f"val={len(val)}, test={len(test)}")
    return train, val, test


def save_jsonl(records: List[Dict], output_path: str) -> None:
    """Save records as JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in tqdm(records, desc=f"Saving {os.path.basename(output_path)}"):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[Preprocessor] Saved {len(records)} records to {output_path}")


def _print_corruption_stats(split_name: str, total: int, logs: List[Dict]) -> None:
    """Print a summary of corruption statistics for one split."""
    n_corrupted = len(logs)
    rate = n_corrupted / total if total > 0 else 0.0

    # Count per error type
    type_counter: Counter = Counter()
    for entry in logs:
        for detail in entry.get("applied_errors", []):
            type_counter[detail.get("type", "UNKNOWN")] += 1

    print(f"\n[Stats:{split_name}] total={total}, "
          f"corrupted={n_corrupted} ({rate:.1%})")
    for etype, count in sorted(type_counter.items()):
        print(f"  {etype:<12} {count:>6} occurrences")


def apply_corruption_to_split(
    split_name: str,
    split_data: List[Dict],
    synth: ClinicalErrorSynthesizer,
    dry_run: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply synthetic corruption to every record in a split.

    Returns:
        (processed_records, corruption_log_entries)
    """
    processed_records = []
    corruption_logs = []

    for record in tqdm(split_data, desc=f"Corrupting {split_name}"):
        source_note = record['input']
        true_summary = record['target']

        result = synth.corrupt(true_summary, source_note)

        # Store both the original and corrupted versions explicitly
        record['true_summary'] = true_summary
        record['corrupted_summary'] = result.corrupted_summary

        # Keep 'input'/'target' for backward compatibility
        processed_records.append(record)

        if result.is_corrupted:
            corruption_logs.append({
                "note_id": record['note_id'],
                "split": split_name,
                "applied_errors": result.error_details,
            })

    _print_corruption_stats(split_name, len(split_data), corruption_logs)
    return processed_records, corruption_logs


def preprocess(
    csv_path: str,
    held_out_path: str,
    output_dir: str,
    min_target_tokens: int = 50,
    max_target_tokens: int = 2000,
    parquet_path: str = "data/raw/drug-dictionary/heh.parquet",
    corruption_rate: float = 0.3,
    seed: int = 42,
    dry_run: bool = False,
    dry_run_n: int = 1000,
    skip_existing: bool = True,
) -> None:
    """
    Full preprocessing pipeline with static error corruption.

    Args:
        dry_run:      If True, process only `dry_run_n` records and skip
                      writing output files (useful to sanity-check corruption).
        dry_run_n:    Number of records to process in dry-run mode.
        skip_existing: If True, skip any split whose .jsonl already exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Check which splits already exist ──────────────────────────────────
    splits_needed = []
    for split_name in ("train", "val", "test"):
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        if skip_existing and not dry_run and os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[Preprocessor] SKIP {split_name}.jsonl — already exists "
                  f"({size / 1e6:.1f} MB). Use --no-skip to overwrite.")
        else:
            splits_needed.append(split_name)

    if not splits_needed and not dry_run:
        print("[Preprocessor] All splits already exist. Nothing to do.")
        return

    # 1. Load held-out IDs
    held_out_ids = load_held_out_ids(held_out_path)

    # 2. Load CSV, excluding held-out
    records = load_mimic_csv(csv_path, held_out_ids)

    # 3. Apply target filter
    records = apply_target_filter(records, min_target_tokens, max_target_tokens)

    # 4. Dry-run: slice to first N records
    if dry_run:
        records = records[:dry_run_n]
        print(f"\n[Preprocessor] DRY RUN — processing first {len(records)} records "
              f"(no files will be written)\n")

    # 5. Stratified split
    print("[Preprocessor] Performing stratified split 80/10/10:")
    train, val, test = stratified_split(records, seed=seed)

    split_map = {"train": train, "val": val, "test": test}

    # 6. Initialize Synthesizer
    print(f"\n[Preprocessor] Initializing synthesizer (rate={corruption_rate})...")
    drug_dict = DrugDictionary(parquet_path=parquet_path, seed=seed)
    synth = ClinicalErrorSynthesizer(drug_dict, corruption_rate=corruption_rate, seed=seed)

    # 7. Apply Corruption and Save (or just print stats in dry-run)
    all_corruption_logs = []

    for split_name in ("train", "val", "test"):
        if split_name not in splits_needed and not dry_run:
            continue

        split_data = split_map[split_name]
        processed_records, corruption_logs = apply_corruption_to_split(
            split_name, split_data, synth, dry_run=dry_run
        )
        all_corruption_logs.extend(corruption_logs)

        if not dry_run:
            save_jsonl(
                processed_records,
                os.path.join(output_dir, f'{split_name}.jsonl'),
            )

    # 8. Save Corruption Log (skip in dry-run)
    if not dry_run:
        log_path = os.path.join(output_dir, 'corruption_log.jsonl')
        with open(log_path, 'w', encoding='utf-8') as f:
            for entry in tqdm(all_corruption_logs, desc="Saving corruption log"):
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\n[Preprocessor] Saved {len(all_corruption_logs)} corruption events "
              f"to {log_path}")
        print("[Preprocessor] Done!")
    else:
        total_corrupted = len(all_corruption_logs)
        total_records = sum(len(split_map[s]) for s in ("train", "val", "test"))
        print(f"\n[Preprocessor] DRY RUN complete — "
              f"{total_corrupted}/{total_records} records corrupted "
              f"({total_corrupted/total_records:.1%} effective rate)")
        print("[Preprocessor] Re-run without --dry-run to write full output.")
