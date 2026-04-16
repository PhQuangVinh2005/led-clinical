"""
Preprocessor — Load MIMIC-IV-BHC data, exclude held-out samples,
apply target filters, and create stratified train/val/test splits.
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


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
        # Use a list for tqdm to get a progress bar if we know the size, 
        # but since it's a reader, we'll just wrap it.
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


def preprocess(
    csv_path: str,
    held_out_path: str,
    output_dir: str,
    min_target_tokens: int = 50,
    max_target_tokens: int = 2000,
    seed: int = 42,
) -> None:
    """Full preprocessing pipeline."""
    # 1. Load held-out IDs
    held_out_ids = load_held_out_ids(held_out_path)

    # 2. Load CSV, excluding held-out
    records = load_mimic_csv(csv_path, held_out_ids)

    # 3. Apply target filter
    records = apply_target_filter(records, min_target_tokens, max_target_tokens)

    # 4. Stratified split
    print("[Preprocessor] Performing stratified split 80/10/10:")
    train, val, test = stratified_split(records, seed=seed)

    # 5. Save
    save_jsonl(train, os.path.join(output_dir, 'train.jsonl'))
    save_jsonl(val, os.path.join(output_dir, 'val.jsonl'))
    save_jsonl(test, os.path.join(output_dir, 'test.jsonl'))

    print("[Preprocessor] Done!")
