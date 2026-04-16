"""
Preprocessor — Load MIMIC-IV-BHC data, exclude held-out samples,
apply target filters, and create stratified train/val/test splits.

Key features:
  - Streaming writes: flushes every `write_every` records so progress
    is never lost to a crash.
  - Crash resume: counts existing lines in the output JSONL and skips
    those records, then appends from where it left off.
  - Per-split corruption statistics printed after each split.
  - --dry-run: first N records only, no files written.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Stratified split across the 3 token-range buckets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    range_groups: Dict[str, List[Dict]] = {}
    for r in records:
        bucket = assign_range(r['input_tokens'])
        r['range'] = bucket
        range_groups.setdefault(bucket, []).append(r)

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

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"[Preprocessor] Split: train={len(train)}, "
          f"val={len(val)}, test={len(test)}")
    return train, val, test


def _count_jsonl_lines(path: str) -> int:
    """Count non-empty lines in an existing JSONL file (fast resume check)."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _print_corruption_stats(split_name: str, total: int, n_corrupted: int,
                             type_counter: Counter) -> None:
    """Pretty-print per-split corruption summary."""
    rate = n_corrupted / total if total > 0 else 0.0
    print(f"\n[Stats:{split_name}] total={total:,}, "
          f"corrupted={n_corrupted:,} ({rate:.1%})")
    for etype, count in sorted(type_counter.items()):
        print(f"  {etype:<14} {count:>7,} occurrences")


# ── Core streaming processor ──────────────────────────────────────────────────

def process_split_streaming(
    split_name: str,
    split_data: List[Dict],
    synth: ClinicalErrorSynthesizer,
    output_path: str,
    log_path: str,
    write_every: int = 5000,
    dry_run: bool = False,
) -> int:
    """
    Process one split with streaming writes and crash-resume support.

    Strategy:
      1. Count existing lines in `output_path` → that many records are already done.
      2. Skip those records in `split_data`.
      3. Open `output_path` in *append* mode and write every `write_every` records.
      4. Append matching entries to `log_path` in real time.

    Returns:
        Number of newly processed records (not counting resumed ones).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Resume: find how many records already written ─────────────────────
    already_done = 0 if dry_run else _count_jsonl_lines(output_path)
    remaining = split_data[already_done:]

    if already_done > 0:
        print(f"[{split_name}] Resuming — skipping {already_done:,} already-processed records, "
              f"{len(remaining):,} remaining.")
    else:
        print(f"[{split_name}] Starting fresh — {len(remaining):,} records.")

    if not remaining:
        print(f"[{split_name}] Already complete. Nothing to do.")
        return 0

    type_counter: Counter = Counter()
    n_corrupted = 0
    buffer: List[str] = []          # JSON lines to flush
    log_buffer: List[str] = []      # log entries to flush

    file_mode = 'a' if (already_done > 0 and not dry_run) else ('w' if not dry_run else None)

    out_f  = open(output_path, file_mode, encoding='utf-8') if not dry_run else None
    log_f  = open(log_path,    'a',       encoding='utf-8') if not dry_run else None

    try:
        pbar = tqdm(remaining, desc=f"Corrupting {split_name}",
                    total=len(remaining), unit="rec")

        for i, record in enumerate(pbar, 1):
            source_note  = record['input']
            true_summary = record['target']

            result = synth.corrupt(true_summary, source_note)

            record['true_summary']      = true_summary
            record['corrupted_summary'] = result.corrupted_summary

            if not dry_run:
                buffer.append(json.dumps(record, ensure_ascii=False))

            if result.is_corrupted:
                n_corrupted += 1
                for detail in result.error_details:
                    type_counter[detail.get('type', 'UNKNOWN')] += 1

                log_entry = {
                    'note_id':        record['note_id'],
                    'split':          split_name,
                    'applied_errors': result.error_details,
                }
                if not dry_run:
                    log_buffer.append(json.dumps(log_entry, ensure_ascii=False))

            # ── Flush every write_every records ───────────────────────────
            if not dry_run and i % write_every == 0:
                out_f.write('\n'.join(buffer) + '\n')
                out_f.flush()
                buffer.clear()

                if log_buffer:
                    log_f.write('\n'.join(log_buffer) + '\n')
                    log_f.flush()
                    log_buffer.clear()

                pbar.set_postfix(flushed=already_done + i, corrupted=n_corrupted)

        # ── Final flush ───────────────────────────────────────────────────
        if not dry_run and buffer:
            out_f.write('\n'.join(buffer) + '\n')
            out_f.flush()
            buffer.clear()

        if not dry_run and log_buffer:
            log_f.write('\n'.join(log_buffer) + '\n')
            log_f.flush()
            log_buffer.clear()

    finally:
        if out_f:
            out_f.close()
        if log_f:
            log_f.close()

    # ── Stats ─────────────────────────────────────────────────────────────
    total_in_split = already_done + len(remaining)
    _print_corruption_stats(split_name, total_in_split, n_corrupted, type_counter)

    return len(remaining)


# ── Public entry point ────────────────────────────────────────────────────────

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
    write_every: int = 5000,
) -> None:
    """
    Full preprocessing pipeline with streaming writes and crash-resume.

    Args:
        dry_run:     Process only `dry_run_n` records; print stats; write nothing.
        dry_run_n:   Records to process in dry-run mode.
        write_every: Flush output to disk every this many records (default 5000).
                     Lower = safer against crashes; higher = faster I/O.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'corruption_log.jsonl')

    # 1. Load held-out IDs
    held_out_ids = load_held_out_ids(held_out_path)

    # 2. Load CSV
    records = load_mimic_csv(csv_path, held_out_ids)

    # 3. Target filter
    records = apply_target_filter(records, min_target_tokens, max_target_tokens)

    # 4. Dry-run slice
    if dry_run:
        records = records[:dry_run_n]
        print(f"\n[Preprocessor] DRY RUN — {len(records):,} records "
              f"(no files will be written)\n")

    # 5. Stratified split
    print("[Preprocessor] Performing stratified split 80/10/10:")
    train, val, test = stratified_split(records, seed=seed)

    # 6. Synthesizer
    print(f"\n[Preprocessor] Initializing synthesizer (rate={corruption_rate})...")
    drug_dict = DrugDictionary(parquet_path=parquet_path, seed=seed)
    synth = ClinicalErrorSynthesizer(drug_dict, corruption_rate=corruption_rate, seed=seed)

    # 7. Process each split (streaming + resume)
    print(f"\n[Preprocessor] write_every={write_every:,} "
          f"(flushing to disk every {write_every:,} records)\n")

    total_new = 0
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        n = process_split_streaming(
            split_name=split_name,
            split_data=split_data,
            synth=synth,
            output_path=out_path,
            log_path=log_path,
            write_every=write_every,
            dry_run=dry_run,
        )
        total_new += n

    if dry_run:
        print(f"\n[Preprocessor] DRY RUN complete — "
              f"processed {total_new:,} records. Re-run without --dry-run to write output.")
    else:
        print(f"\n[Preprocessor] Done! {total_new:,} new records written across all splits.")
        print(f"[Preprocessor] Corruption log → {log_path}")
