#!/usr/bin/env python3
"""
Diagnostic script — find exactly what tensor values are out of range.

Reproduces what the DataLoader would feed to the model WITHOUT needing a GPU.
Run on the server BEFORE training to confirm root cause.

Usage:
    python scripts/diagnose_data.py
    python scripts/diagnose_data.py --n-samples 50 --split val
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from transformers import AutoTokenizer, default_data_collator
from transformers import DataCollatorForSeq2Seq

VOCAB_SIZE = 50265   # expected for allenai/led-base-16384
MODEL_NAME = "allenai/led-base-16384"

# ── helpers ──────────────────────────────────────────────────────────────────

def check_tensor(name: str, t: torch.Tensor, vocab_size: int) -> list:
    """Return list of problems found in tensor t."""
    problems = []
    if name == "labels":
        # Labels must be -100 (ignore) or in [0, vocab_size)
        bad_mask = (t != -100) & ((t < 0) | (t >= vocab_size))
        if bad_mask.any():
            bad_vals = t[bad_mask].unique().tolist()
            problems.append(f"  [LABELS] {bad_mask.sum().item()} bad values: {bad_vals[:10]}")
    else:
        # input_ids / attention_mask / global_attention_mask
        if name == "input_ids":
            bad_mask = (t < 0) | (t >= vocab_size)
            if bad_mask.any():
                bad_vals = t[bad_mask].unique().tolist()
                problems.append(f"  [{name.upper()}] {bad_mask.sum().item()} bad values: {bad_vals[:10]}")
        elif name in ("attention_mask", "global_attention_mask"):
            bad_mask = ~((t == 0) | (t == 1))
            if bad_mask.any():
                bad_vals = t[bad_mask].unique().tolist()
                problems.append(f"  [{name.upper()}] non-binary values: {bad_vals[:10]}")
    return problems


def tokenize_record(record: dict, tokenizer, max_input: int, max_target: int):
    """Exactly replicate dataset.py __getitem__ tokenization."""
    source_note  = record.get('input', '')
    true_summary = record.get('true_summary', record.get('target', ''))
    corrupted    = record.get('corrupted_summary', true_summary)

    input_text  = corrupted + " </s> " + source_note
    target_text = true_summary

    enc_in = tokenizer(
        input_text, max_length=max_input, padding='max_length',
        truncation=True, return_tensors='pt',
    )
    enc_tgt = tokenizer(
        target_text, max_length=max_target, padding='max_length',
        truncation=True, return_tensors='pt',
    )

    input_ids      = enc_in['input_ids'].squeeze(0)
    attention_mask = enc_in['attention_mask'].squeeze(0)
    labels         = enc_tgt['input_ids'].squeeze(0)

    labels[labels == tokenizer.pad_token_id] = -100

    # Global attention mask
    global_attn = torch.zeros_like(input_ids)
    global_attn[0] = 1
    sep_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(sep_positions) > 0:
        global_attn[:sep_positions[0].item()] = 1

    return {
        'input_ids':            input_ids,
        'attention_mask':       attention_mask,
        'global_attention_mask': global_attn,
        'labels':               labels,
    }


def diagnose_with_collator(samples: list, tokenizer, collator_name: str, collator):
    """Collate samples and check the resulting batch for OOB values."""
    batch = collator(samples)
    print(f"\n  --- Collator: {collator_name} ---")
    problems_found = False
    for key, tensor in batch.items():
        t = tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
        probs = check_tensor(key, t, VOCAB_SIZE)
        if probs:
            problems_found = True
            for p in probs:
                print(f"  *** PROBLEM *** {p}")
        else:
            print(f"  [{key}]: shape={list(t.shape)}, min={t.min().item()}, "
                  f"max={t.max().item()} — OK")
    if not problems_found:
        print("  All tensors OK.")
    return problems_found


def main():
    parser = argparse.ArgumentParser(description="Diagnose training data for OOB tensor values")
    parser.add_argument("--data-dir",  default="data/processed")
    parser.add_argument("--split",     default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Number of records to check")
    parser.add_argument("--max-input",  type=int, default=8192)
    parser.add_argument("--max-target", type=int, default=2048)
    args = parser.parse_args()

    jsonl_path = f"{args.data_dir}/{args.split}.jsonl"
    print(f"\n{'='*70}")
    print(f"Diagnostic: {jsonl_path}  (first {args.n_samples} records)")
    print(f"{'='*70}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  len(tokenizer) = {len(tokenizer)}")
    print(f"  pad_token_id   = {tokenizer.pad_token_id}")
    print(f"  eos_token_id   = {tokenizer.eos_token_id}")
    print(f"  bos_token_id   = {tokenizer.bos_token_id}")

    # Load records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if len(records) >= args.n_samples:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\nLoaded {len(records)} records.")
    print(f"Has 'corrupted_summary': {'corrupted_summary' in records[0]}")

    # Tokenize all records
    print(f"\nTokenizing {len(records)} records...")
    samples = []
    for i, rec in enumerate(records):
        s = tokenize_record(rec, tokenizer, args.max_input, args.max_target)
        samples.append(s)
        # Quick per-sample check
        for key, t in s.items():
            probs = check_tensor(key, t, VOCAB_SIZE)
            if probs:
                print(f"  [Record {i}] NOTE_ID={rec.get('note_id','?')}")
                for p in probs:
                    print(f"    {p}")

    print(f"\nAll {len(records)} records tokenized.\n")

    # ── Test 1: torch default_data_collator ──────────────────────────────────
    any_problem = False
    any_problem |= diagnose_with_collator(
        samples, tokenizer,
        "default_data_collator (torch stacking)",
        default_data_collator
    )

    # ── Test 2: DataCollatorForSeq2Seq (what Seq2SeqTrainer auto-creates) ────
    dcs2s = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    # DataCollatorForSeq2Seq expects list-of-dicts with list values (not tensors)
    # Convert tensors -> lists to simulate what happens when __getitem__ returns lists
    samples_as_lists = [
        {k: v.tolist() for k, v in s.items()} for s in samples
    ]
    any_problem |= diagnose_with_collator(
        samples_as_lists, tokenizer,
        "DataCollatorForSeq2Seq (Seq2SeqTrainer auto-creates this)",
        dcs2s
    )

    # ── Test 3: DataCollatorForSeq2Seq with tensors (actual call site) ───────
    any_problem |= diagnose_with_collator(
        samples, tokenizer,
        "DataCollatorForSeq2Seq with raw tensors (actual Trainer behavior)",
        dcs2s
    )

    print(f"\n{'='*70}")
    if any_problem:
        print("*** PROBLEMS FOUND — see above for details ***")
    else:
        print("All checks passed. Data tensors look valid.")
    print(f"{'='*70}\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    all_labels = torch.stack([s['labels'] for s in samples])
    all_input  = torch.stack([s['input_ids'] for s in samples])
    print("Summary stats (from raw tokenized tensors):")
    print(f"  input_ids : min={all_input.min().item():6d}  max={all_input.max().item():6d}")
    print(f"  labels    : min={all_labels.min().item():6d}  max={all_labels.max().item():6d}")
    n_minus100 = (all_labels == -100).sum().item()
    n_valid    = ((all_labels >= 0) & (all_labels < VOCAB_SIZE)).sum().item()
    n_other    = all_labels.numel() - n_minus100 - n_valid
    print(f"  labels breakdown: -100={n_minus100:,}  valid={n_valid:,}  other={n_other:,}")
    if n_other > 0:
        other_vals = all_labels[(all_labels != -100) & ~((all_labels >= 0) & (all_labels < VOCAB_SIZE))].unique().tolist()
        print(f"  *** OTHER values: {other_vals[:20]}")


if __name__ == "__main__":
    main()
