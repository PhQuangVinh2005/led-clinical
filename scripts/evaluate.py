#!/usr/bin/env python3
"""
CLI script for evaluating the trained LED corrector.

Usage:
    python scripts/evaluate.py --checkpoint outputs/led-corrector-v1/final
    python scripts/evaluate.py --checkpoint outputs/led-corrector-v1/final --split test
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, LEDForConditionalGeneration

from src.data.drug_dictionary import DrugDictionary
from src.data.error_synthesizer import ClinicalErrorSynthesizer
from src.utils.metrics import compute_rouge_metrics, compute_correction_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_corrections(
    model: LEDForConditionalGeneration,
    tokenizer: AutoTokenizer,
    input_texts: List[str],
    max_input_length: int = 8192,
    max_output_length: int = 2048,
    batch_size: int = 2,
    num_beams: int = 4,
) -> List[str]:
    """Generate corrected summaries from the model."""
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(input_texts), batch_size), desc="Generating"):
        batch_texts = input_texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(model.device)

        # Set global attention on first token
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                global_attention_mask=global_attention_mask,
                max_length=max_output_length,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate LED clinical corrector")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed JSONL files")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation results (JSON)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples to evaluate (for debugging)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for generation")
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Number of beams for generation")
    parser.add_argument("--corruption-rate", type=float, default=0.3,
                        help="Corruption rate for synthetic errors")

    args = parser.parse_args()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = LEDForConditionalGeneration.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load drug dictionary and synthesizer
    drug_dict = DrugDictionary(
        parquet_path="data/raw/drug-dictionary/heh.parquet",
        seed=42,
    )
    synthesizer = ClinicalErrorSynthesizer(
        drug_dictionary=drug_dict,
        corruption_rate=args.corruption_rate,
        seed=42,
    )

    # Load data
    jsonl_path = os.path.join(args.data_dir, f"{args.split}.jsonl")
    logger.info(f"Loading data from {jsonl_path}")
    records = load_jsonl(jsonl_path)

    if args.max_samples:
        records = records[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # Apply corruption and build inputs
    input_texts = []
    references = []
    corrupted_summaries = []
    corruption_info = []

    for record in tqdm(records, desc="Corrupting"):
        source_note = record['input']
        true_summary = record['target']

        result = synthesizer.corrupt(true_summary, source_note)

        input_text = result.corrupted_summary + " </s> " + source_note
        input_texts.append(input_text)
        references.append(true_summary)
        corrupted_summaries.append(result.corrupted_summary)
        corruption_info.append({
            'is_corrupted': result.is_corrupted,
            'error_types': result.error_types,
        })

    # Generate corrections
    logger.info("Generating corrections...")
    predictions = generate_corrections(
        model=model,
        tokenizer=tokenizer,
        input_texts=input_texts,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )

    # Compute metrics
    logger.info("Computing metrics...")

    # Overall ROUGE
    rouge_scores = compute_rouge_metrics(predictions, references)
    logger.info(f"ROUGE scores: {rouge_scores}")

    # Correction rate
    correction_scores = compute_correction_rate(
        predictions, corrupted_summaries, references
    )
    logger.info(f"Correction scores: {correction_scores}")

    # Per-error-type analysis
    error_type_stats = {}
    for info in corruption_info:
        for et in info['error_types']:
            if et not in error_type_stats:
                error_type_stats[et] = 0
            error_type_stats[et] += 1
    logger.info(f"Error type distribution: {error_type_stats}")

    # Corrupted vs uncorrupted split
    corrupted_indices = [i for i, info in enumerate(corruption_info) if info['is_corrupted']]
    clean_indices = [i for i, info in enumerate(corruption_info) if not info['is_corrupted']]

    if corrupted_indices:
        corrupted_rouge = compute_rouge_metrics(
            [predictions[i] for i in corrupted_indices],
            [references[i] for i in corrupted_indices],
        )
        logger.info(f"Corrupted samples ROUGE ({len(corrupted_indices)} samples): {corrupted_rouge}")
    else:
        corrupted_rouge = {}

    if clean_indices:
        clean_rouge = compute_rouge_metrics(
            [predictions[i] for i in clean_indices],
            [references[i] for i in clean_indices],
        )
        logger.info(f"Clean samples ROUGE ({len(clean_indices)} samples): {clean_rouge}")
    else:
        clean_rouge = {}

    # Save results
    results = {
        'split': args.split,
        'n_samples': len(records),
        'n_corrupted': len(corrupted_indices),
        'n_clean': len(clean_indices),
        'overall_rouge': rouge_scores,
        'correction_rate': correction_scores,
        'error_type_distribution': error_type_stats,
        'corrupted_rouge': corrupted_rouge,
        'clean_rouge': clean_rouge,
    }

    output_file = args.output_file or os.path.join(
        args.checkpoint, f"eval_{args.split}_results.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
