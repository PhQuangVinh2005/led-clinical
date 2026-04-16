"""
Metrics module for evaluation.

Provides ROUGE and per-error-type correction rate metrics.
"""

import evaluate
import numpy as np
from typing import Dict, List


def compute_rouge_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics.

    Args:
        predictions: Model-generated summaries
        references: Ground-truth summaries

    Returns:
        Dict with rouge1, rouge2, rougeL scores
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL'],
    }


def compute_correction_rate(
    predictions: List[str],
    corrupted_inputs: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute how well the model corrects errors.

    - exact_match: fraction of predictions that exactly match reference
    - improvement_rate: fraction where prediction is closer to reference
                        than the corrupted input was
    """
    rouge = evaluate.load("rouge")

    exact_matches = 0
    improvements = 0
    total = len(predictions)

    for pred, corrupt, ref in zip(predictions, corrupted_inputs, references):
        # Exact match
        if pred.strip() == ref.strip():
            exact_matches += 1

        # Improvement: compare ROUGE-L of pred vs corrupt against reference
        pred_score = rouge.compute(
            predictions=[pred], references=[ref], use_stemmer=True
        )['rougeL']
        corrupt_score = rouge.compute(
            predictions=[corrupt], references=[ref], use_stemmer=True
        )['rougeL']

        if pred_score > corrupt_score:
            improvements += 1

    return {
        'exact_match_rate': exact_matches / total if total > 0 else 0.0,
        'improvement_rate': improvements / total if total > 0 else 0.0,
    }
