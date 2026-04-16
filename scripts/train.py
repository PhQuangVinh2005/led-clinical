#!/usr/bin/env python3
"""
CLI script for LED fine-tuning with OOM protection.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    
    # Resume from crash:
    python scripts/train.py --config configs/train_config.yaml --resume

Environment:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.dataset import LEDCorrectionDataset
from src.data.drug_dictionary import DrugDictionary
from src.model.led_corrector import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage periodically to detect leaks."""

    def __init__(self, log_every_n_steps: int = 500):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(
                f"Step {state.global_step} | GPU Memory: "
                f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
                f"max_allocated={max_allocated:.2f}GB"
            )

    def on_evaluate(self, args, state, control, **kwargs):
        """Clear cache after evaluation to free unused memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after evaluation")


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LED for clinical error correction")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set CUDA memory config
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config['model_name'],
        gradient_checkpointing=config.get('gradient_checkpointing', True),
    )

    # Load drug dictionary
    drug_dict = DrugDictionary(
        parquet_path="data/raw/drug-dictionary/heh.parquet",
        seed=config.get('seed', 42),
    )

    # Load datasets
    data_dir = config.get('data_dir', 'data/processed')
    train_dataset = LEDCorrectionDataset(
        jsonl_path=os.path.join(data_dir, 'train.jsonl'),
        tokenizer=tokenizer,
        drug_dictionary=drug_dict,
        max_input_length=config.get('max_input_length', 8192),
        max_target_length=config.get('max_target_length', 2048),
        corruption_rate=config.get('corruption_rate', 0.3),
        is_train=True,
        seed=config.get('seed', 42),
    )

    val_dataset = LEDCorrectionDataset(
        jsonl_path=os.path.join(data_dir, 'val.jsonl'),
        tokenizer=tokenizer,
        drug_dictionary=drug_dict,
        max_input_length=config.get('max_input_length', 8192),
        max_target_length=config.get('max_target_length', 2048),
        corruption_rate=0.0,  # No corruption during validation
        is_train=False,
        seed=config.get('seed', 42),
    )

    # Training arguments
    output_dir = config.get('output_dir', 'outputs/led-corrector-v1')
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('num_train_epochs', 10),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 2),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
        learning_rate=config.get('learning_rate', 3e-5),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        weight_decay=config.get('weight_decay', 0.01),
        fp16=config.get('fp16', True),
        eval_strategy=config.get('eval_strategy', config.get('evaluation_strategy', 'steps')),
        eval_steps=config.get('eval_steps', 2000),
        save_strategy=config.get('save_strategy', 'steps'),
        save_steps=config.get('save_steps', 2000),
        save_total_limit=config.get('save_total_limit', 5),
        load_best_model_at_end=config.get('load_best_model_at_end', True),
        metric_for_best_model=config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=config.get('greater_is_better', False),
        logging_steps=config.get('logging_steps', 100),
        report_to=config.get('report_to', 'none'),
        seed=config.get('seed', 42),
        predict_with_generate=False,  # Use loss for evaluation, not generation
        remove_unused_columns=False,  # Keep global_attention_mask
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[GPUMemoryCallback(log_every_n_steps=500)],
    )

    # Determine resume checkpoint
    resume_from = None
    if args.resume:
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            resume_from = last_ckpt
            logger.info(f"Resuming from checkpoint: {resume_from}")
        else:
            logger.warning("No checkpoint found, starting from scratch")

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
