"""
PyTorch Dataset for LED fine-tuning on clinical error correction.

Loads preprocessed JSONL data, applies synthetic error corruption on-the-fly
during training, and tokenizes for LED input format:
  Input:  [corrupted_summary] </s> [source_clinical_note]
  Target: [true_summary]
"""

import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.drug_dictionary import DrugDictionary
from src.data.error_synthesizer import ClinicalErrorSynthesizer


class LEDCorrectionDataset(Dataset):
    """
    Dataset that loads JSONL records and produces tokenized samples
    for LED conditional generation training.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        drug_dictionary: DrugDictionary,
        max_input_length: int = 8192,
        max_target_length: int = 2048,
        corruption_rate: float = 0.3,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.is_train = is_train

        # Load records
        self.records = self._load_jsonl(jsonl_path)

        # Error synthesizer (only applied during training)
        self.synthesizer = ClinicalErrorSynthesizer(
            drug_dictionary=drug_dictionary,
            corruption_rate=corruption_rate if is_train else 0.0,
            seed=seed,
        )

        print(f"[Dataset] Loaded {len(self.records)} records from {jsonl_path} "
              f"(train={is_train}, corruption_rate={corruption_rate if is_train else 0.0})")

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        """Load all records from a JSONL file."""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        source_note = record.get('input', '')
        # Priority 1: Use pre-applied corruption from preprocessing
        if 'corrupted_summary' in record and 'true_summary' in record:
            corrupted_summary = record['corrupted_summary']
            true_summary = record['true_summary']
        else:
            # Priority 2: Fallback to on-the-fly corruption (legacy/development)
            source_note = record.get('input', '')
            true_summary = record.get('target', '')
            result = self.synthesizer.corrupt(true_summary, source_note)
            corrupted_summary = result.corrupted_summary

        # Build input: [corrupted_summary] </s> [source_note]
        input_text = corrupted_summary + " </s> " + source_note
        target_text = true_summary

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = input_encoding['input_ids'].squeeze(0)
        attention_mask = input_encoding['attention_mask'].squeeze(0)
        labels = target_encoding['input_ids'].squeeze(0)

        # Replace padding token IDs in labels with -100 (ignored by loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Build global attention mask for LED:
        # Global attention on first token (BOS) + summary tokens before </s>
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1  # BOS token gets global attention

        # Find </s> separator position and set global attention on summary tokens
        sep_token_id = self.tokenizer.eos_token_id
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            first_sep = sep_positions[0].item()
            # Set global attention on all summary tokens (before first </s>)
            global_attention_mask[:first_sep] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask,
            'labels': labels,
        }
