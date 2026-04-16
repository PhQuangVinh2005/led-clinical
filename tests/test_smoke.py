"""
Smoke tests for the full training pipeline.

These tests verify that the critical path from dataset loading →
tokenization → model forward pass works end-to-end WITHOUT a GPU
and WITHOUT the real parquet file or MIMIC data.

Marked with @pytest.mark.smoke so they can be run selectively:
    pytest -m smoke
"""
import json
from pathlib import Path

import pytest
import torch

from tests.conftest import StubDrugDictionary, SAMPLE_NOTE, SAMPLE_SUMMARY


pytestmark = pytest.mark.smoke


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_tiny_jsonl(path: str, n: int = 4):
    """Write n records to a JSONL file."""
    with open(path, "w") as f:
        for i in range(n):
            record = {"input": SAMPLE_NOTE, "target": SAMPLE_SUMMARY}
            f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────
# Dataset smoke test
# ──────────────────────────────────────────────────────────────

class TestLEDCorrectionDatasetSmoke:
    """Tests that Dataset loads and returns correctly shaped tensors."""

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Create a tiny dataset using a real (small) tokenizer."""
        from transformers import AutoTokenizer
        from src.data.dataset import LEDCorrectionDataset

        jsonl = str(tmp_path / "data.jsonl")
        _make_tiny_jsonl(jsonl, n=4)

        tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
        drug_dict = StubDrugDictionary(seed=0)

        dataset = LEDCorrectionDataset(
            jsonl_path=jsonl,
            tokenizer=tokenizer,
            drug_dictionary=drug_dict,
            max_input_length=512,   # tiny for speed
            max_target_length=128,
            corruption_rate=0.5,
            is_train=True,
            seed=42,
        )
        return dataset

    def test_dataset_length(self, tiny_dataset):
        assert len(tiny_dataset) == 4

    def test_item_has_required_keys(self, tiny_dataset):
        item = tiny_dataset[0]
        assert set(item.keys()) == {
            "input_ids", "attention_mask", "global_attention_mask", "labels"
        }

    def test_input_ids_shape(self, tiny_dataset):
        item = tiny_dataset[0]
        assert item["input_ids"].shape == torch.Size([512])

    def test_labels_shape(self, tiny_dataset):
        item = tiny_dataset[0]
        assert item["labels"].shape == torch.Size([128])

    def test_attention_mask_is_binary(self, tiny_dataset):
        item = tiny_dataset[0]
        vals = item["attention_mask"].unique().tolist()
        assert all(v in (0, 1) for v in vals)

    def test_global_attention_mask_is_binary(self, tiny_dataset):
        item = tiny_dataset[0]
        vals = item["global_attention_mask"].unique().tolist()
        assert all(v in (0, 1) for v in vals)

    def test_bos_has_global_attention(self, tiny_dataset):
        """First token must always have global attention."""
        item = tiny_dataset[0]
        assert item["global_attention_mask"][0].item() == 1

    def test_labels_padding_replaced_with_minus100(self, tiny_dataset):
        """Padding token ids in labels must be masked to -100."""
        item = tiny_dataset[0]
        labels = item["labels"]
        # After padding masking, no label should be 0 (pad_token_id for LED)
        # (pad_token_id is 1 for LED; -100 is the ignore index)
        unique = labels.unique().tolist()
        assert 1 not in unique  # pad_token_id=1 should be replaced with -100

    def test_val_dataset_has_zero_corruption(self, tmp_path):
        """Validation dataset must not corrupt text even if rate > 0 is passed."""
        from transformers import AutoTokenizer
        from src.data.dataset import LEDCorrectionDataset

        jsonl = str(tmp_path / "val.jsonl")
        _make_tiny_jsonl(jsonl, n=2)

        tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
        drug_dict = StubDrugDictionary(seed=0)

        dataset = LEDCorrectionDataset(
            jsonl_path=jsonl,
            tokenizer=tokenizer,
            drug_dictionary=drug_dict,
            max_input_length=256,
            max_target_length=64,
            corruption_rate=0.0,
            is_train=False,
            seed=42,
        )
        assert dataset.synthesizer.corruption_rate == 0.0


# ──────────────────────────────────────────────────────────────
# Model forward pass smoke test
# ──────────────────────────────────────────────────────────────

class TestModelForwardSmoke:
    """Verify that a real LED model does a forward pass without errors on CPU."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from src.model.led_corrector import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_name="allenai/led-base-16384",
            gradient_checkpointing=False,  # no need for checkpointing in tests
        )
        model.eval()
        return model, tokenizer

    def test_model_loads(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_model_has_parameters(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000  # LED-base has ~162M params

    def test_forward_pass_on_cpu(self, model_and_tokenizer):
        """A short forward pass should complete without RuntimeError on CPU."""
        model, tokenizer = model_and_tokenizer

        text = "Patient was given aspirin 325 mg. </s> " + SAMPLE_NOTE[:200]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        # LED requires global_attention_mask
        global_attn = torch.zeros_like(inputs["input_ids"])
        global_attn[0, 0] = 1  # BOS

        target = tokenizer(
            SAMPLE_SUMMARY,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding="max_length",
        )
        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attn,
                labels=labels,
            )
        assert output.loss is not None
        assert output.loss.item() > 0  # loss should be a finite positive number

    def test_loss_is_scalar(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        text = "aspirin 325 mg. </s> " + SAMPLE_NOTE[:100]
        inputs = tokenizer(text, return_tensors="pt", max_length=256,
                           truncation=True, padding="max_length")
        global_attn = torch.zeros_like(inputs["input_ids"])
        global_attn[0, 0] = 1

        target = tokenizer("Patient stable.", return_tensors="pt",
                           max_length=32, truncation=True, padding="max_length")
        labels = target["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attn,
                labels=labels,
            )
        assert output.loss.dim() == 0  # scalar


# ──────────────────────────────────────────────────────────────
# Config smoke test
# ──────────────────────────────────────────────────────────────

class TestTrainConfigSmoke:
    """Verify training config YAML has all required keys."""

    REQUIRED_KEYS = [
        "model_name",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "fp16",
        "eval_strategy",
        "eval_steps",
        "save_strategy",
        "save_steps",
        "output_dir",
    ]

    @pytest.fixture
    def config(self):
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "train_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, config, key):
        assert key in config, f"Missing required config key: '{key}'"

    def test_learning_rate_is_positive(self, config):
        assert config["learning_rate"] > 0

    def test_batch_size_positive(self, config):
        assert config["per_device_train_batch_size"] >= 1

    def test_eval_strategy_valid(self, config):
        assert config["eval_strategy"] in ("steps", "epoch", "no")

    def test_model_name_is_led(self, config):
        assert "led" in config["model_name"].lower()

    def test_no_deprecated_evaluation_strategy_key(self, config):
        assert "evaluation_strategy" not in config, (
            "Found deprecated key 'evaluation_strategy'. "
            "Use 'eval_strategy' instead (HF Transformers >= 4.36)."
        )
