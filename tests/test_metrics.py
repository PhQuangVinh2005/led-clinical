"""
Unit tests for src/utils/metrics.py.

Mocks the `evaluate` library so tests don't download any HuggingFace
datasets and run instantly in CI.
"""
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_rouge_mock(rouge1=0.8, rouge2=0.6, rougeL=0.75):
    """Return a mock rouge object that compute() returns fixed scores."""
    rouge = MagicMock()
    rouge.compute.return_value = {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }
    return rouge


# ──────────────────────────────────────────────────────────────
# compute_rouge_metrics
# ──────────────────────────────────────────────────────────────

class TestComputeRougeMetrics:
    @patch("src.utils.metrics.evaluate.load")
    def test_returns_three_keys(self, mock_load):
        mock_load.return_value = _make_rouge_mock()
        from src.utils.metrics import compute_rouge_metrics
        result = compute_rouge_metrics(["pred"], ["ref"])
        assert set(result.keys()) == {"rouge1", "rouge2", "rougeL"}

    @patch("src.utils.metrics.evaluate.load")
    def test_values_are_floats(self, mock_load):
        mock_load.return_value = _make_rouge_mock(0.5, 0.3, 0.45)
        from src.utils.metrics import compute_rouge_metrics
        result = compute_rouge_metrics(["pred"], ["ref"])
        for v in result.values():
            assert isinstance(v, float)

    @patch("src.utils.metrics.evaluate.load")
    def test_passes_use_stemmer(self, mock_load):
        rouge_mock = _make_rouge_mock()
        mock_load.return_value = rouge_mock
        from src.utils.metrics import compute_rouge_metrics
        compute_rouge_metrics(["a"], ["b"])
        _, kwargs = rouge_mock.compute.call_args
        assert kwargs.get("use_stemmer") is True

    @patch("src.utils.metrics.evaluate.load")
    def test_multiple_predictions(self, mock_load):
        mock_load.return_value = _make_rouge_mock()
        from src.utils.metrics import compute_rouge_metrics
        preds = ["pred one", "pred two", "pred three"]
        refs = ["ref one", "ref two", "ref three"]
        result = compute_rouge_metrics(preds, refs)
        assert "rouge1" in result
        rouge_mock = mock_load.return_value
        rouge_mock.compute.assert_called_once()

    @patch("src.utils.metrics.evaluate.load")
    def test_scores_propagated_correctly(self, mock_load):
        mock_load.return_value = _make_rouge_mock(0.9, 0.7, 0.85)
        from src.utils.metrics import compute_rouge_metrics
        result = compute_rouge_metrics(["p"], ["r"])
        assert result["rouge1"] == pytest.approx(0.9)
        assert result["rouge2"] == pytest.approx(0.7)
        assert result["rougeL"] == pytest.approx(0.85)


# ──────────────────────────────────────────────────────────────
# compute_correction_rate
# ──────────────────────────────────────────────────────────────

class TestComputeCorrectionRate:
    def _patch_rouge(self, pred_score: float, corrupt_score: float):
        """Return a mock that returns pred_score for first call, corrupt_score for second."""
        rouge = MagicMock()
        rouge.compute.side_effect = [
            {"rougeL": pred_score},
            {"rougeL": corrupt_score},
        ] * 10  # enough for multiple samples
        return rouge

    @patch("src.utils.metrics.evaluate.load")
    def test_returns_two_keys(self, mock_load):
        mock_load.return_value = self._patch_rouge(0.9, 0.5)
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(["pred"], ["corrupt"], ["ref"])
        assert set(result.keys()) == {"exact_match_rate", "improvement_rate"}

    @patch("src.utils.metrics.evaluate.load")
    def test_exact_match_when_identical(self, mock_load):
        mock_load.return_value = self._patch_rouge(1.0, 0.5)
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(
            predictions=["same text"],
            corrupted_inputs=["wrong text"],
            references=["same text"],
        )
        assert result["exact_match_rate"] == pytest.approx(1.0)

    @patch("src.utils.metrics.evaluate.load")
    def test_no_exact_match_when_different(self, mock_load):
        mock_load.return_value = self._patch_rouge(0.8, 0.5)
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(
            predictions=["predicted"],
            corrupted_inputs=["corrupt"],
            references=["reference"],
        )
        assert result["exact_match_rate"] == pytest.approx(0.0)

    @patch("src.utils.metrics.evaluate.load")
    def test_improvement_rate_when_pred_better_than_corrupt(self, mock_load):
        # pred_score > corrupt_score → improvement
        mock_load.return_value = self._patch_rouge(0.9, 0.4)
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(["pred"], ["corrupt"], ["ref"])
        assert result["improvement_rate"] == pytest.approx(1.0)

    @patch("src.utils.metrics.evaluate.load")
    def test_no_improvement_when_pred_worse(self, mock_load):
        # pred_score < corrupt_score → no improvement
        mock_load.return_value = self._patch_rouge(0.3, 0.8)
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(["pred"], ["corrupt"], ["ref"])
        assert result["improvement_rate"] == pytest.approx(0.0)

    @patch("src.utils.metrics.evaluate.load")
    def test_empty_inputs_returns_zeros(self, mock_load):
        mock_load.return_value = _make_rouge_mock()
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate([], [], [])
        assert result["exact_match_rate"] == 0.0
        assert result["improvement_rate"] == 0.0

    @patch("src.utils.metrics.evaluate.load")
    def test_mixed_batch(self, mock_load):
        """2 samples: one improves, one doesn't → improvement_rate = 0.5."""
        rouge = MagicMock()
        # Sample 1: pred 0.9 > corrupt 0.5 → improvement
        # Sample 2: pred 0.3 < corrupt 0.8 → no improvement
        rouge.compute.side_effect = [
            {"rougeL": 0.9}, {"rougeL": 0.5},  # sample 1
            {"rougeL": 0.3}, {"rougeL": 0.8},  # sample 2
        ]
        mock_load.return_value = rouge
        from src.utils.metrics import compute_correction_rate
        result = compute_correction_rate(
            predictions=["good pred", "bad pred"],
            corrupted_inputs=["corrupt1", "corrupt2"],
            references=["ref1", "ref2"],
        )
        assert result["improvement_rate"] == pytest.approx(0.5)
