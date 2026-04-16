"""
Shared pytest fixtures for led-clinical-corrector test suite.

The key challenge: DrugDictionary normally loads data/raw/drug-dictionary/heh.parquet.
We mock that out so unit tests run in CI without the 200MB file.
"""
import json
import os
import tempfile
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# Stub DrugDictionary (no parquet I/O)
# ──────────────────────────────────────────────────────────────

class StubDrugDictionary:
    """Minimal in-memory drug dictionary for tests."""

    def __init__(self, drugs: Optional[List[str]] = None, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.all_drug_names: Set[str] = set(drugs or [
            "metformin", "lisinopril", "atorvastatin", "aspirin",
            "warfarin", "furosemide", "metoprolol", "amlodipine",
            "omeprazole", "simvastatin",
        ])
        self.name_to_id: Dict[str, str] = {n: f"D{i}" for i, n in enumerate(self.all_drug_names)}
        self.id_to_name: Dict[str, str] = {v: k for k, v in self.name_to_id.items()}
        self.source_to_names: Dict[str, List[str]] = {
            "Drug::Brand": list(self.all_drug_names)
        }

    def find_drugs_in_text(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for name in self.all_drug_names:
            idx = text_lower.find(name)
            if idx != -1:
                before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
                after_idx = idx + len(name)
                after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
                if before_ok and after_ok:
                    found.append(name)
        return found

    def get_random_substitute(self, original_name: str) -> Optional[str]:
        candidates = list(self.all_drug_names - {original_name.lower()})
        return self.rng.choice(candidates) if candidates else None

    def get_any_random_drug(self) -> str:
        return self.rng.choice(list(self.all_drug_names))


@pytest.fixture
def drug_dict():
    """Stub drug dictionary for unit tests (no I/O)."""
    return StubDrugDictionary()


# ──────────────────────────────────────────────────────────────
# Sample clinical texts
# ──────────────────────────────────────────────────────────────

SAMPLE_SUMMARY = (
    "Patient was admitted with chest pain. "
    "She was started on aspirin 325 mg and lisinopril 10 mg daily. "
    "Patient had hypertension and denied shortness of breath. "
    "She was discharged on POD #3 in stable condition."
)

SAMPLE_NOTE = (
    "<MAJOR SURGICAL OR INVASIVE PROCEDURE>coronary angiography, percutaneous coronary intervention</>\n"
    "Patient underwent coronary angiography and percutaneous coronary intervention. "
    "She was admitted on 01/15/2024 and discharged on 01/18/2024. "
    "HD #2: vitals stable. "
    "Patient developed hypertension. "
    "She was started on aspirin 325 mg."
)


@pytest.fixture
def sample_summary():
    return SAMPLE_SUMMARY


@pytest.fixture
def sample_note():
    return SAMPLE_NOTE


# ──────────────────────────────────────────────────────────────
# Temporary JSONL file fixture
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_jsonl(tmp_path):
    """Creates a small temporary JSONL dataset file."""
    records = [
        {"input": SAMPLE_NOTE, "target": SAMPLE_SUMMARY},
        {"input": SAMPLE_NOTE + " Additional text.", "target": SAMPLE_SUMMARY + " Follow-up needed."},
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)
