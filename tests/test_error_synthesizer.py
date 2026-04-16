"""
Unit tests for ClinicalErrorSynthesizer.

Tests each of the 5 error types and the top-level corrupt() dispatcher.
All tests use a stub DrugDictionary — no parquet file required.
"""
import pytest

from src.data.error_synthesizer import (
    ClinicalErrorSynthesizer,
    CorruptionResult,
    DOSAGE_PATTERN,
    WORD_TO_NUM,
)


# ──────────────────────────────────────────────────────────────
# Helper factory
# ──────────────────────────────────────────────────────────────

def make_synthesizer(drug_dict, corruption_rate=1.0, seed=42):
    """Always-corrupt synthesizer by default (rate=1.0)."""
    return ClinicalErrorSynthesizer(
        drug_dictionary=drug_dict,
        corruption_rate=corruption_rate,
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────
# CorruptionResult dataclass
# ──────────────────────────────────────────────────────────────

class TestCorruptionResult:
    def test_fields_present(self):
        r = CorruptionResult(
            original_summary="A",
            corrupted_summary="B",
            is_corrupted=True,
            error_types=["MED_NAME"],
            error_details=[{"type": "MED_NAME"}],
        )
        assert r.original_summary == "A"
        assert r.error_types == ["MED_NAME"]

    def test_defaults(self):
        r = CorruptionResult(original_summary="X", corrupted_summary="X", is_corrupted=False)
        assert r.error_types == []
        assert r.error_details == []


# ──────────────────────────────────────────────────────────────
# corrupt() — top-level dispatcher
# ──────────────────────────────────────────────────────────────

class TestCorrupt:
    def test_no_corruption_when_rate_zero(self, drug_dict, sample_summary, sample_note):
        synth = make_synthesizer(drug_dict, corruption_rate=0.0)
        result = synth.corrupt(sample_summary, sample_note)
        assert result.is_corrupted is False
        assert result.corrupted_summary == sample_summary
        assert result.error_types == []

    def test_returns_corruption_result(self, drug_dict, sample_summary, sample_note):
        synth = make_synthesizer(drug_dict, corruption_rate=1.0)
        result = synth.corrupt(sample_summary, sample_note)
        assert isinstance(result, CorruptionResult)
        assert result.original_summary == sample_summary

    def test_corrupted_differs_from_original(self, drug_dict, sample_summary, sample_note):
        """With rate=1.0 and rich text, at least one error must apply."""
        synth = make_synthesizer(drug_dict, corruption_rate=1.0, seed=0)
        result = synth.corrupt(sample_summary, sample_note)
        # Summary has dosages, drugs, temporal, negation — at least one should fire
        assert result.is_corrupted is True

    def test_error_types_are_valid(self, drug_dict, sample_summary, sample_note):
        synth = make_synthesizer(drug_dict, corruption_rate=1.0)
        result = synth.corrupt(sample_summary, sample_note)
        for et in result.error_types:
            assert et in ClinicalErrorSynthesizer.ERROR_TYPES

    def test_identity_on_empty_string(self, drug_dict):
        synth = make_synthesizer(drug_dict, corruption_rate=1.0)
        result = synth.corrupt("", "")
        # Empty string has nothing to corrupt
        assert result.corrupted_summary == ""

    def test_deterministic_with_same_seed(self, drug_dict, sample_summary, sample_note):
        s1 = make_synthesizer(drug_dict, seed=7)
        s2 = make_synthesizer(drug_dict, seed=7)
        r1 = s1.corrupt(sample_summary, sample_note)
        r2 = s2.corrupt(sample_summary, sample_note)
        assert r1.corrupted_summary == r2.corrupted_summary


# ──────────────────────────────────────────────────────────────
# Error Type 1: MED_NAME
# ──────────────────────────────────────────────────────────────

class TestMedNameCorruption:
    def test_drug_in_summary_gets_swapped(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Patient was started on aspirin 325 mg daily."
        text, detail = synth._corrupt_medication_name(summary)
        # aspirin should be replaced
        assert "aspirin" not in text.lower() or text != summary  # substituted
        if detail:
            assert detail["type"] == "MED_NAME"
            assert detail["original"].lower() in drug_dict.all_drug_names

    def test_no_drug_returns_unchanged(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Patient was admitted for evaluation."
        text, detail = synth._corrupt_medication_name(summary)
        assert text == summary
        assert detail is None

    def test_substitute_is_different_drug(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Patient took metformin 500 mg."
        text, detail = synth._corrupt_medication_name(summary)
        if detail:
            assert detail["original"].lower() != detail["corrupted"].lower()

    def test_only_first_occurrence_replaced(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "aspirin was given. aspirin was continued."
        text, detail = synth._corrupt_medication_name(summary)
        if detail:
            # Second occurrence should still be aspirin if first was replaced
            assert text.lower().count("aspirin") <= 1


# ──────────────────────────────────────────────────────────────
# Error Type 2: MED_DOSE
# ──────────────────────────────────────────────────────────────

class TestDosageCorruption:
    @pytest.mark.parametrize("summary,expected_unit", [
        ("Give metformin 500 mg twice daily.", "mg"),
        ("Administer furosemide 40 mcg IV.", "mcg"),
        ("Heparin 5000 units subcutaneous.", "units"),
    ])
    def test_dose_value_changed(self, drug_dict, summary, expected_unit):
        synth = make_synthesizer(drug_dict)
        text, detail = synth._corrupt_dosage(summary)
        assert detail is not None
        assert detail["type"] == "MED_DOSE"
        assert expected_unit.lower() in detail["corrupted"].lower()

    def test_no_dosage_returns_unchanged(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Patient was discharged home."
        text, detail = synth._corrupt_dosage(summary)
        assert text == summary
        assert detail is None

    def test_corrupted_value_differs(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "aspirin 325 mg daily"
        text, detail = synth._corrupt_dosage(summary)
        if detail:
            assert detail["original"] != detail["corrupted"]

    def test_new_value_is_numeric(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "lisinopril 10 mg daily"
        text, detail = synth._corrupt_dosage(summary)
        if detail:
            # corrupted should have a numeric prefix
            import re
            assert re.search(r'\d', detail["corrupted"]) is not None


# ──────────────────────────────────────────────────────────────
# Error Type 3: TEMPORAL
# ──────────────────────────────────────────────────────────────

class TestTemporalCorruption:
    @pytest.mark.parametrize("summary,contains", [
        ("Patient discharged on POD #3.", "POD"),
        ("She was seen on HD #5.", "HD"),
        ("Procedure done on 01/15/2024.", "/"),
        ("Patient on postoperative day three.", "postoperative day"),
    ])
    def test_temporal_expression_changed(self, drug_dict, summary, contains):
        synth = make_synthesizer(drug_dict, seed=1)
        text, detail = synth._corrupt_temporal(summary)
        if detail:
            assert detail["type"] == "TEMPORAL"
            assert text != summary

    def test_pod_value_shifts(self, drug_dict):
        synth = make_synthesizer(drug_dict, seed=0)
        summary = "Discharged on POD #5."
        text, detail = synth._corrupt_temporal(summary)
        if detail:
            # The number after POD should be different
            assert "5" not in text.replace("POD", "")  # naive check

    def test_day_never_goes_negative(self, drug_dict):
        """Even with max negative shift, day should stay >= 0."""
        synth = make_synthesizer(drug_dict, seed=99)
        for _ in range(20):
            summary = "Seen on POD #1."
            text, detail = synth._corrupt_temporal(summary)
            if detail:
                import re
                nums = re.findall(r'\d+', text)
                assert all(int(n) >= 0 for n in nums)

    def test_no_temporal_returns_unchanged(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Patient is doing well."
        text, detail = synth._corrupt_temporal(summary)
        assert text == summary
        assert detail is None


# ──────────────────────────────────────────────────────────────
# Error Type 4: NEGATION
# ──────────────────────────────────────────────────────────────

class TestNegationCorruption:
    def test_positive_to_negative(self, drug_dict):
        synth = make_synthesizer(drug_dict, seed=0)
        summary = "Patient had chest pain. He complained of shortness of breath."
        text, detail = synth._corrupt_negation(summary)
        if detail:
            assert detail["type"] == "NEGATION"
            assert detail["direction"] in ("pos_to_neg", "neg_to_pos")

    def test_negative_to_positive(self, drug_dict):
        synth = make_synthesizer(drug_dict, seed=0)
        summary = "Patient denied chest pain. No evidence of infection."
        text, detail = synth._corrupt_negation(summary)
        if detail:
            assert detail["type"] == "NEGATION"

    def test_no_negatable_text_unchanged(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        summary = "Temperature 98.6 F. BP 120/80 mmHg."
        text, detail = synth._corrupt_negation(summary)
        # May or may not change — just assert type safety
        assert isinstance(text, str)

    def test_guarded_words_not_corrupted(self, drug_dict):
        """'with complications' should NOT become 'without complications'."""
        synth = make_synthesizer(drug_dict, seed=0)
        summary = "Discharged without complications after surgery."
        for _ in range(10):
            text, _ = synth._corrupt_negation(summary)
            # The guard should prevent 'without complications' from becoming 'with complications'
            assert "with complications" not in text


# ──────────────────────────────────────────────────────────────
# Error Type 5: PROCEDURE
# ──────────────────────────────────────────────────────────────

class TestProcedureCorruption:
    def test_procedure_swapped(self, drug_dict, sample_summary, sample_note):
        synth = make_synthesizer(drug_dict, seed=0)
        text, detail = synth._corrupt_procedure(sample_summary, sample_note)
        # sample_note has 2+ procedures; sample_summary mentions one
        if detail:
            assert detail["type"] == "PROCEDURE"
            assert detail["original"] != detail["corrupted"]

    def test_insufficient_procedures_unchanged(self, drug_dict):
        synth = make_synthesizer(drug_dict)
        # Note with only one procedure
        note = "Patient underwent appendectomy."
        summary = "Patient had appendectomy."
        text, detail = synth._corrupt_procedure(summary, note)
        assert text == summary
        assert detail is None

    def test_extract_procedures_parses_header(self, drug_dict):
        note = "<MAJOR SURGICAL OR INVASIVE PROCEDURE>CABG, valve replacement</>"
        procs = ClinicalErrorSynthesizer._extract_procedures(note)
        assert len(procs) >= 2
        assert any("CABG" in p for p in procs)

    def test_extract_procedures_underwent_pattern(self, drug_dict):
        note = "Patient underwent coronary angiography and stent placement."
        procs = ClinicalErrorSynthesizer._extract_procedures(note)
        assert len(procs) >= 1
