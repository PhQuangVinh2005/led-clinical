"""
Unit tests for DrugDictionary logic.

Because DrugDictionary.__init__ calls pq.read_table(), we test
via the StubDrugDictionary fixture from conftest.py which exposes
the exact same public API without any I/O.
"""
import pytest


# ──────────────────────────────────────────────────────────────
# find_drugs_in_text
# ──────────────────────────────────────────────────────────────

class TestFindDrugsInText:
    def test_exact_match(self, drug_dict):
        found = drug_dict.find_drugs_in_text("Patient was given aspirin daily.")
        assert "aspirin" in found

    def test_case_insensitive(self, drug_dict):
        found = drug_dict.find_drugs_in_text("Patient was given ASPIRIN.")
        assert "aspirin" in found

    def test_multiple_drugs(self, drug_dict):
        found = drug_dict.find_drugs_in_text(
            "Started on aspirin 325 mg and lisinopril 10 mg."
        )
        assert "aspirin" in found
        assert "lisinopril" in found

    def test_word_boundary_no_partial_match(self, drug_dict):
        # "aspirinated" should NOT match "aspirin"
        found = drug_dict.find_drugs_in_text("The patient was aspirinated.")
        assert "aspirin" not in found

    def test_empty_text_returns_empty(self, drug_dict):
        found = drug_dict.find_drugs_in_text("")
        assert found == []

    def test_text_with_no_drugs_returns_empty(self, drug_dict):
        found = drug_dict.find_drugs_in_text(
            "The patient was discharged in stable condition."
        )
        assert found == []

    def test_drug_at_start_of_string(self, drug_dict):
        found = drug_dict.find_drugs_in_text("aspirin was prescribed.")
        assert "aspirin" in found

    def test_drug_at_end_of_string(self, drug_dict):
        found = drug_dict.find_drugs_in_text("The medication prescribed was aspirin")
        assert "aspirin" in found

    def test_returns_list(self, drug_dict):
        result = drug_dict.find_drugs_in_text("aspirin daily")
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────
# get_random_substitute
# ──────────────────────────────────────────────────────────────

class TestGetRandomSubstitute:
    def test_returns_different_drug(self, drug_dict):
        sub = drug_dict.get_random_substitute("aspirin")
        assert sub is not None
        assert sub.lower() != "aspirin"

    def test_returns_string(self, drug_dict):
        sub = drug_dict.get_random_substitute("metformin")
        assert isinstance(sub, str)

    def test_substitute_is_from_dictionary(self, drug_dict):
        sub = drug_dict.get_random_substitute("warfarin")
        assert sub in drug_dict.all_drug_names

    def test_unknown_drug_returns_fallback(self, drug_dict):
        # "xyz_unknown" not in dict — should fall back to any drug
        sub = drug_dict.get_random_substitute("xyz_unknown")
        assert sub is not None
        assert sub in drug_dict.all_drug_names

    def test_deterministic_with_same_seed(self):
        from tests.conftest import StubDrugDictionary
        d1 = StubDrugDictionary(seed=99)
        d2 = StubDrugDictionary(seed=99)
        assert d1.get_random_substitute("aspirin") == d2.get_random_substitute("aspirin")

    def test_case_insensitive_lookup(self, drug_dict):
        # Passing uppercase should still find substitute
        sub = drug_dict.get_random_substitute("ASPIRIN")
        assert sub is not None


# ──────────────────────────────────────────────────────────────
# get_any_random_drug
# ──────────────────────────────────────────────────────────────

class TestGetAnyRandomDrug:
    def test_returns_string(self, drug_dict):
        assert isinstance(drug_dict.get_any_random_drug(), str)

    def test_returns_known_drug(self, drug_dict):
        drug = drug_dict.get_any_random_drug()
        assert drug in drug_dict.all_drug_names

    def test_multiple_calls_not_always_same(self, drug_dict):
        results = {drug_dict.get_any_random_drug() for _ in range(30)}
        assert len(results) > 1  # should sample variety across 10 drugs


# ──────────────────────────────────────────────────────────────
# Internal state consistency
# ──────────────────────────────────────────────────────────────

class TestDictionaryState:
    def test_name_to_id_keys_match_all_drug_names(self, drug_dict):
        assert set(drug_dict.name_to_id.keys()) == drug_dict.all_drug_names

    def test_id_to_name_values_are_drug_names(self, drug_dict):
        for name in drug_dict.id_to_name.values():
            assert name in drug_dict.all_drug_names

    def test_source_to_names_covers_all_drugs(self, drug_dict):
        all_in_source = set()
        for names in drug_dict.source_to_names.values():
            all_in_source.update(names)
        # Every drug should appear in at least one source category
        assert drug_dict.all_drug_names.issubset(all_in_source)
