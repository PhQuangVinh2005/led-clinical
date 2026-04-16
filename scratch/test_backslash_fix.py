
import re
from src.data.error_synthesizer import ClinicalErrorSynthesizer
from src.data.drug_dictionary import DrugDictionary

def test_backslash_bug():
    # Simple mock dictionary
    class MockDict:
        def get_substitute(self, name, rng): return "Drug\\With\\Backslash"
        def get_entity_class(self, name): return "Drug"
        def find_drugs(self, text): return ["Aspirin"] if "Aspirin" in text else []
        def initialize(self): pass

    synth = ClinicalErrorSynthesizer(MockDict(), corruption_rate=1.0, seed=42)
    
    summary = "Patient took Aspirin."
    source = "Patient took Drug\\With\\Backslash"
    
    print(f"Original: {summary}")
    try:
        corrupted, detail = synth._corrupt_medication_name(summary, source)
        print(f"Corrupted: {corrupted}")
        print(f"Detail: {detail}")
        print("Test PASSED (No crash)")
    except Exception as e:
        print(f"Test FAILED with error: {e}")

    # Test Procedure swap
    summary = "Patient underwent procedure X."
    source = "Underwent procedure X. Underwent procedure \\A_Backslash."
    
    try:
        # Mock extract_procedures for simple test
        synth._extract_procedures = lambda _: ["procedure X", "procedure \\A_Backslash"]
        corrupted, detail = synth._corrupt_procedure(summary, source)
        print(f"\nOriginal Proc: {summary}")
        print(f"Corrupted Proc: {corrupted}")
        print(f"Detail: {detail}")
        print("Test PASSED (No crash)")
    except Exception as e:
        print(f"Test FAILED with error: {e}")

if __name__ == "__main__":
    test_backslash_bug()
