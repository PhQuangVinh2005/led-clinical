"""
Clinical Error Synthesizer — Rule-based synthetic error generation.

Generates 5 types of clinical factual errors for training the LED corrector:
1. Medication Name Swap (MED_NAME)
2. Dosage Corruption (MED_DOSE)
3. Temporal/POD Corruption (TEMPORAL)
4. Negation Insertion/Removal (NEGATION)
5. Procedure Swap (PROCEDURE)
"""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.data.drug_dictionary import DrugDictionary


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DOSAGE_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*(mg|mcg|mL|units?|g|meq|mmol)\b',
    re.IGNORECASE
)

TEMPORAL_PATTERNS = [
    (re.compile(r'POD\s*#?\s*(\d+)', re.IGNORECASE), 'POD'),
    (re.compile(r'post-operative day\s*#?\s*(\d+)', re.IGNORECASE), 'POD'),
    (re.compile(r'postoperative day\s*(\w+)', re.IGNORECASE), 'POD_WORD'),
    (re.compile(r'HD\s*#?\s*(\d+)', re.IGNORECASE), 'HD'),
    (re.compile(r'hospital day\s*#?\s*(\d+)', re.IGNORECASE), 'HD'),
    (re.compile(r'day\s*(\d+)\s*(?:of|post)', re.IGNORECASE), 'DAY'),
    (re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})'), 'DATE_SLASH'),
    (re.compile(r'(\d{1,2})-(\d{1,2})-(\d{2,4})'), 'DATE_DASH'),
]

# Word-to-number mapping for "postoperative day one" style
WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'first': 1, 'second': 2, 'third': 3,
    'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
}
NUM_TO_WORD = {v: k for k, v in WORD_TO_NUM.items() if k not in
               ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']}

# ── Negation patterns ──
# Guard words: don't corrupt these structural phrases
NEGATION_GUARDS = {'difficulty', 'issues', 'services', 'instructions',
                   'complications', 'incident', 'problems', 'event'}

# Positive → Negative (insert negation)
POSITIVE_TO_NEGATIVE = [
    # "patient had/has/developed X" → "patient denied X"
    (re.compile(
        r'(patient|pt\.?|he|she)\s+(had|has|developed|experienced|presented with)\s+',
        re.IGNORECASE),
     lambda m: m.group(1) + ' denied '),
    # "was found to have" → "was not found to have"
    (re.compile(r'(was|were)\s+(found|noted)\s+to\s+have', re.IGNORECASE),
     lambda m: m.group(1) + ' not ' + m.group(2) + ' to have'),
    # "positive for" → "negative for"
    (re.compile(r'positive\s+for', re.IGNORECASE),
     lambda m: 'negative for'),
    # "complained of" → "denied"
    (re.compile(r'complained?\s+of', re.IGNORECASE),
     lambda m: 'denied'),
    # "with [symptom]" → "without [symptom]" (guarded)
    (re.compile(r'\bwith\s+(?!' + '|'.join(NEGATION_GUARDS) + r')', re.IGNORECASE),
     lambda m: 'without '),
]

# Negative → Positive (remove negation)
NEGATIVE_TO_POSITIVE = [
    # "denied/denies X" → "had X"
    (re.compile(r'(denied|denies)\s+', re.IGNORECASE),
     lambda m: 'had '),
    # "no [evidence of] X" → "X"
    (re.compile(r'\bno\s+(?:evidence\s+of\s+)?', re.IGNORECASE),
     lambda m: ''),
    # "without X" → "with X" (guarded)
    (re.compile(r'\bwithout\s+(?!' + '|'.join(NEGATION_GUARDS) + r')', re.IGNORECASE),
     lambda m: 'with '),
    # "negative for" → "positive for"
    (re.compile(r'negative\s+for', re.IGNORECASE),
     lambda m: 'positive for'),
    # "was not" → "was"
    (re.compile(r'(was|were)\s+not\s+', re.IGNORECASE),
     lambda m: m.group(1) + ' '),
]

# ── Procedure extraction patterns ──
PROCEDURE_SECTION_PATTERNS = [
    re.compile(r'<MAJOR SURGICAL OR INVASIVE PROCEDURE>(.*?)(?=<|$)',
               re.IGNORECASE | re.DOTALL),
    re.compile(r'<PROCEDURES>(.*?)(?=<|$)',
               re.IGNORECASE | re.DOTALL),
    re.compile(r'underwent\s+([^.]+)', re.IGNORECASE),
    re.compile(r's/p\s+([^.]+)', re.IGNORECASE),
]


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class CorruptionResult:
    """Result of corrupting a summary."""
    original_summary: str
    corrupted_summary: str
    is_corrupted: bool
    error_types: List[str] = field(default_factory=list)
    error_details: List[Dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Main synthesizer
# ──────────────────────────────────────────────────────────────

class ClinicalErrorSynthesizer:
    """
    Rule-based synthetic error generator for clinical summaries.

    Applies up to 5 types of factual errors to create training data
    for the LED error correction model.
    """

    ERROR_TYPES = ['MED_NAME', 'MED_DOSE', 'TEMPORAL', 'NEGATION', 'PROCEDURE']

    def __init__(
        self,
        drug_dictionary: DrugDictionary,
        corruption_rate: float = 0.3,
        seed: int = 42,
    ):
        self.drug_dict = drug_dictionary
        self.corruption_rate = corruption_rate
        self.rng = random.Random(seed)

    def corrupt(
        self,
        summary: str,
        source_note: str,
    ) -> CorruptionResult:
        """
        Potentially corrupt a summary with synthetic errors.

        With probability `corruption_rate`, apply 1-3 random error types.
        Otherwise, return the summary unchanged (identity mapping).
        """
        # Decide whether to corrupt this sample
        if self.rng.random() > self.corruption_rate:
            return CorruptionResult(
                original_summary=summary,
                corrupted_summary=summary,
                is_corrupted=False,
            )

        # Select 1-3 random error types
        n_errors = self.rng.randint(1, 3)
        selected_types = self.rng.sample(self.ERROR_TYPES, min(n_errors, len(self.ERROR_TYPES)))

        corrupted = summary
        applied_types = []
        error_details = []

        for error_type in selected_types:
            result_text, detail = self._apply_error(corrupted, source_note, error_type)
            if result_text != corrupted:  # Error was actually applied
                corrupted = result_text
                applied_types.append(error_type)
                if detail:
                    error_details.append(detail)

        return CorruptionResult(
            original_summary=summary,
            corrupted_summary=corrupted,
            is_corrupted=corrupted != summary,
            error_types=applied_types,
            error_details=error_details,
        )

    def _apply_error(
        self,
        summary: str,
        source_note: str,
        error_type: str,
    ) -> Tuple[str, Optional[Dict]]:
        """Apply a single error type to the summary."""
        if error_type == 'MED_NAME':
            return self._corrupt_medication_name(summary)
        elif error_type == 'MED_DOSE':
            return self._corrupt_dosage(summary)
        elif error_type == 'TEMPORAL':
            return self._corrupt_temporal(summary)
        elif error_type == 'NEGATION':
            return self._corrupt_negation(summary)
        elif error_type == 'PROCEDURE':
            return self._corrupt_procedure(summary, source_note)
        return summary, None

    # ── Error Type 1: Medication Name Swap ──

    def _corrupt_medication_name(self, summary: str) -> Tuple[str, Optional[Dict]]:
        """Replace a medication name with a different drug from the dictionary."""
        found_drugs = self.drug_dict.find_drugs_in_text(summary)
        if not found_drugs:
            return summary, None

        # Pick one drug to corrupt
        target_drug = self.rng.choice(found_drugs)
        substitute = self.drug_dict.get_random_substitute(target_drug)
        if not substitute:
            return summary, None

        # Replace first occurrence (case-insensitive)
        pattern = re.compile(re.escape(target_drug), re.IGNORECASE)
        corrupted = pattern.sub(substitute, summary, count=1)

        return corrupted, {
            'type': 'MED_NAME',
            'original': target_drug,
            'corrupted': substitute,
        }

    # ── Error Type 2: Dosage Corruption ──

    def _corrupt_dosage(self, summary: str) -> Tuple[str, Optional[Dict]]:
        """Modify a dosage value by multiplying with a random factor."""
        matches = list(DOSAGE_PATTERN.finditer(summary))
        if not matches:
            return summary, None

        # Pick one dosage to corrupt
        match = self.rng.choice(matches)
        original_value = float(match.group(1))
        unit = match.group(2)

        # Multiply by random factor [0.5, 2.0], avoiding 1.0
        factor = self.rng.choice([0.5, 0.25, 1.5, 2.0, 3.0])
        new_value = original_value * factor

        # Round appropriately
        if new_value == int(new_value):
            new_str = str(int(new_value))
        else:
            new_str = f"{new_value:.1f}"

        original_str = match.group(0)
        corrupted_str = f"{new_str} {unit}"
        corrupted = summary[:match.start()] + corrupted_str + summary[match.end():]

        return corrupted, {
            'type': 'MED_DOSE',
            'original': original_str,
            'corrupted': corrupted_str,
        }

    # ── Error Type 3: Temporal/POD Corruption ──

    def _corrupt_temporal(self, summary: str) -> Tuple[str, Optional[Dict]]:
        """Shift a temporal reference (POD, HD, date) by ±1-3 days."""
        for pattern, ptype in TEMPORAL_PATTERNS:
            match = pattern.search(summary)
            if not match:
                continue

            if ptype in ('POD', 'HD', 'DAY'):
                original_num = int(match.group(1))
                shift = self.rng.choice([-3, -2, -1, 1, 2, 3])
                new_num = max(0, original_num + shift)
                original_str = match.group(0)
                corrupted_str = original_str.replace(str(original_num), str(new_num))
                corrupted = summary[:match.start()] + corrupted_str + summary[match.end():]
                return corrupted, {
                    'type': 'TEMPORAL',
                    'subtype': ptype,
                    'original': original_str,
                    'corrupted': corrupted_str,
                }

            elif ptype == 'POD_WORD':
                word = match.group(1).lower()
                if word in WORD_TO_NUM:
                    original_num = WORD_TO_NUM[word]
                    shift = self.rng.choice([-2, -1, 1, 2])
                    new_num = max(1, original_num + shift)
                    new_word = NUM_TO_WORD.get(new_num, str(new_num))
                    original_str = match.group(0)
                    corrupted_str = original_str.replace(match.group(1), new_word)
                    corrupted = summary[:match.start()] + corrupted_str + summary[match.end():]
                    return corrupted, {
                        'type': 'TEMPORAL',
                        'subtype': 'POD_WORD',
                        'original': original_str,
                        'corrupted': corrupted_str,
                    }

            elif ptype in ('DATE_SLASH', 'DATE_DASH'):
                # Shift the day component
                day = int(match.group(2))
                shift = self.rng.choice([-3, -2, -1, 1, 2, 3])
                new_day = max(1, min(31, day + shift))
                sep = '/' if ptype == 'DATE_SLASH' else '-'
                original_str = match.group(0)
                corrupted_str = f"{match.group(1)}{sep}{new_day}{sep}{match.group(3)}"
                corrupted = summary[:match.start()] + corrupted_str + summary[match.end():]
                return corrupted, {
                    'type': 'TEMPORAL',
                    'subtype': ptype,
                    'original': original_str,
                    'corrupted': corrupted_str,
                }

        return summary, None

    # ── Error Type 4: Negation Insertion/Removal ──

    def _corrupt_negation(self, summary: str) -> Tuple[str, Optional[Dict]]:
        """Insert or remove negation in a clinical statement."""
        # Split into sentences for targeted corruption
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        if not sentences:
            return summary, None

        # Try each sentence until we find one we can corrupt
        sentence_indices = list(range(len(sentences)))
        self.rng.shuffle(sentence_indices)

        for idx in sentence_indices:
            sentence = sentences[idx]

            # First try: remove existing negation (Negative → Positive)
            for pattern, replacement_fn in NEGATIVE_TO_POSITIVE:
                match = pattern.search(sentence)
                if match:
                    corrupted_sentence = pattern.sub(replacement_fn, sentence, count=1)
                    if corrupted_sentence != sentence:
                        sentences[idx] = corrupted_sentence
                        corrupted = ' '.join(sentences)
                        return corrupted, {
                            'type': 'NEGATION',
                            'direction': 'neg_to_pos',
                            'original': sentence.strip(),
                            'corrupted': corrupted_sentence.strip(),
                        }

            # Second try: insert negation (Positive → Negative)
            for pattern, replacement_fn in POSITIVE_TO_NEGATIVE:
                match = pattern.search(sentence)
                if match:
                    corrupted_sentence = pattern.sub(replacement_fn, sentence, count=1)
                    if corrupted_sentence != sentence:
                        sentences[idx] = corrupted_sentence
                        corrupted = ' '.join(sentences)
                        return corrupted, {
                            'type': 'NEGATION',
                            'direction': 'pos_to_neg',
                            'original': sentence.strip(),
                            'corrupted': corrupted_sentence.strip(),
                        }

        return summary, None

    # ── Error Type 5: Procedure Swap ──

    def _corrupt_procedure(
        self,
        summary: str,
        source_note: str,
    ) -> Tuple[str, Optional[Dict]]:
        """Swap a procedure mention with another procedure from the same note."""
        procedures = self._extract_procedures(source_note)
        if len(procedures) < 2:
            return summary, None

        # Find procedures mentioned in the summary
        summary_lower = summary.lower()
        for proc in procedures:
            if proc.lower() in summary_lower:
                # Find a different procedure to substitute
                other_procs = [p for p in procedures
                               if p.lower() != proc.lower() and len(p) > 5]
                if other_procs:
                    wrong_proc = self.rng.choice(other_procs)
                    # Case-insensitive replace, first occurrence only
                    pattern = re.compile(re.escape(proc), re.IGNORECASE)
                    corrupted = pattern.sub(wrong_proc, summary, count=1)
                    if corrupted != summary:
                        return corrupted, {
                            'type': 'PROCEDURE',
                            'original': proc,
                            'corrupted': wrong_proc,
                        }

        return summary, None

    @staticmethod
    def _extract_procedures(text: str) -> List[str]:
        """Extract procedure names from clinical note section headers."""
        procedures = []
        for pattern in PROCEDURE_SECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                proc_text = match.group(1)
                for proc in re.split(r',|\band\b|with', proc_text):
                    proc = proc.strip().strip('_').strip()
                    if len(proc) > 2 and not proc.isdigit():
                        procedures.append(proc)
        return procedures
