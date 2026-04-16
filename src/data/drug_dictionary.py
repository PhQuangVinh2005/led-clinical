"""
Drug Dictionary module for medication name corruption.

Loads the heh.parquet drug/disease dictionary and provides O(n) lookup
of drug names in text using the Aho-Corasick algorithm (pyahocorasick).

Why Aho-Corasick?
  The naive approach (iterating all ~173K drug names against each text) is
  O(n_drugs × text_len) per call — too slow for 265K samples at preprocessing.
  Aho-Corasick builds a finite automaton once (O(Σ key lengths)) and then
  finds all occurrences in a single O(text_len + matches) pass.
"""

import random
from typing import Dict, List, Optional, Set

import ahocorasick
import pyarrow.parquet as pq


class DrugDictionary:
    """Loads and queries the drug dictionary for medication name swaps."""

    def __init__(self, parquet_path: str, seed: int = 42):
        self.rng = random.Random(seed)
        self._load(parquet_path)

    def _load(self, parquet_path: str) -> None:
        """Load parquet and build Aho-Corasick automaton + lookup structures."""
        table = pq.read_table(parquet_path)

        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}
        self.source_to_names: Dict[str, List[str]] = {}
        self.all_drug_names: Set[str] = set()
        self.name_to_class: Dict[str, str] = {}

        # Only keep drug entries (filter out disease entries)
        for i in range(table.num_rows):
            label = table.column("labels")[i].as_py()
            name = table.column("name")[i].as_py()
            drug_id = table.column("id")[i].as_py()

            if "Drug" not in label:
                continue

            # Skip very short names (likely abbreviations/noise)
            # and very long chemical compound names
            if len(name) < 4 or len(name) > 60:
                continue

            name_lower = name.lower()
            self.name_to_id[name_lower] = drug_id
            self.id_to_name[drug_id] = name
            self.all_drug_names.add(name_lower)
            self.name_to_class[name_lower] = label

            # Group by source label for same-class swaps
            if label not in self.source_to_names:
                self.source_to_names[label] = []
            self.source_to_names[label].append(name_lower)

        print(f"[DrugDictionary] Loaded {len(self.all_drug_names)} drug entries "
              f"from {len(self.source_to_names)} source categories")

        # Build Aho-Corasick automaton for O(text_len) multi-pattern search
        self._automaton = ahocorasick.Automaton()
        for name_lower in self.all_drug_names:
            self._automaton.add_word(name_lower, name_lower)
        self._automaton.make_automaton()

        print(f"[DrugDictionary] Aho-Corasick automaton built "
              f"({len(self._automaton)} patterns)")

    def find_drugs_in_text(self, text: str) -> List[str]:
        """
        Find drug names present in the given text (case-insensitive).

        Uses Aho-Corasick for O(text_len + matches) complexity instead of
        the naive O(n_drugs × text_len) scan.
        Only returns whole-word matches (checked via boundary chars).
        """
        text_lower = text.lower()
        found = []
        seen: Set[str] = set()  # deduplicate multi-occurrence hits

        for end_idx, name_lower in self._automaton.iter(text_lower):
            start_idx = end_idx - len(name_lower) + 1

            # Word-boundary check
            before_ok = start_idx == 0 or not text_lower[start_idx - 1].isalnum()
            after_idx = end_idx + 1
            after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()

            if before_ok and after_ok and name_lower not in seen:
                found.append(name_lower)
                seen.add(name_lower)

        return found

    def get_random_substitute(self, original_name: str) -> Optional[str]:
        """Get a random substitute drug from the same source category."""
        original_lower = original_name.lower()

        # Find which source category this drug belongs to
        for source, names in self.source_to_names.items():
            if original_lower in names:
                # Pick a different drug from the same category
                candidates = [n for n in names if n != original_lower]
                if candidates:
                    return self.rng.choice(candidates)

        # Fallback: pick from any category
        all_names = list(self.all_drug_names - {original_lower})
        if all_names:
            return self.rng.choice(all_names)

        return None

    def get_any_random_drug(self) -> str:
        """Get any random drug name from the dictionary."""
        return self.rng.choice(list(self.all_drug_names))

    def get_entity_class(self, name: str) -> Optional[str]:
        """Return the class/label for a given entity name."""
        return self.name_to_class.get(name.lower())
