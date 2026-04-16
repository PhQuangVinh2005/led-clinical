"""
Drug Dictionary module for medication name corruption.

Loads the heh.parquet drug/disease dictionary and provides lookup
functionality for medication name swaps during synthetic error generation.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import pyarrow.parquet as pq


class DrugDictionary:
    """Loads and queries the drug dictionary for medication name swaps."""

    def __init__(self, parquet_path: str, seed: int = 42):
        self.rng = random.Random(seed)
        self._load(parquet_path)

    def _load(self, parquet_path: str) -> None:
        """Load parquet and build lookup structures."""
        table = pq.read_table(parquet_path)

        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}
        self.source_to_names: Dict[str, List[str]] = {}
        self.all_drug_names: Set[str] = set()
        self.name_to_class: Dict[str, str] = {}  # Added to track entity class

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
            self.name_to_class[name_lower] = label  # Store the source class/label

            # Group by source label for same-class swaps
            if label not in self.source_to_names:
                self.source_to_names[label] = []
            self.source_to_names[label].append(name_lower)

        print(f"[DrugDictionary] Loaded {len(self.all_drug_names)} drug entries "
              f"from {len(self.source_to_names)} source categories")

    def find_drugs_in_text(self, text: str) -> List[str]:
        """Find drug names present in the given text (case-insensitive)."""
        text_lower = text.lower()
        found = []
        for name in self.all_drug_names:
            # Only match whole words (avoid partial matches)
            # Simple approach: check if name appears as substring
            # with word boundaries
            idx = text_lower.find(name)
            if idx != -1:
                # Check word boundaries
                before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
                after_idx = idx + len(name)
                after_ok = (after_idx >= len(text_lower) or
                            not text_lower[after_idx].isalnum())
                if before_ok and after_ok:
                    found.append(name)
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
