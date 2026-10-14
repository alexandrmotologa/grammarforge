"""Precomputed static mask tables for zero-cost punctuation and literal transitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from grammarforge.engine.trie import VocabularyTrie


class PrecomputedMasks:
    """Stores pre-compiled token bitmasks for common JSON structural transitions."""

    def __init__(self, vocab_trie: VocabularyTrie, neg_inf_val: float = -1e9) -> None:
        self.vocab = vocab_trie
        self.neg_inf = neg_inf_val
        self._token_sets: Dict[str, List[int]] = {}
        self._mask_vectors: Dict[str, List[float]] = {}
        self._precompute_common()

    def _precompute_common(self) -> None:
        """Pre-index standard structural delimiters and boolean/null literals."""
        literals = {
            "open_object": ['{"', "{", "{\n", "{\n  "],
            "close_object": ["}", "}\n", "\n}"],
            "open_array": ['["', "[", "[\n", "[\n  "],
            "close_array": ["]", "]\n", "\n]"],
            "colon": [":", ": ", '":', '": "'],
            "comma": [",", ", ", '",', '",\n', '", "'],
            "bool": ["true", "false"],
            "null": ["null"],
        }

        for key, prefixes in literals.items():
            matching_ids: Set[int] = set()
            for p in prefixes:
                matching_ids.update(self.vocab.tokens_with_prefix(p))
            tok_list = sorted(list(matching_ids))
            self._token_sets[key] = tok_list

    def get_allowed_tokens(self, category: str) -> Optional[List[int]]:
        """Retrieve precomputed list of allowed token IDs for a structural category."""
        return self._token_sets.get(category)

    def get_mask_vector(self, category: str) -> Optional[List[float]]:
        """Retrieve precomputed full vocabulary logit mask vector."""
        if category in self._mask_vectors:
            return self._mask_vectors[category]

        token_list = self._token_sets.get(category)
        if token_list is None:
            return None

        vector = [self.neg_inf] * self.vocab.vocab_size
        for tid in token_list:
            if 0 <= tid < self.vocab.vocab_size:
                vector[tid] = 0.0

        self._mask_vectors[category] = vector
        return vector

    def register_custom_literals(self, category_name: str, values: List[str]) -> None:
        """Precompute mask for custom enum or keyword values."""
        matching_ids: Set[int] = set()
        for v in values:
            matching_ids.update(self.vocab.tokens_with_prefix(v))
        self._token_sets[category_name] = sorted(list(matching_ids))
        if category_name in self._mask_vectors:
            del self._mask_vectors[category_name]
