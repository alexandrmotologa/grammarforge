"""Unit tests for precomputed static mask tables."""

import pytest
from grammarforge.engine.precomputed import PrecomputedMasks
from grammarforge.engine.trie import VocabularyTrie


def test_precomputed_masks() -> None:
    trie = VocabularyTrie.create_synthetic_vocab(size=1000)
    pre = PrecomputedMasks(trie)

    # Check structural token sets
    open_obj_tokens = pre.get_allowed_tokens("open_object")
    assert open_obj_tokens is not None
    assert len(open_obj_tokens) > 0

    colon_tokens = pre.get_allowed_tokens("colon")
    assert colon_tokens is not None
    assert len(colon_tokens) > 0

    # Check mask vector
    vec = pre.get_mask_vector("bool")
    assert vec is not None
    assert len(vec) == 1000
    assert 0.0 in vec
    assert -1e9 in vec


def test_register_custom_literals() -> None:
    trie = VocabularyTrie.create_synthetic_vocab(size=500)
    pre = PrecomputedMasks(trie)

    pre.register_custom_literals("sentiment", ["positive", "neutral", "negative"])
    toks = pre.get_allowed_tokens("sentiment")
    assert toks is not None
