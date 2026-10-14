"""Unit tests for Vocabulary Trie and token indexing."""

import pytest
from grammarforge.engine.trie import VocabularyTrie


def test_trie_insert_and_lookup() -> None:
    trie = VocabularyTrie()
    trie.insert(0, "apple")
    trie.insert(1, "app")
    trie.insert(2, "application")
    trie.insert(3, "banana")

    assert trie.get_token_bytes(0) == b"apple"
    assert trie.get_token_str(0) == "apple"
    assert trie.vocab_size == 4

    tokens_app = trie.tokens_with_prefix("app")
    assert 0 in tokens_app
    assert 1 in tokens_app
    assert 2 in tokens_app
    assert 3 not in tokens_app

    tokens_ban = trie.tokens_with_prefix("ban")
    assert tokens_ban == [3]


def test_synthetic_vocab_creation() -> None:
    trie = VocabularyTrie.create_synthetic_vocab(size=1000)
    assert trie.vocab_size == 1000
    # Check special tokens
    assert 0 in trie.special_tokens
    assert 1 in trie.special_tokens
    assert 2 in trie.special_tokens

    # Check structural JSON tokens
    tokens_brace = trie.tokens_with_prefix('{"')
    assert len(tokens_brace) > 0


def test_trie_from_dict() -> None:
    vocab_dict = {
        0: "<pad>",
        1: "hello",
        2: "world",
    }
    trie = VocabularyTrie.from_dict(vocab_dict)
    assert trie.vocab_size == 3
    assert trie.get_token_str(1) == "hello"
