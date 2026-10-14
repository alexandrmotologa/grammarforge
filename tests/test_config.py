"""Unit tests for configuration models."""

import pytest
from grammarforge.core.config import (
    CompilerConfig,
    EngineConfig,
    KeyOrdering,
    WhitespaceMode,
)


def test_compiler_config_defaults() -> None:
    config = CompilerConfig()
    assert config.whitespace_mode == WhitespaceMode.LENIENT
    assert config.key_ordering == KeyOrdering.CANONICAL
    assert config.max_recursion_depth == 10
    assert config.root_rule_name == "root"
    assert config.allow_empty_objects is True
    assert config.strict_numbers is True


def test_compiler_config_custom() -> None:
    config = CompilerConfig(
        whitespace_mode=WhitespaceMode.COMPACT,
        key_ordering=KeyOrdering.FLEXIBLE,
        root_rule_name="custom_root",
        allow_empty_objects=False,
    )
    assert config.whitespace_mode == WhitespaceMode.COMPACT
    assert config.key_ordering == KeyOrdering.FLEXIBLE
    assert config.root_rule_name == "custom_root"
    assert config.allow_empty_objects is False


def test_engine_config_defaults() -> None:
    config = EngineConfig()
    assert config.vocab_size == 32000
    assert config.eos_token_id == 2
    assert config.neg_inf_value == -1e9
    assert config.cache_trie_traversals is True
