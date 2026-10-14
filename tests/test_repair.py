"""Unit tests for JSON auto-repair and stream healer."""

import json
import pytest
from grammarforge.engine.repair import heal_partial_json


def test_heal_unclosed_string_and_brace() -> None:
    truncated = '{"name": "Alex'
    result = heal_partial_json(truncated)
    assert result.is_valid_json
    assert result.repaired_text == '{"name": "Alex"}'
    assert result.suffix_added == '"}'


def test_heal_nested_structures() -> None:
    truncated = '{"user": {"id": 10, "tags": ["admin", "dev"'
    result = heal_partial_json(truncated)
    assert result.is_valid_json
    assert result.repaired_text == '{"user": {"id": 10, "tags": ["admin", "dev"]}}'


def test_heal_dangling_colon() -> None:
    truncated = '{"count": 5, "details":'
    result = heal_partial_json(truncated)
    assert result.is_valid_json
    parsed = json.loads(result.repaired_text)
    assert parsed["count"] == 5
    assert parsed["details"] is None


def test_heal_trailing_comma() -> None:
    truncated = '{"items": [1, 2, '
    result = heal_partial_json(truncated)
    assert result.is_valid_json
    parsed = json.loads(result.repaired_text)
    assert parsed["items"] == [1, 2]


def test_heal_empty_string() -> None:
    schema = {"type": "object"}
    result = heal_partial_json("", schema=schema)
    assert result.is_valid_json
    assert result.repaired_text == "{}"
