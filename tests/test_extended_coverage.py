"""Extended unit tests to ensure high test coverage across all edge cases."""

from dataclasses import dataclass
import json
import pytest
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.compiler.pydantic_bridge import get_schema_from_type, load_model_from_spec
from grammarforge.core.config import CompilerConfig, WhitespaceMode
from grammarforge.core.exceptions import (
    GrammarExhaustedError,
    GrammarForgeError,
    InvalidGrammarError,
    MaskingError,
    RegexCompilationError,
    SchemaCompilationError,
)
from grammarforge.engine.gateway import GrammarForgeLogitsProcessor, StreamingValidator
from grammarforge.engine.masking import LogitMasker
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie


@dataclass
class Point3D:
    x: float
    y: float
    z: float


def test_dataclass_introspection() -> None:
    schema = get_schema_from_type(Point3D)
    assert schema["type"] == "object"
    assert "x" in schema["properties"]
    assert "y" in schema["properties"]
    assert "z" in schema["properties"]


def test_schema_boolean_and_empty_edge_cases() -> None:
    # Boolean schemas
    gbnf_true = compile_json_schema(True)
    assert "root ::= json-value" in gbnf_true

    gbnf_false = compile_json_schema(False)
    assert 'root ::= ""' in gbnf_false

    # Strictly empty object
    gbnf_empty_obj = compile_json_schema({"type": "object", "additionalProperties": False})
    assert '"{"' in gbnf_empty_obj
    assert '"}"' in gbnf_empty_obj

    # Object with additionalProperties schema
    gbnf_map = compile_json_schema({
        "type": "object",
        "additionalProperties": {"type": "integer"},
    })
    assert "json-string" in gbnf_map

    # Array tuple validation
    gbnf_tuple = compile_json_schema({
        "type": "array",
        "items": [{"type": "string"}, {"type": "integer"}],
    })
    assert "json-string" in gbnf_tuple and "json-int" in gbnf_tuple


def test_state_machine_extended_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "str_field": {"type": "string", "minLength": 2, "maxLength": 5},
            "int_field": {"type": "integer", "minimum": 10, "maximum": 20},
            "num_field": {"type": "number", "minimum": 0.5},
            "bool_field": {"type": "boolean"},
            "null_field": {"type": "null"},
            "arr_field": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer"}},
        },
        "required": ["str_field", "int_field", "bool_field"],
    }
    sm = JsonStateMachine(schema)

    # Valid complete
    valid_payload = json.dumps({
        "str_field": "test",
        "int_field": 15,
        "num_field": 1.5,
        "bool_field": False,
        "null_field": None,
        "arr_field": [1, 2],
    })
    status, err = sm.check_partial_json(valid_payload)
    assert status == ParserStatus.ACCEPTING
    assert err is None

    # Length violation
    bad_str_payload = json.dumps({
        "str_field": "toolongstring",
        "int_field": 15,
        "bool_field": True,
    })
    status_bad, err_bad = sm.check_partial_json(bad_str_payload)
    assert status_bad == ParserStatus.REJECTED
    assert "maxLength" in err_bad

    # Minimum int violation
    bad_int_payload = json.dumps({
        "str_field": "ok",
        "int_field": 5,
        "bool_field": True,
    })
    status_int, err_int = sm.check_partial_json(bad_int_payload)
    assert status_int == ParserStatus.REJECTED
    assert "minimum" in err_int


def test_streaming_validator_reset_and_ids() -> None:
    schema = {"type": "string"}
    trie = VocabularyTrie.create_synthetic_vocab(size=300)
    val = StreamingValidator(schema, vocab_trie=trie)

    # Feed token ID
    val.feed_token(trie.tokens_with_prefix('"')[0])
    assert len(val.accumulated_text) > 0

    val.reset()
    assert val.accumulated_text == ""
    assert len(val.token_history) == 0


def test_exception_classes() -> None:
    e1 = SchemaCompilationError("Failed", schema_path="/test", details={"code": 1})
    assert e1.schema_path == "/test"
    assert e1.details["code"] == 1

    e2 = RegexCompilationError("Invalid", pattern="[a-", details={"col": 2})
    assert e2.pattern == "[a-"

    e3 = GrammarExhaustedError("Exhausted")
    assert isinstance(e3, MaskingError)
    assert isinstance(e3, GrammarForgeError)
