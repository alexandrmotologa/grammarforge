"""Unit tests for JSON Schema to GBNF compilation."""

import pytest
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.core.config import CompilerConfig, KeyOrdering, WhitespaceMode


def test_compile_primitive_types() -> None:
    # String
    gbnf_str = compile_json_schema({"type": "string"})
    assert "root ::= json-string" in gbnf_str

    # Integer
    gbnf_int = compile_json_schema({"type": "integer"})
    assert "root ::= json-int" in gbnf_int

    # Number
    gbnf_num = compile_json_schema({"type": "number"})
    assert "root ::= json-number" in gbnf_num

    # Boolean
    gbnf_bool = compile_json_schema({"type": "boolean"})
    assert "root ::= json-bool" in gbnf_bool

    # Null
    gbnf_null = compile_json_schema({"type": "null"})
    assert "root ::= json-null" in gbnf_null


def test_compile_enum_and_const() -> None:
    schema_enum = {"type": "string", "enum": ["red", "green", "blue"]}
    gbnf_enum = compile_json_schema(schema_enum)
    assert '"\\"red\\""' in gbnf_enum or '"red"' in gbnf_enum

    schema_const = {"const": "exact_token"}
    gbnf_const = compile_json_schema(schema_const)
    assert '"\\"exact_token\\""' in gbnf_const


def test_compile_regex_pattern_in_schema() -> None:
    schema = {
        "type": "string",
        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
    }
    gbnf = compile_json_schema(schema)
    assert "root ::= root-pattern" in gbnf or "pattern" in gbnf
    assert "[0-9]" in gbnf
    assert '"-"' in gbnf


def test_compile_format_in_schema() -> None:
    schema = {"type": "string", "format": "email"}
    gbnf = compile_json_schema(schema)
    assert "fmt-email" in gbnf
    assert '"@"' in gbnf


def test_compile_object_canonical_and_flexible() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    # Canonical
    gbnf_canonical = compile_json_schema(schema, CompilerConfig(key_ordering=KeyOrdering.CANONICAL))
    assert '"\\\"name\\\""' in gbnf_canonical or '"name"' in gbnf_canonical
    assert '"\\\"age\\\""' in gbnf_canonical or '"age"' in gbnf_canonical

    # Flexible
    gbnf_flexible = compile_json_schema(schema, CompilerConfig(key_ordering=KeyOrdering.FLEXIBLE))
    assert "member" in gbnf_flexible


def test_compile_array() -> None:
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
    }
    gbnf = compile_json_schema(schema)
    assert '"["' in gbnf
    assert '"]"' in gbnf
    assert "item" in gbnf


def test_compile_recursive_ref() -> None:
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "val": {"type": "integer"},
                    "left": {"$ref": "#/$defs/Node"},
                    "right": {"$ref": "#/$defs/Node"},
                },
                "required": ["val"],
            }
        },
        "$ref": "#/$defs/Node",
    }
    gbnf = compile_json_schema(schema)
    assert "ref-node" in gbnf
