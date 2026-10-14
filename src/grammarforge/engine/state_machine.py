"""Incremental grammar state tracker and partial syntax validator."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class ParserStatus:
    """Status returned after attempting to advance the state."""

    VALID = "valid"          # Prefix is valid and incomplete
    ACCEPTING = "accepting"  # Prefix is valid and forms a complete acceptable target
    REJECTED = "rejected"    # Prefix violates grammar rules


class JsonStateMachine:
    """Fast incremental state machine for validating partial JSON strings against a schema."""

    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema
        self._compiled_patterns: Dict[str, re.Pattern] = {}

    def _get_pattern(self, pattern_str: str) -> re.Pattern:
        if pattern_str not in self._compiled_patterns:
            self._compiled_patterns[pattern_str] = re.compile(pattern_str)
        return self._compiled_patterns[pattern_str]

    def check_partial_json(self, text: str) -> Tuple[str, Optional[str]]:
        """Evaluate a partial JSON string.

        Returns:
            Tuple of (status, error_message)
            status is one of 'accepting', 'valid', or 'rejected'.
        """
        stripped = text.strip()
        if not stripped:
            return ParserStatus.VALID, None

        # Try parsing as complete JSON
        try:
            parsed = json.loads(text)
            # If successfully parsed as complete JSON, validate against schema
            is_valid, err = self._validate_complete(parsed, self.schema)
            if is_valid:
                return ParserStatus.ACCEPTING, None
            else:
                return ParserStatus.REJECTED, err
        except json.JSONDecodeError:
            pass

        # Text is incomplete. Check if prefix syntax is valid so far
        is_struct_valid, struct_err = self._check_structural_prefix(text)
        if not is_struct_valid:
            return ParserStatus.REJECTED, struct_err

        return ParserStatus.VALID, None

    def _check_structural_prefix(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate JSON structural balance (quotes, braces, brackets, commas, colons)."""
        in_string = False
        escape = False
        stack: List[str] = []

        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch == "}":
                    if not stack or stack[-1] != "{":
                        return False, "Unmatched closing brace '}'"
                    stack.pop()
                elif ch == "]":
                    if not stack or stack[-1] != "[":
                        return False, "Unmatched closing bracket ']'"
                    stack.pop()
                elif ch in " \t\n\r":
                    pass
                elif ch in ":,":
                    pass
                elif ch in "-0123456789.eE":
                    pass
                elif text[i:i + 4] == "true":
                    i += 3
                elif text[i:i + 5] == "false":
                    i += 4
                elif text[i:i + 4] == "null":
                    i += 3
                elif any(word.startswith(text[i:]) for word in ["true", "false", "null"]):
                    # Incomplete literal at the end of input
                    break
                else:
                    # Unrecognized token outside string
                    return False, f"Unexpected character '{ch}' at position {i}"
            i += 1

        return True, None

    def _validate_complete(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate fully parsed JSON object against JSON Schema constraints."""
        schema_type = schema.get("type")

        # Enum check
        if "enum" in schema:
            if data not in schema["enum"]:
                return False, f"Value {data!r} not in enum {schema['enum']}"

        # Const check
        if "const" in schema:
            if data != schema["const"]:
                return False, f"Value {data!r} != const {schema['const']!r}"

        # Union types
        if isinstance(schema_type, list):
            for t in schema_type:
                sub = dict(schema)
                sub["type"] = t
                ok, _ = self._validate_complete(data, sub)
                if ok:
                    return True, None
            return False, f"Value does not match any union type {schema_type}"

        # Primitive type checks
        if schema_type == "string":
            if not isinstance(data, str):
                return False, f"Expected string, got {type(data).__name__}"
            if "pattern" in schema:
                pat = self._get_pattern(schema["pattern"])
                if not pat.search(data):
                    return False, f"String does not match pattern {schema['pattern']}"
            if "minLength" in schema and len(data) < schema["minLength"]:
                return False, f"String length {len(data)} < minLength {schema['minLength']}"
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                return False, f"String length {len(data)} > maxLength {schema['maxLength']}"

        elif schema_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                return False, f"Expected integer, got {type(data).__name__}"
            if "minimum" in schema and data < schema["minimum"]:
                return False, f"Integer {data} < minimum {schema['minimum']}"
            if "maximum" in schema and data > schema["maximum"]:
                return False, f"Integer {data} > maximum {schema['maximum']}"

        elif schema_type == "number":
            if not isinstance(data, (int, float)) or isinstance(data, bool):
                return False, f"Expected number, got {type(data).__name__}"
            if "minimum" in schema and data < schema["minimum"]:
                return False, f"Number {data} < minimum {schema['minimum']}"
            if "maximum" in schema and data > schema["maximum"]:
                return False, f"Number {data} > maximum {schema['maximum']}"

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                return False, f"Expected boolean, got {type(data).__name__}"

        elif schema_type == "null":
            if data is not None:
                return False, f"Expected null, got {data!r}"

        elif schema_type == "object":
            if not isinstance(data, dict):
                return False, f"Expected object, got {type(data).__name__}"
            required = schema.get("required", [])
            for req in required:
                if req not in data:
                    return False, f"Missing required property {req!r}"
            properties = schema.get("properties", {})
            for key, val in data.items():
                if key in properties:
                    ok, err = self._validate_complete(val, properties[key])
                    if not ok:
                        return False, f"Property {key!r}: {err}"
                elif schema.get("additionalProperties") is False:
                    return False, f"Disallowed additional property {key!r}"

        elif schema_type == "array":
            if not isinstance(data, list):
                return False, f"Expected array, got {type(data).__name__}"
            if "minItems" in schema and len(data) < schema["minItems"]:
                return False, f"Array length {len(data)} < minItems {schema['minItems']}"
            if "maxItems" in schema and len(data) > schema["maxItems"]:
                return False, f"Array length {len(data)} > maxItems {schema['maxItems']}"
            items_schema = schema.get("items")
            if isinstance(items_schema, dict):
                for idx, item in enumerate(data):
                    ok, err = self._validate_complete(item, items_schema)
                    if not ok:
                        return False, f"Array item [{idx}]: {err}"

        return True, None
