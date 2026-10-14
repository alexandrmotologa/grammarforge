"""Deterministic JSON auto-repair and stream healer for truncated LLM completions."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus


class RepairResult:
    """Result of attempting to repair an incomplete JSON string."""

    def __init__(
        self,
        original_text: str,
        repaired_text: str,
        suffix_added: str,
        is_valid_json: bool,
        conforms_to_schema: bool,
        error: Optional[str] = None,
    ) -> None:
        self.original_text = original_text
        self.repaired_text = repaired_text
        self.suffix_added = suffix_added
        self.is_valid_json = is_valid_json
        self.conforms_to_schema = conforms_to_schema
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original_text),
            "repaired_length": len(self.repaired_text),
            "suffix_added": self.suffix_added,
            "is_valid_json": self.is_valid_json,
            "conforms_to_schema": self.conforms_to_schema,
            "error": self.error,
        }


def heal_partial_json(
    partial_text: str,
    schema: Optional[Dict[str, Any]] = None,
) -> RepairResult:
    """Calculate and append the minimal valid suffix to close truncated JSON strings.

    Handles:
    - Unclosed string literals (appends closing quote '"')
    - Dangling colons in object key-value pairs (appends null or schema-appropriate placeholder)
    - Dangling commas (strips trailing commas before closing braces/brackets)
    - Unmatched opening brackets '[' and braces '{' (closes in LIFO order)
    """
    text = partial_text.strip()
    if not text:
        empty_target = "{}" if schema and schema.get("type") == "object" else "[]"
        return RepairResult(
            original_text=partial_text,
            repaired_text=empty_target,
            suffix_added=empty_target,
            is_valid_json=True,
            conforms_to_schema=True,
        )

    # First check if text is already valid complete JSON
    try:
        parsed = json.loads(text)
        conforms = True
        err = None
        if schema:
            sm = JsonStateMachine(schema)
            conforms, err = sm._validate_complete(parsed, schema)
        return RepairResult(
            original_text=partial_text,
            repaired_text=text,
            suffix_added="",
            is_valid_json=True,
            conforms_to_schema=conforms,
            error=err,
        )
    except json.JSONDecodeError:
        pass

    # Inspect character stream to trace balance and active state
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
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
        i += 1

    suffix_parts: List[str] = []
    repaired_base = text

    # Step 1: Close dangling string quote
    if in_string:
        if escape:
            # Drop trailing backslash before closing
            repaired_base = repaired_base[:-1]
        repaired_base += '"'
        suffix_parts.append('"')

    # Strip any trailing whitespace
    repaired_base_stripped = repaired_base.rstrip()

    # Step 2: Handle dangling colon (e.g. `{"key": `)
    if repaired_base_stripped.endswith(":"):
        placeholder = "null"
        repaired_base = repaired_base_stripped + " " + placeholder
        suffix_parts.append(" " + placeholder)
    elif repaired_base_stripped.endswith(","):
        # Strip trailing comma before closing structures
        repaired_base = repaired_base_stripped[:-1]
        suffix_parts.append("(stripped trailing comma)")

    # Step 3: Close remaining open brackets and braces in reverse order
    for delimiter in reversed(stack):
        if delimiter == "{":
            repaired_base += "}"
            suffix_parts.append("}")
        elif delimiter == "[":
            repaired_base += "]"
            suffix_parts.append("]")

    # Verify repaired JSON
    conforms = False
    err = None
    is_valid_json = False
    try:
        parsed = json.loads(repaired_base)
        is_valid_json = True
        if schema:
            sm = JsonStateMachine(schema)
            conforms, err = sm._validate_complete(parsed, schema)
        else:
            conforms = True
    except json.JSONDecodeError as decode_err:
        err = f"Failed to parse repaired JSON: {decode_err}"

    added_suffix = "".join(p for p in suffix_parts if not p.startswith("("))
    return RepairResult(
        original_text=partial_text,
        repaired_text=repaired_base,
        suffix_added=added_suffix,
        is_valid_json=is_valid_json,
        conforms_to_schema=conforms,
        error=err,
    )
