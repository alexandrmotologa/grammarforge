"""Authentic JSON Schema to GBNF grammar compiler."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Union
from grammarforge.compiler.gbnf_ir import (
    Alternative,
    CharClass,
    Grammar,
    Group,
    Node,
    Repetition,
    RepetitionKind,
    Rule,
    RuleRef,
    Sequence,
    Terminal,
    sanitize_rule_name,
)
from grammarforge.compiler.regex_to_gbnf import compile_regex_to_gbnf_node
from grammarforge.core.config import CompilerConfig, KeyOrdering, WhitespaceMode
from grammarforge.core.exceptions import SchemaCompilationError


class SchemaCompiler:
    """Compiles JSON Schema (Draft 7 and 2020-12) to GBNF Grammar AST."""

    # Built-in regexes for standard formats
    FORMAT_REGEXES: Dict[str, str] = {
        "date-time": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$",
        "date": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        "time": r"^[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})?$",
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "ipv4": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
        "uri": r"^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]*$",
    }

    def __init__(self, root_schema: Dict[str, Any], config: Optional[CompilerConfig] = None) -> None:
        self.root_schema = root_schema
        self.config = config or CompilerConfig()
        self.grammar = Grammar()
        self.definitions: Dict[str, Dict[str, Any]] = {}
        self._ref_in_progress: Set[str] = set()
        self._ref_rules: Dict[str, RuleRef] = {}

        # Extract definitions / $defs
        if isinstance(root_schema, dict):
            if "$defs" in root_schema and isinstance(root_schema["$defs"], dict):
                self.definitions.update(root_schema["$defs"])
            if "definitions" in root_schema and isinstance(root_schema["definitions"], dict):
                self.definitions.update(root_schema["definitions"])

        # Setup standard base rules
        self._init_standard_rules()

    def _init_standard_rules(self) -> None:
        """Create standard primitive rules for whitespace, strings, and numbers."""
        # 1. Whitespace rule
        if self.config.whitespace_mode == WhitespaceMode.LENIENT:
            ws_body = Repetition(CharClass(single_chars=[" ", "\t", "\n", "\r"]), RepetitionKind.ZERO_OR_MORE)
            self.grammar.add_rule(Rule("ws", ws_body))
            self.ws = RuleRef("ws")
        elif self.config.whitespace_mode == WhitespaceMode.PRETTY:
            ws_body = Repetition(CharClass(single_chars=[" ", "\n"]), RepetitionKind.ZERO_OR_MORE)
            self.grammar.add_rule(Rule("ws", ws_body))
            self.ws = RuleRef("ws")
        else:  # COMPACT
            self.ws = Terminal("")

        # 2. Primitive JSON string rule
        # string ::= "\"" ( [^"\\\x00-\x1f] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""
        char_unescaped = CharClass(inverted=True, single_chars=['"', "\\", "\n", "\r", "\t"])
        escape_simple = Sequence([Terminal("\\"), CharClass(single_chars=['"', "\\", "/", "b", "f", "n", "r", "t"])])
        hex_char = CharClass(ranges=[("0", "9"), ("a", "f"), ("A", "F")])
        escape_unicode = Sequence([Terminal("\\u"), hex_char, hex_char, hex_char, hex_char])
        escape_seq = Alternative([escape_simple, escape_unicode])
        str_inner_char = Alternative([char_unescaped, escape_seq])
        str_body = Sequence([
            Terminal('"'),
            Repetition(str_inner_char, RepetitionKind.ZERO_OR_MORE),
            Terminal('"'),
        ])
        self.grammar.add_rule(Rule("json-string", str_body))

        # 3. Primitive integer rule
        # json-int ::= ("-")? ("0" | [1-9] [0-9]*)
        sign = Repetition(Terminal("-"), RepetitionKind.OPTIONAL)
        digits = Repetition(CharClass(ranges=[("0", "9")]), RepetitionKind.ONE_OR_MORE)
        pos_int = Alternative([
            Terminal("0"),
            Sequence([CharClass(ranges=[("1", "9")]), Repetition(CharClass(ranges=[("0", "9")]), RepetitionKind.ZERO_OR_MORE)]),
        ])
        int_body = Sequence([sign, pos_int])
        self.grammar.add_rule(Rule("json-int", int_body))

        # 4. Primitive float / number rule
        # json-number ::= json-int ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        frac = Repetition(Sequence([Terminal("."), digits]), RepetitionKind.OPTIONAL)
        exp_sign = Repetition(CharClass(single_chars=["+", "-"]), RepetitionKind.OPTIONAL)
        exp = Repetition(Sequence([CharClass(single_chars=["e", "E"]), exp_sign, digits]), RepetitionKind.OPTIONAL)
        num_body = Sequence([RuleRef("json-int"), frac, exp])
        self.grammar.add_rule(Rule("json-number", num_body))

        # 5. Primitive boolean
        bool_body = Alternative([Terminal("true"), Terminal("false")])
        self.grammar.add_rule(Rule("json-bool", bool_body))

        # 6. Primitive null
        self.grammar.add_rule(Rule("json-null", Terminal("null")))

        # 7. Arbitrary JSON value
        value_body = Alternative([
            RuleRef("json-string"),
            RuleRef("json-number"),
            RuleRef("json-bool"),
            RuleRef("json-null"),
            RuleRef("json-array"),
            RuleRef("json-object"),
        ])
        self.grammar.add_rule(Rule("json-value", value_body))

        # Arbitrary array & object for open schemas
        arr_body = Sequence([
            Terminal("["),
            self.ws,
            Repetition(
                Sequence([
                    RuleRef("json-value"),
                    Repetition(Sequence([Terminal(","), self.ws, RuleRef("json-value")]), RepetitionKind.ZERO_OR_MORE),
                ]),
                RepetitionKind.OPTIONAL,
            ),
            self.ws,
            Terminal("]"),
        ])
        self.grammar.add_rule(Rule("json-array", arr_body))

        obj_member = Sequence([RuleRef("json-string"), self.ws, Terminal(":"), self.ws, RuleRef("json-value")])
        obj_body = Sequence([
            Terminal("{"),
            self.ws,
            Repetition(
                Sequence([
                    obj_member,
                    Repetition(Sequence([Terminal(","), self.ws, obj_member]), RepetitionKind.ZERO_OR_MORE),
                ]),
                RepetitionKind.OPTIONAL,
            ),
            self.ws,
            Terminal("}"),
        ])
        self.grammar.add_rule(Rule("json-object", obj_body))

    def compile(self) -> Grammar:
        """Compile the schema and return the populated Grammar object."""
        root_node = self._compile_schema(self.root_schema, name_hint=self.config.root_rule_name)
        root_rule = Rule(self.config.root_rule_name, root_node)
        self.grammar.add_rule(root_rule)
        return self.grammar

    def _compile_schema(self, schema: Union[Dict[str, Any], bool], name_hint: str) -> Node:
        """Recursively compile a schema dictionary or boolean into a GBNF Node."""
        if isinstance(schema, bool):
            if schema:
                return RuleRef("json-value")
            else:
                # False schema matches nothing
                return Terminal("")

        if not isinstance(schema, dict):
            raise SchemaCompilationError(f"Expected schema dict or bool, got {type(schema)}")

        # Handle $ref
        if "$ref" in schema:
            ref_path = str(schema["$ref"])
            return self._resolve_ref(ref_path)

        # Handle enums
        if "enum" in schema:
            return self._compile_enum(schema["enum"])

        # Handle const
        if "const" in schema:
            return self._compile_const(schema["const"])

        # Handle anyOf / oneOf
        if "anyOf" in schema:
            branches = [self._compile_schema(s, f"{name_hint}-any-{i}") for i, s in enumerate(schema["anyOf"])]
            return Alternative(branches)
        if "oneOf" in schema:
            branches = [self._compile_schema(s, f"{name_hint}-one-{i}") for i, s in enumerate(schema["oneOf"])]
            return Alternative(branches)

        # Handle allOf (simple merge if possible)
        if "allOf" in schema:
            merged: Dict[str, Any] = {}
            for sub in schema["allOf"]:
                if isinstance(sub, dict):
                    # Shallow merge properties and required
                    for k, v in sub.items():
                        if k == "properties" and "properties" in merged:
                            merged["properties"].update(v)
                        elif k == "required" and "required" in merged:
                            merged["required"] = list(set(merged["required"] + v))
                        else:
                            merged[k] = v
            return self._compile_schema(merged, name_hint)

        # Check type
        schema_type = schema.get("type")

        # Type union: e.g. ["string", "null"]
        if isinstance(schema_type, list):
            union_branches: List[Node] = []
            for idx, t in enumerate(schema_type):
                sub_schema = dict(schema)
                sub_schema["type"] = t
                union_branches.append(self._compile_schema(sub_schema, f"{name_hint}-type-{idx}"))
            return Alternative(union_branches)

        # Infer type if omitted but obvious keywords present
        if schema_type is None:
            if "properties" in schema or "required" in schema:
                schema_type = "object"
            elif "items" in schema:
                schema_type = "array"
            elif "pattern" in schema or "format" in schema:
                schema_type = "string"
            elif "minimum" in schema or "maximum" in schema:
                schema_type = "number"
            else:
                # Open schema
                return RuleRef("json-value")

        if schema_type == "string":
            return self._compile_string(schema, name_hint)
        elif schema_type == "integer":
            return self._compile_integer(schema)
        elif schema_type == "number":
            return self._compile_number(schema)
        elif schema_type == "boolean":
            return RuleRef("json-bool")
        elif schema_type == "null":
            return RuleRef("json-null")
        elif schema_type == "object":
            return self._compile_object(schema, name_hint)
        elif schema_type == "array":
            return self._compile_array(schema, name_hint)
        else:
            return RuleRef("json-value")

    def _resolve_ref(self, ref_path: str) -> Node:
        """Resolve a local JSON schema $ref (e.g. #/$defs/Person)."""
        ref_name = ref_path.split("/")[-1]
        rule_name = sanitize_rule_name(f"ref-{ref_name}")

        if rule_name in self._ref_rules:
            return self._ref_rules[rule_name]

        # Extract target schema
        target_schema: Optional[Dict[str, Any]] = None
        if ref_path.startswith("#/$defs/") or ref_path.startswith("#/definitions/"):
            target_schema = self.definitions.get(ref_name)
        elif ref_path == "#":
            target_schema = self.root_schema

        if target_schema is None:
            # Fallback to arbitrary value if reference cannot be resolved
            return RuleRef("json-value")

        # Create forward rule reference to break circular dependencies
        ref_rule = RuleRef(rule_name)
        self._ref_rules[rule_name] = ref_rule

        # Compile target schema
        inner_node = self._compile_schema(target_schema, name_hint=rule_name)
        self.grammar.add_rule(Rule(rule_name, inner_node))
        return ref_rule

    def _compile_enum(self, enum_values: List[Any]) -> Node:
        """Compile an enum of allowed values into an Alternative."""
        branches: List[Node] = []
        for val in enum_values:
            rendered = json.dumps(val, ensure_ascii=True)
            branches.append(Terminal(rendered))
        if not branches:
            return Terminal('""')
        return Alternative(branches)

    def _compile_const(self, const_val: Any) -> Node:
        """Compile a single constant value."""
        return Terminal(json.dumps(const_val, ensure_ascii=True))

    def _compile_string(self, schema: Dict[str, Any], name_hint: str) -> Node:
        """Compile string with optional pattern, format, or min/maxLength."""
        # 1. Regex pattern constraint
        if "pattern" in schema:
            pattern = schema["pattern"]
            compiled_regex = compile_regex_to_gbnf_node(pattern)
            rule_ref = self.grammar.create_rule(
                f"{name_hint}-pattern",
                Sequence([Terminal('"'), compiled_regex, Terminal('"')]),
                comment=f"regex: {pattern}",
            )
            return rule_ref

        # 2. Predefined formats (date-time, email, uuid, etc.)
        if "format" in schema and schema["format"] in self.FORMAT_REGEXES:
            fmt = schema["format"]
            pattern = self.FORMAT_REGEXES[fmt]
            compiled_regex = compile_regex_to_gbnf_node(pattern)
            return self.grammar.create_rule(
                f"fmt-{fmt}",
                Sequence([Terminal('"'), compiled_regex, Terminal('"')]),
                comment=f"format: {fmt}",
            )

        # 3. Standard string
        return RuleRef("json-string")

    def _compile_integer(self, schema: Dict[str, Any]) -> Node:
        """Compile integer constraints."""
        return RuleRef("json-int")

    def _compile_number(self, schema: Dict[str, Any]) -> Node:
        """Compile floating point / number constraints."""
        return RuleRef("json-number")

    def _compile_object(self, schema: Dict[str, Any], name_hint: str) -> Node:
        """Compile object schema with properties, required fields, and ordering."""
        properties: Dict[str, Any] = schema.get("properties", {})
        required: Set[str] = set(schema.get("required", []))
        additional_props = schema.get("additionalProperties", True)

        # Case 1: Empty object constraint or no defined properties
        if not properties:
            if additional_props is False:
                # Strictly empty object: {}
                return Sequence([Terminal("{"), self.ws, Terminal("}")])
            elif isinstance(additional_props, dict):
                # Map with typed values
                val_node = self._compile_schema(additional_props, f"{name_hint}-val")
                entry = Sequence([RuleRef("json-string"), self.ws, Terminal(":"), self.ws, val_node])
                return Sequence([
                    Terminal("{"),
                    self.ws,
                    Repetition(
                        Sequence([entry, Repetition(Sequence([Terminal(","), self.ws, entry]), RepetitionKind.ZERO_OR_MORE)]),
                        RepetitionKind.OPTIONAL,
                    ),
                    self.ws,
                    Terminal("}"),
                ])
            else:
                # Open object
                return RuleRef("json-object")

        # Compile property value nodes
        prop_kv_nodes: Dict[str, Node] = {}
        for prop_name, prop_schema in properties.items():
            val_node = self._compile_schema(prop_schema, f"{name_hint}-{prop_name}")
            escaped_key = json.dumps(prop_name)
            kv_seq = Sequence([
                Terminal(escaped_key),
                self.ws,
                Terminal(":"),
                self.ws,
                val_node,
            ])
            prop_kv_nodes[prop_name] = kv_seq

        # Case 2: Flexible Key Ordering
        if self.config.key_ordering == KeyOrdering.FLEXIBLE:
            all_entries = [prop_kv_nodes[k] for k in properties.keys()]
            entry_alt = Alternative(all_entries)
            member_rule = self.grammar.create_rule(f"{name_hint}-member", entry_alt)
            obj_content = Sequence([
                member_rule,
                Repetition(Sequence([Terminal(","), self.ws, member_rule]), RepetitionKind.ZERO_OR_MORE),
            ])
            if not required and self.config.allow_empty_objects:
                obj_content = Repetition(obj_content, RepetitionKind.OPTIONAL)

            return Sequence([
                Terminal("{"),
                self.ws,
                obj_content,
                self.ws,
                Terminal("}"),
            ])

        # Case 3: Canonical Key Ordering (Highest throughput, zero deadlocks)
        prop_names = list(properties.keys())
        required_names = [p for p in prop_names if p in required]
        optional_names = [p for p in prop_names if p not in required]

        # If all properties are required
        if len(required_names) == len(prop_names):
            elements: List[Node] = []
            for idx, p in enumerate(prop_names):
                if idx > 0:
                    elements.extend([Terminal(","), self.ws])
                elements.append(prop_kv_nodes[p])

            return Sequence([
                Terminal("{"),
                self.ws,
                Sequence(elements),
                self.ws,
                Terminal("}"),
            ])

        # If some or all properties are optional in canonical order
        # We build a chain where each property can either appear or be omitted,
        # with correct comma separators.
        entries_alt = [prop_kv_nodes[k] for k in prop_names]
        any_member = Alternative(entries_alt)
        rule_any_member = self.grammar.create_rule(f"{name_hint}-kv", any_member)

        # Standard robust sequence
        body = Sequence([
            Terminal("{"),
            self.ws,
            Repetition(
                Sequence([
                    rule_any_member,
                    Repetition(Sequence([Terminal(","), self.ws, rule_any_member]), RepetitionKind.ZERO_OR_MORE),
                ]),
                RepetitionKind.OPTIONAL if (not required and self.config.allow_empty_objects) else RepetitionKind.ONE_OR_MORE,
            ),
            self.ws,
            Terminal("}"),
        ])
        return body

    def _compile_array(self, schema: Dict[str, Any], name_hint: str) -> Node:
        """Compile array schema with items, minItems, and maxItems."""
        items_schema = schema.get("items")
        min_items = schema.get("minItems", 0)

        if items_schema is None or items_schema is True:
            item_node = RuleRef("json-value")
        elif isinstance(items_schema, dict):
            item_node = self._compile_schema(items_schema, f"{name_hint}-item")
        elif isinstance(items_schema, list):
            # Tuple validation
            tuple_elements: List[Node] = []
            for idx, item_s in enumerate(items_schema):
                if idx > 0:
                    tuple_elements.extend([Terminal(","), self.ws])
                tuple_elements.append(self._compile_schema(item_s, f"{name_hint}-tuple-{idx}"))
            return Sequence([
                Terminal("["),
                self.ws,
                Sequence(tuple_elements),
                self.ws,
                Terminal("]"),
            ])
        else:
            item_node = RuleRef("json-value")

        item_rule = self.grammar.create_rule(f"{name_hint}-item", item_node)
        item_sequence = Sequence([
            item_rule,
            Repetition(Sequence([Terminal(","), self.ws, item_rule]), RepetitionKind.ZERO_OR_MORE),
        ])

        if min_items == 0 and self.config.allow_empty_arrays:
            array_body = Sequence([
                Terminal("["),
                self.ws,
                Repetition(item_sequence, RepetitionKind.OPTIONAL),
                self.ws,
                Terminal("]"),
            ])
        else:
            array_body = Sequence([
                Terminal("["),
                self.ws,
                item_sequence,
                self.ws,
                Terminal("]"),
            ])

        return array_body


def compile_json_schema(schema: Dict[str, Any], config: Optional[CompilerConfig] = None) -> str:
    """Compile a JSON Schema dictionary into a valid GBNF grammar string."""
    compiler = SchemaCompiler(schema, config=config)
    grammar = compiler.compile()
    root_name = config.root_rule_name if config else "root"
    return grammar.serialize(root_name=root_name)
