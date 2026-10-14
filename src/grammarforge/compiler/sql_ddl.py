"""SQL DDL schema parser and compiler to GBNF grammars for structured database generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from grammarforge.compiler.gbnf_ir import (
    Alternative,
    CharClass,
    Grammar,
    Node,
    Repetition,
    RepetitionKind,
    Rule,
    RuleRef,
    Sequence,
    Terminal,
)
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.core.config import CompilerConfig


class ColumnDef:
    """Parsed SQL column definition."""

    def __init__(
        self,
        name: str,
        data_type: str,
        is_nullable: bool = True,
        is_primary_key: bool = False,
        enum_values: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.data_type = data_type.upper()
        self.is_nullable = is_nullable
        self.is_primary_key = is_primary_key
        self.enum_values = enum_values or []

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert column type to equivalent JSON Schema type definition."""
        if self.enum_values:
            return {"type": "string", "enum": self.enum_values}

        dt = self.data_type
        if any(t in dt for t in ("INT", "SERIAL")):
            return {"type": "integer"}
        elif any(t in dt for t in ("FLOAT", "REAL", "DOUBLE", "NUMERIC", "DECIMAL")):
            return {"type": "number"}
        elif any(t in dt for t in ("BOOL", "BOOLEAN")):
            return {"type": "boolean"}
        elif "DATE" in dt or "TIME" in dt:
            return {"type": "string", "format": "date-time"}
        elif "UUID" in dt:
            return {"type": "string", "format": "uuid"}
        elif "JSON" in dt:
            return {}
        else:
            return {"type": "string"}


class TableDef:
    """Parsed SQL table definition."""

    def __init__(self, name: str, columns: List[ColumnDef]) -> None:
        self.name = name
        self.columns = columns

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert entire table into a row-level JSON Schema."""
        props: Dict[str, Any] = {}
        required: List[str] = []

        for col in self.columns:
            props[col.name] = col.to_json_schema()
            if not col.is_nullable or col.is_primary_key:
                required.append(col.name)

        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }


def parse_create_table_ddl(sql: str) -> TableDef:
    """Parse standard CREATE TABLE DDL statement into structured TableDef."""
    # Match CREATE TABLE [IF NOT EXISTS] table_name ( ... )
    table_match = re.search(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[\"`\[]?\w+[\"`\]]?\.)?[\"`\[]?([a-zA-Z0-9_]+)[\"`\]]?\s*\((.*)\)",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not table_match:
        raise ValueError("Invalid SQL DDL: Could not find 'CREATE TABLE table_name (...)' block.")

    table_name = table_match.group(1)
    body = table_match.group(2).strip()

    # Split column definitions by top-level commas (ignoring commas inside parentheses)
    col_chunks: List[str] = []
    depth = 0
    curr: List[str] = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            col_chunks.append("".join(curr).strip())
            curr.clear()
            continue
        curr.append(char)
    if curr:
        col_chunks.append("".join(curr).strip())

    columns: List[ColumnDef] = []
    for chunk in col_chunks:
        chunk_clean = chunk.strip()
        # Skip table constraints like PRIMARY KEY (col), CONSTRAINT, FOREIGN KEY
        if re.match(r"^(PRIMARY\s+KEY|FOREIGN\s+KEY|CONSTRAINT|UNIQUE|CHECK)\b", chunk_clean, re.I):
            continue

        parts = chunk_clean.split()
        if len(parts) < 2:
            continue

        raw_col_name = re.sub(r"[\"`\[\]]", "", parts[0])
        col_type = parts[1].upper()

        upper_chunk = chunk_clean.upper()
        is_primary = "PRIMARY KEY" in upper_chunk
        is_nullable = "NOT NULL" not in upper_chunk and not is_primary

        # Check for CHECK(col IN ('val1', 'val2')) enum constraint
        enum_vals: List[str] = []
        check_match = re.search(r"CHECK\s*\(\s*\w+\s+IN\s*\((.*?)\)\s*\)", chunk_clean, re.I)
        if check_match:
            raw_enums = check_match.group(1).split(",")
            enum_vals = [re.sub(r"[\"'\s]", "", v) for v in raw_enums]

        columns.append(
            ColumnDef(
                name=raw_col_name,
                data_type=col_type,
                is_nullable=is_nullable,
                is_primary_key=is_primary,
                enum_values=enum_vals,
            )
        )

    return TableDef(name=table_name, columns=columns)


def compile_sql_ddl_to_gbnf(
    sql: str,
    output_format: str = "json",
    config: Optional[CompilerConfig] = None,
) -> str:
    """Compile SQL DDL schema to a GBNF grammar.

    Args:
        sql: Raw CREATE TABLE SQL statement.
        output_format: 'json' (valid row object) or 'insert' (valid SQL INSERT statement).
        config: Compiler configuration.
    """
    table = parse_create_table_ddl(sql)

    if output_format == "json":
        schema = table.to_json_schema()
        return compile_json_schema(schema, config=config)

    # Format 'insert' SQL statement grammar
    grammar = Grammar()

    # ws
    ws = RuleRef("ws")
    ws_body = Repetition(CharClass(single_chars=[" ", "\t", "\n"]), RepetitionKind.ZERO_OR_MORE)
    grammar.add_rule(Rule("ws", ws_body))

    # Primitive SQL values
    sql_str = Sequence([
        Terminal("'"),
        Repetition(CharClass(inverted=True, single_chars=["'"]), RepetitionKind.ZERO_OR_MORE),
        Terminal("'"),
    ])
    grammar.add_rule(Rule("sql-string", sql_str))

    sql_int = Sequence([
        Repetition(Terminal("-"), RepetitionKind.OPTIONAL),
        Repetition(CharClass(ranges=[("0", "9")]), RepetitionKind.ONE_OR_MORE),
    ])
    grammar.add_rule(Rule("sql-int", sql_int))

    sql_num = Sequence([
        sql_int,
        Repetition(Sequence([Terminal("."), Repetition(CharClass(ranges=[("0", "9")]), RepetitionKind.ONE_OR_MORE)]), RepetitionKind.OPTIONAL),
    ])
    grammar.add_rule(Rule("sql-num", sql_num))

    # Values list according to column types
    val_nodes: List[Node] = []
    for idx, col in enumerate(table.columns):
        if idx > 0:
            val_nodes.extend([Terminal(","), ws])

        dt = col.data_type
        if col.enum_values:
            enum_branches = [Terminal(f"'{v}'") for v in col.enum_values]
            node = Alternative(enum_branches)
        elif any(t in dt for t in ("INT", "SERIAL")):
            node = RuleRef("sql-int")
        elif any(t in dt for t in ("FLOAT", "REAL", "DOUBLE", "NUMERIC")):
            node = RuleRef("sql-num")
        else:
            node = RuleRef("sql-string")

        if col.is_nullable and not col.is_primary_key:
            node = Alternative([node, Terminal("NULL")])

        val_nodes.append(node)

    values_seq = Sequence(val_nodes)

    col_names_str = ", ".join(c.name for c in table.columns)
    root_seq = Sequence([
        Terminal(f"INSERT INTO {table.name} ({col_names_str}) VALUES ("),
        ws,
        values_seq,
        ws,
        Terminal(");"),
    ])

    grammar.add_rule(Rule("root", root_seq, comment=f"SQL Insert for table {table.name}"))
    return grammar.serialize(root_name="root")
