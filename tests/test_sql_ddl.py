"""Unit tests for SQL DDL to GBNF compiler."""

import pytest
from grammarforge.compiler.sql_ddl import compile_sql_ddl_to_gbnf, parse_create_table_ddl


SQL_SAMPLE = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    is_active BOOLEAN,
    status VARCHAR(20) CHECK(status IN ('active', 'pending', 'banned')),
    created_at TIMESTAMP
);
"""


def test_parse_create_table() -> None:
    table = parse_create_table_ddl(SQL_SAMPLE)
    assert table.name == "users"
    assert len(table.columns) == 6

    col_names = [c.name for c in table.columns]
    assert "id" in col_names
    assert "email" in col_names
    assert "status" in col_names

    status_col = next(c for c in table.columns if c.name == "status")
    assert status_col.enum_values == ["active", "pending", "banned"]


def test_compile_sql_to_json_gbnf() -> None:
    gbnf = compile_sql_ddl_to_gbnf(SQL_SAMPLE, output_format="json")
    assert "root" in gbnf
    assert '"username"' in gbnf or '"\\\"username\\\""' in gbnf
    assert '"active"' in gbnf or '"\\\"active\\\""' in gbnf


def test_compile_sql_to_insert_gbnf() -> None:
    gbnf = compile_sql_ddl_to_gbnf(SQL_SAMPLE, output_format="insert")
    assert "INSERT INTO users" in gbnf
    assert "VALUES (" in gbnf
    assert "sql-string" in gbnf
    assert "sql-int" in gbnf
