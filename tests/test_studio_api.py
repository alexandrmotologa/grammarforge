"""Unit tests for FastAPI Web Studio endpoints."""

import pytest
from fastapi.testclient import TestClient
from grammarforge.studio.server import app

client = TestClient(app)


def test_api_examples() -> None:
    res = client.get("/api/examples")
    assert res.status_code == 200
    data = res.json()
    assert "tool_call" in data
    assert "user_profile" in data


def test_api_compile_schema() -> None:
    req = {
        "source_type": "schema",
        "schema_data": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
            },
            "required": ["task"],
        },
    }
    res = client.post("/api/compile", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "gbnf" in data
    assert data["rule_count"] > 0


def test_api_compile_regex() -> None:
    req = {
        "source_type": "regex",
        "regex_pattern": "^[a-z0-9_]+$",
    }
    res = client.post("/api/compile", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "root" in data["gbnf"]


def test_api_validate() -> None:
    req = {
        "schema_data": {"type": "number"},
        "payload": "3.1415",
    }
    res = client.post("/api/validate", json=req)
    assert res.status_code == 200
    assert res.json()["is_valid"] is True


def test_api_mask() -> None:
    req = {
        "schema_data": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "prefix": '{"name": "',
        "vocab_size": 1000,
    }
    res = client.post("/api/mask", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["allowed_count"] > 0
    assert data["latency_us"] > 0.0


def test_serve_index() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "GrammarForge" in res.text
