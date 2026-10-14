"""Unit tests for Pydantic and type introspection bridge."""

from typing import List, Optional
import pytest
from pydantic import BaseModel, Field
from grammarforge.compiler.pydantic_bridge import (
    compile_pydantic_model,
    compile_pydantic_source,
    get_schema_from_type,
)


class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(ge=0, le=150)
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)


def test_pydantic_model_compilation() -> None:
    schema = get_schema_from_type(UserProfile)
    assert schema["type"] == "object"
    assert "username" in schema["properties"]
    assert "age" in schema["properties"]

    gbnf = compile_pydantic_model(UserProfile)
    assert "root" in gbnf
    assert '"username"' in gbnf or '"\\\"username\\\""' in gbnf


def test_pydantic_source_code_compilation() -> None:
    source = """
from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    limit: int
"""
    gbnf = compile_pydantic_source(source, "SearchQuery")
    assert "root" in gbnf
    assert '"query"' in gbnf or '"\\\"query\\\""' in gbnf
