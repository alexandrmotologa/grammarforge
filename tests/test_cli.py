"""Unit tests for GrammarForge CLI commands."""

import json
import pytest
from typer.testing import CliRunner
from grammarforge.cli.main import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "GrammarForge" in result.stdout


def test_cli_compile_schema(tmp_path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "required": ["name"],
    }
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema), encoding="utf-8")

    out_file = tmp_path / "out.gbnf"

    result = runner.invoke(app, ["compile", str(schema_file), "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "root" in content


def test_cli_regex() -> None:
    result = runner.invoke(app, ["regex", "^[0-9]{3,5}$"])
    assert result.exit_code == 0
    assert "[0-9]" in result.stdout


def test_cli_validate(tmp_path) -> None:
    schema = {"type": "object", "properties": {"active": {"type": "boolean"}}}
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(schema_file), '{"active": true}'])
    assert result.exit_code == 0
    assert "SUCCESS" in result.stdout


def test_cli_bench() -> None:
    result = runner.invoke(app, ["bench", "--vocab-size", "500", "--iterations", "10"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout


def test_cli_mask(tmp_path) -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema), encoding="utf-8")

    result = runner.invoke(app, ["mask", str(schema_file), "--prefix", '{"', "--vocab-size", "500"])
    assert result.exit_code == 0
    assert "Logit Masking Step" in result.stdout


def test_cli_pydantic(tmp_path) -> None:
    model_file = tmp_path / "test_model.py"
    model_file.write_text(
        "from pydantic import BaseModel\nclass CliTestModel(BaseModel):\n    num: int\n",
        encoding="utf-8"
    )

    out_file = tmp_path / "model.gbnf"
    result = runner.invoke(app, ["pydantic", f"{model_file}:CliTestModel", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    assert "root" in out_file.read_text(encoding="utf-8")

