"""Unit tests for GBNF linter and formatter."""

import pytest
from grammarforge.compiler.linter import GbnfLinter, format_gbnf


def test_linter_clean_grammar() -> None:
    gbnf = 'root ::= "hello" " " name\nname ::= [a-z]+'
    linter = GbnfLinter(gbnf)
    diags = linter.lint()
    errors = [d for d in diags if d.severity == "error"]
    assert len(errors) == 0


def test_linter_unbound_rule() -> None:
    gbnf = 'root ::= "item" " " missing-rule'
    linter = GbnfLinter(gbnf)
    diags = linter.lint()
    errors = [d for d in diags if d.severity == "error"]
    assert any("missing-rule" in e.message for e in errors)


def test_linter_dead_rule() -> None:
    gbnf = 'root ::= "active"\ndead-rule ::= "unused"'
    linter = GbnfLinter(gbnf)
    diags = linter.lint()
    warnings = [d for d in diags if d.severity == "warning"]
    assert any("dead-rule" in w.message for w in warnings)


def test_linter_direct_left_recursion() -> None:
    gbnf = 'root ::= root "a"'
    linter = GbnfLinter(gbnf)
    diags = linter.lint()
    errors = [d for d in diags if d.severity == "error"]
    assert any("left-recursion" in e.message for e in errors)


def test_formatter_alignment() -> None:
    unformatted = 'root   ::=   "A"|"B"\nlong-rule-name::= "C"'
    formatted = format_gbnf(unformatted)
    assert "root           ::= \"A\" | \"B\"" in formatted
    assert "long-rule-name ::= \"C\"" in formatted
