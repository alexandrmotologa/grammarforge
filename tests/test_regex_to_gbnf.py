"""Unit tests for Regex to GBNF compiler."""

import pytest
from grammarforge.compiler.regex_to_gbnf import RegexParser, compile_regex_to_gbnf_node


def test_regex_literals_and_alternation() -> None:
    node = compile_regex_to_gbnf_node("cat|dog|fish")
    assert node.to_gbnf() == '"cat" | "dog" | "fish"'


def test_regex_quantifiers() -> None:
    node1 = compile_regex_to_gbnf_node("a+")
    assert node1.to_gbnf() == '"a"+'

    node2 = compile_regex_to_gbnf_node("b*")
    assert node2.to_gbnf() == '"b"*'

    node3 = compile_regex_to_gbnf_node("c?")
    assert node3.to_gbnf() == '"c"?'


def test_regex_exact_counted_repetition() -> None:
    # {3} -> 3 sequence items
    node = compile_regex_to_gbnf_node("[0-9]{3}")
    assert node.to_gbnf() == "[0-9] [0-9] [0-9]"


def test_regex_min_max_counted_repetition() -> None:
    # {1,3} -> 1 base + 2 optional
    node = compile_regex_to_gbnf_node("[a-z]{1,3}")
    gbnf = node.to_gbnf()
    assert "[a-z]" in gbnf
    assert "[a-z]?" in gbnf


def test_regex_character_classes_and_escapes() -> None:
    # \d+ -> [0-9]+
    node_d = compile_regex_to_gbnf_node(r"\d+")
    assert node_d.to_gbnf() == "[0-9]+"

    # \w+
    node_w = compile_regex_to_gbnf_node(r"\w+")
    assert "0-9A-Za-z" in node_w.to_gbnf() or "a-zA-Z0-9" in node_w.to_gbnf() or "_" in node_w.to_gbnf()

    # Custom class
    node_custom = compile_regex_to_gbnf_node(r"[A-Z0-9_]+")
    assert "[0-9A-Z_]+" in node_custom.to_gbnf()


def test_regex_groups_and_anchors() -> None:
    node = compile_regex_to_gbnf_node(r"^(alpha|beta)_[0-9]+$")
    gbnf = node.to_gbnf()
    assert '("alpha" | "beta")' in gbnf
    assert '"_"' in gbnf
    assert "[0-9]+" in gbnf
