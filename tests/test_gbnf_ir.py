"""Unit tests for GBNF Intermediate Representation AST and serializer."""

import pytest
from grammarforge.compiler.gbnf_ir import (
    Alternative,
    CharClass,
    Grammar,
    Group,
    Repetition,
    RepetitionKind,
    Rule,
    RuleRef,
    Sequence,
    Terminal,
    sanitize_rule_name,
)


def test_sanitize_rule_name() -> None:
    assert sanitize_rule_name("my_rule") == "my_rule"
    assert sanitize_rule_name("My Custom Rule 1") == "my-custom-rule-1"
    assert sanitize_rule_name("123abc") == "r-123abc"
    assert sanitize_rule_name("!@#$%^") == "rule"


def test_terminal_serialization() -> None:
    t = Terminal("hello world")
    assert t.to_gbnf() == '"hello world"'

    t_escaped = Terminal('hello "quoted" \n \t')
    assert t_escaped.to_gbnf() == '"hello \\"quoted\\" \\n \\t"'


def test_char_class_serialization() -> None:
    cc = CharClass(ranges=[("a", "z"), ("0", "9")], single_chars=["_"])
    assert cc.to_gbnf() == "[0-9a-z_]"

    cc_inv = CharClass(single_chars=['"', "\\"], inverted=True)
    assert cc_inv.to_gbnf() == '[^\\"\\\\]'


def test_sequence_and_alternative() -> None:
    seq = Sequence([Terminal("A"), RuleRef("b"), Terminal("C")])
    assert seq.to_gbnf() == '"A" b "C"'

    alt = Alternative([Terminal("yes"), Terminal("no")])
    assert alt.to_gbnf() == '"yes" | "no"'


def test_repetitions() -> None:
    rep_star = Repetition(Terminal("x"), RepetitionKind.ZERO_OR_MORE)
    assert rep_star.to_gbnf() == '"x"*'

    rep_plus = Repetition(Terminal("y"), RepetitionKind.ONE_OR_MORE)
    assert rep_plus.to_gbnf() == '"y"+'

    rep_opt = Repetition(Terminal("z"), RepetitionKind.OPTIONAL)
    assert rep_opt.to_gbnf() == '"z"?'


def test_grammar_serialization_and_deduplication() -> None:
    g = Grammar()
    r1 = g.create_rule("sub", Terminal("constant"))
    r2 = g.create_rule("sub_duplicate", Terminal("constant"))

    # Should deduplicate to the same rule name
    assert r1.name == r2.name

    root = Rule("root", Sequence([r1, Terminal("!")]))
    g.add_rule(root)

    serialized = g.serialize(root_name="root")
    assert "root ::= sub \"!\"" in serialized
    assert "sub ::= \"constant\"" in serialized
