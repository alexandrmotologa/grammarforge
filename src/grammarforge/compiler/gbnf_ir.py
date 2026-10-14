"""GBNF (GGML BNF) Intermediate Representation, AST nodes, and serializer."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


class RepetitionKind(str, Enum):
    """Repetition operator in GBNF."""

    ZERO_OR_MORE = "*"
    ONE_OR_MORE = "+"
    OPTIONAL = "?"


def sanitize_rule_name(name: str) -> str:
    """Sanitize arbitrary strings into valid GBNF non-terminal identifiers [a-zA-Z0-9_-]."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = "rule"
    if cleaned[0].isdigit():
        cleaned = f"r-{cleaned}"
    return cleaned


def escape_terminal_char(ch: str) -> str:
    """Escape a character inside a literal string."""
    if ch == '"':
        return '\\"'
    if ch == "\\":
        return "\\\\"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    code = ord(ch)
    if code < 32 or code > 126:
        return f"\\x{code:02x}" if code < 256 else f"\\u{code:04x}"
    return ch


def escape_char_class_member(ch: str) -> str:
    """Escape a character inside a character class [...]."""
    if ch in ("]", "\\", "-", "^", '"'):
        return f"\\{ch}"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    code = ord(ch)
    if code < 32 or code > 126:
        return f"\\x{code:02x}" if code < 256 else f"\\u{code:04x}"
    return ch


class Node(ABC):
    """Base class for GBNF AST nodes."""

    @abstractmethod
    def to_gbnf(self, precedence: int = 0) -> str:
        """Serialize node to GBNF string. Precedence helps minimize unnecessary parentheses."""
        pass

    @abstractmethod
    def references(self) -> Set[str]:
        """Return rule names referenced by this node."""
        pass


class Terminal(Node):
    """A literal string terminal: "abc"."""

    def __init__(self, value: str) -> None:
        self.value = value

    def to_gbnf(self, precedence: int = 0) -> str:
        escaped = "".join(escape_terminal_char(c) for c in self.value)
        return f'"{escaped}"'

    def references(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        return f"Terminal({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Terminal) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("Terminal", self.value))


class CharClass(Node):
    """A character class terminal: [a-z0-9] or [^a-z]."""

    def __init__(
        self,
        single_chars: Optional[Iterable[str]] = None,
        ranges: Optional[Iterable[Tuple[str, str]]] = None,
        inverted: bool = False,
    ) -> None:
        self.single_chars = sorted(list(set(single_chars or [])))
        self.ranges = sorted(list(set(ranges or [])))
        self.inverted = inverted

    def to_gbnf(self, precedence: int = 0) -> str:
        parts: List[str] = []
        if self.inverted:
            parts.append("^")
        for start, end in self.ranges:
            parts.append(f"{escape_char_class_member(start)}-{escape_char_class_member(end)}")
        for ch in self.single_chars:
            parts.append(escape_char_class_member(ch))
        return f"[{''.join(parts)}]"

    def references(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        inv = "^" if self.inverted else ""
        return f"CharClass({inv}{self.ranges!r}, {self.single_chars!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CharClass)
            and self.single_chars == other.single_chars
            and self.ranges == other.ranges
            and self.inverted == other.inverted
        )

    def __hash__(self) -> int:
        return hash(("CharClass", tuple(self.single_chars), tuple(self.ranges), self.inverted))


class RuleRef(Node):
    """Reference to another rule non-terminal."""

    def __init__(self, name: str) -> None:
        self.name = sanitize_rule_name(name)

    def to_gbnf(self, precedence: int = 0) -> str:
        return self.name

    def references(self) -> Set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return f"RuleRef({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RuleRef) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("RuleRef", self.name))


class Group(Node):
    """Grouped expression: ( expr )."""

    def __init__(self, inner: Node) -> None:
        self.inner = inner

    def to_gbnf(self, precedence: int = 0) -> str:
        return f"({self.inner.to_gbnf(0)})"

    def references(self) -> Set[str]:
        return self.inner.references()

    def __repr__(self) -> str:
        return f"Group({self.inner!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Group) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("Group", self.inner))


class Sequence(Node):
    """Sequence of expressions: A B C."""

    # Precedence level: Alternation (1) < Sequence (2) < Repetition (3) < Atom (4)
    def __init__(self, elements: List[Node]) -> None:
        # Flatten nested sequences
        flattened: List[Node] = []
        for el in elements:
            if isinstance(el, Sequence):
                flattened.extend(el.elements)
            else:
                flattened.append(el)
        self.elements = flattened

    def to_gbnf(self, precedence: int = 0) -> str:
        if not self.elements:
            return '""'
        if len(self.elements) == 1:
            return self.elements[0].to_gbnf(precedence)
        rendered = " ".join(e.to_gbnf(2) for e in self.elements)
        if precedence > 2:
            return f"({rendered})"
        return rendered

    def references(self) -> Set[str]:
        refs: Set[str] = set()
        for el in self.elements:
            refs.update(el.references())
        return refs

    def __repr__(self) -> str:
        return f"Sequence({self.elements!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sequence) and self.elements == other.elements

    def __hash__(self) -> int:
        return hash(("Sequence", tuple(self.elements)))


class Alternative(Node):
    """Alternation of expressions: A | B | C."""

    def __init__(self, elements: List[Node]) -> None:
        # Flatten nested alternatives and deduplicate while keeping order
        flattened: List[Node] = []
        seen: Set[str] = set()
        for el in elements:
            if isinstance(el, Alternative):
                sub = el.elements
            else:
                sub = [el]
            for s in sub:
                gbnf_rep = s.to_gbnf()
                if gbnf_rep not in seen:
                    seen.add(gbnf_rep)
                    flattened.append(s)
        self.elements = flattened

    def to_gbnf(self, precedence: int = 0) -> str:
        if not self.elements:
            return '""'
        if len(self.elements) == 1:
            return self.elements[0].to_gbnf(precedence)
        rendered = " | ".join(e.to_gbnf(1) for e in self.elements)
        if precedence > 1:
            return f"({rendered})"
        return rendered

    def references(self) -> Set[str]:
        refs: Set[str] = set()
        for el in self.elements:
            refs.update(el.references())
        return refs

    def __repr__(self) -> str:
        return f"Alternative({self.elements!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Alternative) and self.elements == other.elements

    def __hash__(self) -> int:
        return hash(("Alternative", tuple(self.elements)))


class Repetition(Node):
    """Repetition of an expression: expr*, expr+, expr?."""

    def __init__(self, inner: Node, kind: RepetitionKind) -> None:
        self.inner = inner
        self.kind = kind

    def to_gbnf(self, precedence: int = 0) -> str:
        inner_str = self.inner.to_gbnf(3)
        return f"{inner_str}{self.kind.value}"

    def references(self) -> Set[str]:
        return self.inner.references()

    def __repr__(self) -> str:
        return f"Repetition({self.inner!r}, {self.kind.value})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Repetition)
            and self.inner == other.inner
            and self.kind == other.kind
        )

    def __hash__(self) -> int:
        return hash(("Repetition", self.inner, self.kind))


class Rule:
    """A named GBNF production rule: name ::= body."""

    def __init__(self, name: str, body: Node, comment: Optional[str] = None) -> None:
        self.name = sanitize_rule_name(name)
        self.body = body
        self.comment = comment

    def to_gbnf(self) -> str:
        out = ""
        if self.comment:
            out += f"# {self.comment}\n"
        out += f"{self.name} ::= {self.body.to_gbnf(0)}"
        return out

    def __repr__(self) -> str:
        return f"Rule({self.name} ::= {self.body!r})"


class Grammar:
    """Container for GBNF rules with dependency ordering and deduplication."""

    def __init__(self, root_rule: Optional[Rule] = None) -> None:
        self.rules: Dict[str, Rule] = {}
        self.body_to_name: Dict[str, str] = {}
        self._counter: int = 1
        if root_rule:
            self.add_rule(root_rule)

    def add_rule(self, rule: Rule) -> Rule:
        """Add a rule to the grammar."""
        self.rules[rule.name] = rule
        body_key = rule.body.to_gbnf(0)
        self.body_to_name[body_key] = rule.name
        return rule

    def create_rule(self, name_hint: str, body: Node, comment: Optional[str] = None) -> RuleRef:
        """Create a rule, deduplicating equivalent bodies if an exact match exists."""
        body_key = body.to_gbnf(0)
        if body_key in self.body_to_name:
            return RuleRef(self.body_to_name[body_key])

        base_name = sanitize_rule_name(name_hint)
        candidate = base_name
        while candidate in self.rules:
            self._counter += 1
            candidate = f"{base_name}-{self._counter}"

        rule = Rule(candidate, body, comment)
        self.add_rule(rule)
        return RuleRef(candidate)

    def has_rule(self, name: str) -> bool:
        return sanitize_rule_name(name) in self.rules

    def get_rule(self, name: str) -> Optional[Rule]:
        return self.rules.get(sanitize_rule_name(name))

    def serialize(self, root_name: str = "root") -> str:
        """Serialize grammar into a deterministic, valid GBNF string."""
        root_key = sanitize_rule_name(root_name)
        if root_key not in self.rules:
            raise KeyError(f"Root rule '{root_name}' not found in grammar.")

        # Traverse reachable rules starting from root
        reachable_order: List[str] = []
        visited: Set[str] = set()

        def dfs(name: str) -> None:
            if name in visited or name not in self.rules:
                return
            visited.add(name)
            reachable_order.append(name)
            for ref in sorted(self.rules[name].body.references()):
                dfs(ref)

        dfs(root_key)

        # Append any remaining disconnected rules in sorted order
        for name in sorted(self.rules.keys()):
            if name not in visited:
                reachable_order.append(name)

        return "\n\n".join(self.rules[k].to_gbnf() for k in reachable_order) + "\n"
