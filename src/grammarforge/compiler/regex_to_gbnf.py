"""Authentic regular expression parser and compiler to GBNF AST nodes."""

from __future__ import annotations

from typing import List, Optional, Tuple
from grammarforge.compiler.gbnf_ir import (
    Alternative,
    CharClass,
    Group,
    Node,
    Repetition,
    RepetitionKind,
    Sequence,
    Terminal,
)
from grammarforge.core.exceptions import RegexCompilationError


class RegexParser:
    """Recursive descent parser converting standard regex patterns into GBNF AST nodes."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0
        self.length = len(pattern)

    def _peek(self, offset: int = 0) -> Optional[str]:
        idx = self.pos + offset
        if idx < self.length:
            return self.pattern[idx]
        return None

    def _next(self) -> Optional[str]:
        if self.pos < self.length:
            ch = self.pattern[self.pos]
            self.pos += 1
            return ch
        return None

    def _consume(self, expected: str) -> bool:
        if self._peek() == expected:
            self.pos += 1
            return True
        return False

    def parse(self) -> Node:
        """Parse complete pattern into a GBNF Node."""
        # Strip leading ^ and trailing $ if present as GBNF applies globally to the rule
        if self.pattern.startswith("^"):
            self.pos += 1
        has_trailing_anchor = self.pattern.endswith("$") and not self.pattern.endswith(r"\$")

        node = self._parse_alternation()

        if has_trailing_anchor and self._peek() == "$":
            self.pos += 1

        if self.pos < self.length:
            # Check if remaining is just '$'
            if self.pos == self.length - 1 and self.pattern[self.pos] == "$":
                self.pos += 1
            else:
                raise RegexCompilationError(
                    f"Unexpected token at position {self.pos}: '{self.pattern[self.pos:]}' in pattern {self.pattern!r}",
                    pattern=self.pattern,
                )
        return node

    def _parse_alternation(self) -> Node:
        """Parse alternations: expr1 | expr2 | expr3."""
        branches: List[Node] = [self._parse_concatenation()]
        while self._consume("|"):
            branches.append(self._parse_concatenation())
        if len(branches) == 1:
            return branches[0]
        return Alternative(branches)

    def _parse_concatenation(self) -> Node:
        """Parse concatenated items: item1 item2 ..."""
        raw_items: List[Node] = []
        while self.pos < self.length:
            ch = self._peek()
            if ch is None or ch in ")|":
                break
            if ch == "$" and self.pos == self.length - 1:
                break
            raw_items.append(self._parse_quantified())

        # Coalesce adjacent Terminal nodes for clean and concise GBNF strings
        items: List[Node] = []
        for item in raw_items:
            if items and isinstance(items[-1], Terminal) and isinstance(item, Terminal):
                items[-1] = Terminal(items[-1].value + item.value)
            else:
                items.append(item)

        if not items:
            return Terminal("")
        if len(items) == 1:
            return items[0]
        return Sequence(items)

    def _parse_quantified(self) -> Node:
        """Parse an atom followed by optional quantifier (*, +, ?, {m,n})."""
        atom = self._parse_atom()

        ch = self._peek()
        if ch == "*":
            self.pos += 1
            return Repetition(atom, RepetitionKind.ZERO_OR_MORE)
        elif ch == "+":
            self.pos += 1
            return Repetition(atom, RepetitionKind.ONE_OR_MORE)
        elif ch == "?":
            self.pos += 1
            return Repetition(atom, RepetitionKind.OPTIONAL)
        elif ch == "{":
            return self._parse_counted_repetition(atom)

        return atom

    def _parse_counted_repetition(self, atom: Node) -> Node:
        """Parse {n}, {n,}, or {min,max} repetition."""
        self.pos += 1  # consume '{'
        buf = ""
        while self.pos < self.length and self._peek() != "}":
            buf += self._next()

        if not self._consume("}"):
            raise RegexCompilationError(
                f"Unterminated repetition quantifier '{{{buf}' in pattern {self.pattern!r}",
                pattern=self.pattern,
            )

        parts = buf.split(",")
        try:
            if len(parts) == 1:
                n = int(parts[0].strip())
                if n < 0:
                    raise ValueError
                return Sequence([atom for _ in range(n)])
            elif len(parts) == 2:
                min_n = int(parts[0].strip()) if parts[0].strip() else 0
                max_str = parts[1].strip()
                if not max_str:
                    # {min,} -> min instances followed by atom*
                    seq: List[Node] = [atom for _ in range(min_n)]
                    seq.append(Repetition(atom, RepetitionKind.ZERO_OR_MORE))
                    return Sequence(seq)
                else:
                    max_n = int(max_str)
                    if min_n > max_n:
                        raise ValueError(f"Min ({min_n}) > Max ({max_n})")
                    seq = [atom for _ in range(min_n)]
                    optional_count = max_n - min_n
                    # Add optional elements
                    opt_atom = Repetition(atom, RepetitionKind.OPTIONAL)
                    for _ in range(optional_count):
                        seq.append(opt_atom)
                    return Sequence(seq)
            else:
                raise ValueError("Too many commas in repetition quantifier")
        except ValueError as err:
            raise RegexCompilationError(
                f"Invalid quantifier range '{{{buf}}}': {err}",
                pattern=self.pattern,
            ) from err

    def _parse_atom(self) -> Node:
        """Parse atomic element: group, character class, escape sequence, or literal."""
        ch = self._peek()
        if ch is None:
            raise RegexCompilationError("Unexpected end of pattern", pattern=self.pattern)

        if ch == "(":
            return self._parse_group()
        elif ch == "[":
            return self._parse_char_class()
        elif ch == "\\":
            return self._parse_escape()
        elif ch == ".":
            self.pos += 1
            # Wildcard: any char except newline in standard regex
            return CharClass(inverted=True, single_chars=["\n", "\r"])
        else:
            self.pos += 1
            return Terminal(ch)

    def _parse_group(self) -> Node:
        """Parse group: (?:...) or (...)"""
        self.pos += 1  # consume '('
        # Check for non-capturing prefix ?:
        if self._peek() == "?" and self._peek(1) == ":":
            self.pos += 2

        inner = self._parse_alternation()
        if not self._consume(")"):
            raise RegexCompilationError("Unclosed group in regex", pattern=self.pattern)
        return Group(inner)

    def _parse_char_class(self) -> Node:
        """Parse character class: [a-z0-9] or [^abc]."""
        self.pos += 1  # consume '['
        inverted = False
        if self._peek() == "^":
            inverted = True
            self.pos += 1

        ranges: List[Tuple[str, str]] = []
        single_chars: List[str] = []

        first = True
        while self.pos < self.length:
            ch = self._peek()
            if ch == "]" and not first:
                self.pos += 1
                break
            first = False

            char_val = self._parse_char_class_char()

            # Check if this is a range: char_val '-' next_char
            if self._peek() == "-" and self._peek(1) is not None and self._peek(1) != "]":
                self.pos += 1  # consume '-'
                end_char = self._parse_char_class_char()
                ranges.append((char_val, end_char))
            else:
                single_chars.append(char_val)
        else:
            raise RegexCompilationError("Unclosed character class in regex", pattern=self.pattern)

        return CharClass(single_chars=single_chars, ranges=ranges, inverted=inverted)

    def _parse_char_class_char(self) -> str:
        """Parse a single character (possibly escaped) inside a character class."""
        ch = self._next()
        if ch is None:
            raise RegexCompilationError("Unexpected EOF in character class", pattern=self.pattern)
        if ch == "\\":
            esc = self._next()
            if esc is None:
                raise RegexCompilationError("Dangling escape in char class", pattern=self.pattern)
            if esc == "n":
                return "\n"
            if esc == "r":
                return "\r"
            if esc == "t":
                return "\t"
            if esc in "-]\\[^":
                return esc
            return esc
        return ch

    def _parse_escape(self) -> Node:
        """Parse an escape sequence outside a character class."""
        self.pos += 1  # consume '\'
        esc = self._next()
        if esc is None:
            raise RegexCompilationError("Dangling escape at end of pattern", pattern=self.pattern)

        # Predefined shorthand character classes
        if esc == "d":
            return CharClass(ranges=[("0", "9")])
        elif esc == "D":
            return CharClass(ranges=[("0", "9")], inverted=True)
        elif esc == "w":
            return CharClass(
                ranges=[("a", "z"), ("A", "Z"), ("0", "9")],
                single_chars=["_"],
            )
        elif esc == "W":
            return CharClass(
                ranges=[("a", "z"), ("A", "Z"), ("0", "9")],
                single_chars=["_"],
                inverted=True,
            )
        elif esc == "s":
            return CharClass(single_chars=[" ", "\t", "\n", "\r"])
        elif esc == "S":
            return CharClass(single_chars=[" ", "\t", "\n", "\r"], inverted=True)
        elif esc == "n":
            return Terminal("\n")
        elif esc == "r":
            return Terminal("\r")
        elif esc == "t":
            return Terminal("\t")
        elif esc in r"\/.^$*+?()[]{}|":
            return Terminal(esc)
        else:
            return Terminal(esc)


def compile_regex_to_gbnf_node(pattern: str) -> Node:
    """Compile a regex pattern into a GBNF AST Node."""
    parser = RegexParser(pattern)
    return parser.parse()
