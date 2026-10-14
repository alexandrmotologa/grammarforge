"""GBNF Grammar linter, diagnostic analyzer, and syntax formatter."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


class LintSeverity:
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LintDiagnostic:
    """A single diagnostic issue found during GBNF grammar linting."""

    def __init__(
        self,
        rule_name: str,
        message: str,
        severity: str = LintSeverity.ERROR,
        line_number: Optional[int] = None,
    ) -> None:
        self.rule_name = rule_name
        self.message = message
        self.severity = severity
        self.line_number = line_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule_name,
            "message": self.message,
            "severity": self.severity,
            "line": self.line_number,
        }

    def __repr__(self) -> str:
        loc = f":{self.line_number}" if self.line_number else ""
        return f"[{self.severity.upper()}]{loc} (rule '{self.rule_name}'): {self.message}"


class GbnfLinter:
    """Static analysis and linting engine for GBNF grammars."""

    RULE_DEF_RE = re.compile(r"^([a-zA-Z0-9_\-]+)\s*::=\s*(.*)$")
    TOKEN_REF_RE = re.compile(r'\b([a-zA-Z0-9_\-]+)\b')

    def __init__(self, gbnf_text: str) -> None:
        self.text = gbnf_text
        self.rules: Dict[str, str] = {}
        self.rule_lines: Dict[str, int] = {}
        self.diagnostics: List[LintDiagnostic] = []
        self._parse_grammar()

    def _parse_grammar(self) -> None:
        """Parse raw GBNF text into rule definitions."""
        lines = self.text.splitlines()
        curr_rule: Optional[str] = None
        curr_body_lines: List[str] = []
        curr_start_line = 1

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Ignore standalone comment lines
            if stripped.startswith("#"):
                continue

            match = self.RULE_DEF_RE.match(line)
            if match:
                if curr_rule:
                    self.rules[curr_rule] = " ".join(curr_body_lines).strip()
                    self.rule_lines[curr_rule] = curr_start_line
                curr_rule = match.group(1)
                curr_body_lines = [match.group(2)]
                curr_start_line = idx
            elif curr_rule and line.startswith((" ", "\t")):
                # Continuation line
                curr_body_lines.append(line.strip())
            elif stripped:
                # Malformed line outside rule
                self.diagnostics.append(
                    LintDiagnostic(
                        rule_name=curr_rule or "<global>",
                        message=f"Syntax error: unexpected unindented content '{stripped[:40]}'",
                        severity=LintSeverity.ERROR,
                        line_number=idx,
                    )
                )

        if curr_rule:
            self.rules[curr_rule] = " ".join(curr_body_lines).strip()
            self.rule_lines[curr_rule] = curr_start_line

    def extract_references(self, body: str) -> Set[str]:
        """Extract all rule identifiers referenced in an expression body."""
        # Strip literal strings "..."
        stripped = re.sub(r'"(\\.|[^"\\])*"', '""', body)
        # Strip character classes [...]
        stripped = re.sub(r'\[(\\.|[^\]\\])*\]', '[]', stripped)
        # Find potential word tokens
        words = self.TOKEN_REF_RE.findall(stripped)
        return {w for w in words if w in self.rules or w not in ("", "ws")}

    def lint(self) -> List[LintDiagnostic]:
        """Execute full suite of static analysis checks."""
        self.diagnostics.clear()

        if not self.rules:
            self.diagnostics.append(
                LintDiagnostic(
                    rule_name="<global>",
                    message="Grammar contains no valid rule definitions (expected 'rule ::= expr')",
                    severity=LintSeverity.ERROR,
                )
            )
            return self.diagnostics

        # Check 1: Root rule presence
        if "root" not in self.rules:
            first_rule = next(iter(self.rules.keys()))
            self.diagnostics.append(
                LintDiagnostic(
                    rule_name="<global>",
                    message=f"Missing explicit 'root' rule. Engine will default to first rule '{first_rule}'.",
                    severity=LintSeverity.WARNING,
                )
            )

        # Check 2: Unbound rule references
        rule_deps: Dict[str, Set[str]] = {}
        for r_name, r_body in self.rules.items():
            line_no = self.rule_lines.get(r_name, 1)
            if not r_body:
                self.diagnostics.append(
                    LintDiagnostic(
                        rule_name=r_name,
                        message="Rule body is empty.",
                        severity=LintSeverity.ERROR,
                        line_number=line_no,
                    )
                )
                rule_deps[r_name] = set()
                continue

            # Check quotes & brackets balance
            if r_body.count('"') % 2 != 0:
                self.diagnostics.append(
                    LintDiagnostic(
                        rule_name=r_name,
                        message="Unterminated string literal in rule body.",
                        severity=LintSeverity.ERROR,
                        line_number=line_no,
                    )
                )

            refs = self.extract_references(r_body)
            rule_deps[r_name] = refs

            for ref in refs:
                if ref not in self.rules:
                    self.diagnostics.append(
                        LintDiagnostic(
                            rule_name=r_name,
                            message=f"Referenced rule '{ref}' is not defined in the grammar.",
                            severity=LintSeverity.ERROR,
                            line_number=line_no,
                        )
                    )

        # Check 3: Dead / Unreachable rules
        entry_point = "root" if "root" in self.rules else next(iter(self.rules.keys()))
        reachable: Set[str] = set()

        def dfs(name: str) -> None:
            if name in reachable or name not in self.rules:
                return
            reachable.add(name)
            for dep in rule_deps.get(name, set()):
                dfs(dep)

        dfs(entry_point)

        for r_name in self.rules:
            if r_name not in reachable:
                self.diagnostics.append(
                    LintDiagnostic(
                        rule_name=r_name,
                        message=f"Dead rule: '{r_name}' is unreachable from '{entry_point}'.",
                        severity=LintSeverity.WARNING,
                        line_number=self.rule_lines.get(r_name),
                    )
                )

        # Check 4: Direct Left-Recursion
        for r_name, r_body in self.rules.items():
            first_token_match = re.match(r"^([a-zA-Z0-9_\-]+)", r_body.strip())
            if first_token_match and first_token_match.group(1) == r_name:
                self.diagnostics.append(
                    LintDiagnostic(
                        rule_name=r_name,
                        message=f"Direct left-recursion detected in rule '{r_name}'. May cause infinite loops in LLM parsers.",
                        severity=LintSeverity.ERROR,
                        line_number=self.rule_lines.get(r_name),
                    )
                )

        return self.diagnostics


def format_gbnf(gbnf_text: str) -> str:
    """Format and re-indent GBNF grammar text with aligned ::= operators."""
    lines = gbnf_text.splitlines()
    rules: List[Tuple[str, str, Optional[str]]] = []
    curr_name: Optional[str] = None
    curr_body_parts: List[str] = []
    curr_comment: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            curr_comment = stripped[1:].strip()
            continue

        match = re.match(r"^([a-zA-Z0-9_\-]+)\s*::=\s*(.*)$", line)
        if match:
            if curr_name:
                rules.append((curr_name, " ".join(curr_body_parts).strip(), curr_comment))
                curr_comment = None
            curr_name = match.group(1)
            curr_body_parts = [match.group(2)]
        elif curr_name and line.startswith((" ", "\t")):
            curr_body_parts.append(line.strip())

    if curr_name:
        rules.append((curr_name, " ".join(curr_body_parts).strip(), curr_comment))

    if not rules:
        return gbnf_text.strip() + "\n"

    # Find longest rule name for clean alignment
    max_name_len = max(len(r[0]) for r in rules)
    formatted_blocks: List[str] = []

    for name, body, comment in rules:
        block = ""
        if comment:
            block += f"# {comment}\n"
        padding = " " * (max_name_len - len(name))
        # Normalize internal whitespace
        norm_body = re.sub(r"\s+", " ", body)
        norm_body = re.sub(r"\s*\|\s*", " | ", norm_body)
        block += f"{name}{padding} ::= {norm_body}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks) + "\n"
