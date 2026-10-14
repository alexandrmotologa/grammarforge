"""Custom exception hierarchy for GrammarForge."""

from typing import Optional


class GrammarForgeError(Exception):
    """Base exception for all GrammarForge errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SchemaCompilationError(GrammarForgeError):
    """Raised when a JSON Schema cannot be compiled into GBNF."""

    def __init__(self, message: str, schema_path: Optional[str] = None, details: Optional[dict] = None) -> None:
        super().__init__(message, details)
        self.schema_path = schema_path


class RegexCompilationError(GrammarForgeError):
    """Raised when a regular expression pattern cannot be parsed or converted into GBNF."""

    def __init__(self, message: str, pattern: Optional[str] = None, details: Optional[dict] = None) -> None:
        super().__init__(message, details)
        self.pattern = pattern


class InvalidGrammarError(GrammarForgeError):
    """Raised when GBNF grammar contains syntax or rule resolution errors."""
    pass


class MaskingError(GrammarForgeError):
    """Raised when token logit masking encounters an unrecoverable state."""
    pass


class GrammarExhaustedError(MaskingError):
    """Raised when the grammar reaches a terminal state and no more tokens are allowed."""
    pass
