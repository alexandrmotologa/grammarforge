"""Core configuration and settings models for GrammarForge."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class WhitespaceMode(str, Enum):
    """Whitespace handling strategy for generated GBNF grammars."""

    LENIENT = "lenient"  # Standard GBNF ws ::= [ \t\n\r]* between tokens
    COMPACT = "compact"  # Minified JSON without redundant whitespace
    PRETTY = "pretty"    # 2-space indented JSON layout


class KeyOrdering(str, Enum):
    """Ordering constraint for object properties."""

    CANONICAL = "canonical"  # Strict schema order for fastest decoding
    FLEXIBLE = "flexible"    # Allows keys in any order via permutations


class CompilerConfig(BaseModel):
    """Configuration options for the JSON Schema to GBNF compiler."""

    whitespace_mode: WhitespaceMode = Field(
        default=WhitespaceMode.LENIENT,
        description="Whitespace handling strategy for compiled grammars.",
    )
    key_ordering: KeyOrdering = Field(
        default=KeyOrdering.CANONICAL,
        description="Property key ordering in JSON objects.",
    )
    max_recursion_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum recursion depth for recursive $ref schemas.",
    )
    root_rule_name: str = Field(
        default="root",
        description="Name of the root non-terminal GBNF rule.",
    )
    rule_prefix: str = Field(
        default="",
        description="Optional prefix prepended to generated subrules.",
    )
    allow_empty_objects: bool = Field(
        default=True,
        description="Whether empty JSON objects {} are permitted when properties are optional.",
    )
    allow_empty_arrays: bool = Field(
        default=True,
        description="Whether empty arrays [] are permitted when minItems is 0.",
    )
    strict_numbers: bool = Field(
        default=True,
        description="Disallow leading zeros in integers according to RFC 8259.",
    )
    compact_literals: bool = Field(
        default=False,
        description="Inline small literal sets directly into parent rules.",
    )


class EngineConfig(BaseModel):
    """Configuration for the token masking and constrained decoding engine."""

    vocab_size: int = Field(
        default=32000,
        ge=256,
        description="Size of the tokenizer vocabulary.",
    )
    eos_token_id: int = Field(
        default=2,
        ge=0,
        description="Token ID of the End-Of-Sequence token.",
    )
    bos_token_id: Optional[int] = Field(
        default=1,
        description="Token ID of the Beginning-Of-Sequence token, if present.",
    )
    neg_inf_value: float = Field(
        default=-1e9,
        description="Value used for masked logits.",
    )
    cache_trie_traversals: bool = Field(
        default=True,
        description="Cache sub-trie lookups across generation steps.",
    )
    max_prefix_cache_size: int = Field(
        default=1024,
        ge=64,
        description="Maximum number of prefix state transitions to keep in memory.",
    )
