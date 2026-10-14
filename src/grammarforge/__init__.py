"""GrammarForge: High-throughput constrained decoding engine & GBNF compiler for local LLM gateways."""

from grammarforge.compiler.gbnf_ir import (
    Alternative,
    CharClass,
    Grammar,
    Group,
    Node,
    Repetition,
    RepetitionKind,
    Rule,
    RuleRef,
    Sequence,
    Terminal,
)
from grammarforge.compiler.json_schema import SchemaCompiler, compile_json_schema
from grammarforge.compiler.linter import GbnfLinter, LintDiagnostic, format_gbnf
from grammarforge.compiler.pydantic_bridge import (
    compile_pydantic_model,
    compile_pydantic_source,
    get_schema_from_type,
    load_model_from_spec,
)
from grammarforge.compiler.regex_to_gbnf import RegexParser, compile_regex_to_gbnf_node
from grammarforge.compiler.sql_ddl import ColumnDef, TableDef, compile_sql_ddl_to_gbnf, parse_create_table_ddl
from grammarforge.core.config import (
    CompilerConfig,
    EngineConfig,
    KeyOrdering,
    WhitespaceMode,
)
from grammarforge.core.exceptions import (
    GrammarExhaustedError,
    GrammarForgeError,
    InvalidGrammarError,
    MaskingError,
    RegexCompilationError,
    SchemaCompilationError,
)
from grammarforge.engine.gateway import (
    GrammarForgeLogitsProcessor,
    LlamaCppGrammarConnector,
    StreamingValidator,
)
from grammarforge.engine.masking import LogitMasker, MaskingResult
from grammarforge.engine.precomputed import PrecomputedMasks
from grammarforge.engine.repair import RepairResult, heal_partial_json
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import TrieNode, VocabularyTrie
from grammarforge.gateway.proxy import create_gateway_proxy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core
    "CompilerConfig",
    "EngineConfig",
    "KeyOrdering",
    "WhitespaceMode",
    "GrammarForgeError",
    "SchemaCompilationError",
    "RegexCompilationError",
    "InvalidGrammarError",
    "MaskingError",
    "GrammarExhaustedError",
    # Compiler
    "Grammar",
    "Rule",
    "RuleRef",
    "Node",
    "Terminal",
    "CharClass",
    "Sequence",
    "Alternative",
    "Repetition",
    "RepetitionKind",
    "Group",
    "SchemaCompiler",
    "compile_json_schema",
    "RegexParser",
    "compile_regex_to_gbnf_node",
    "compile_pydantic_model",
    "compile_pydantic_source",
    "get_schema_from_type",
    "load_model_from_spec",
    "GbnfLinter",
    "LintDiagnostic",
    "format_gbnf",
    "compile_sql_ddl_to_gbnf",
    "parse_create_table_ddl",
    "ColumnDef",
    "TableDef",
    # Engine
    "TrieNode",
    "VocabularyTrie",
    "JsonStateMachine",
    "ParserStatus",
    "LogitMasker",
    "MaskingResult",
    "PrecomputedMasks",
    "heal_partial_json",
    "RepairResult",
    "GrammarForgeLogitsProcessor",
    "StreamingValidator",
    "LlamaCppGrammarConnector",
    # Gateway
    "create_gateway_proxy",
]
