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
from grammarforge.compiler.pydantic_bridge import (
    compile_pydantic_model,
    compile_pydantic_source,
    get_schema_from_type,
    load_model_from_spec,
)
from grammarforge.compiler.regex_to_gbnf import RegexParser, compile_regex_to_gbnf_node
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
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import TrieNode, VocabularyTrie

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
    # Engine
    "TrieNode",
    "VocabularyTrie",
    "JsonStateMachine",
    "ParserStatus",
    "LogitMasker",
    "MaskingResult",
    "GrammarForgeLogitsProcessor",
    "StreamingValidator",
    "LlamaCppGrammarConnector",
]
