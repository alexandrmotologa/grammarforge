"""FastAPI API route handlers for GrammarForge Web Studio."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from grammarforge.compiler.gbnf_ir import Grammar, Rule
from grammarforge.compiler.json_schema import SchemaCompiler, compile_json_schema
from grammarforge.compiler.linter import GbnfLinter, format_gbnf
from grammarforge.compiler.pydantic_bridge import compile_pydantic_source
from grammarforge.compiler.regex_to_gbnf import compile_regex_to_gbnf_node
from grammarforge.compiler.sql_ddl import compile_sql_ddl_to_gbnf
from grammarforge.core.config import CompilerConfig, KeyOrdering, WhitespaceMode
from grammarforge.engine.masking import LogitMasker
from grammarforge.engine.repair import heal_partial_json
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie

router = APIRouter(prefix="/api")

# Shared synthetic vocab cache for fast API demo requests
_GLOBAL_VOCAB: Optional[VocabularyTrie] = None


def get_shared_vocab(size: int = 32000) -> VocabularyTrie:
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is None or _GLOBAL_VOCAB.vocab_size < size:
        _GLOBAL_VOCAB = VocabularyTrie.create_synthetic_vocab(size)
    return _GLOBAL_VOCAB


class CompileRequest(BaseModel):
    source_type: str = Field(default="schema", description="'schema', 'regex', or 'pydantic'")
    schema_data: Optional[Dict[str, Any]] = None
    regex_pattern: Optional[str] = None
    pydantic_source: Optional[str] = None
    pydantic_class: Optional[str] = "Model"
    whitespace_mode: WhitespaceMode = WhitespaceMode.LENIENT
    key_ordering: KeyOrdering = KeyOrdering.CANONICAL


class CompileResponse(BaseModel):
    gbnf: str
    rule_count: int
    rule_names: List[str]
    root_rule: str


class ValidateRequest(BaseModel):
    schema_data: Dict[str, Any]
    payload: str


class ValidateResponse(BaseModel):
    status: str
    is_valid: bool
    error: Optional[str] = None


class MaskRequest(BaseModel):
    schema_data: Dict[str, Any]
    prefix: str
    vocab_size: int = 32000


class MaskResponse(BaseModel):
    allowed_count: int
    total_vocab: int
    is_accepting: bool
    latency_us: float
    sample_allowed_tokens: List[Dict[str, Any]]
    status: str


@router.post("/compile", response_model=CompileResponse)
async def api_compile(req: CompileRequest) -> CompileResponse:
    """Compile input schema, regex, or Pydantic source to GBNF."""
    try:
        config = CompilerConfig(
            whitespace_mode=req.whitespace_mode,
            key_ordering=req.key_ordering,
        )

        if req.source_type == "schema":
            if not req.schema_data:
                raise HTTPException(status_code=400, detail="Missing 'schema_data'.")
            compiler = SchemaCompiler(req.schema_data, config=config)
            grammar = compiler.compile()
            gbnf = grammar.serialize()
            return CompileResponse(
                gbnf=gbnf,
                rule_count=len(grammar.rules),
                rule_names=list(grammar.rules.keys()),
                root_rule="root",
            )

        elif req.source_type == "regex":
            if not req.regex_pattern:
                raise HTTPException(status_code=400, detail="Missing 'regex_pattern'.")
            node = compile_regex_to_gbnf_node(req.regex_pattern)
            grammar = Grammar()
            grammar.add_rule(Rule("root", node, comment=f"regex: {req.regex_pattern}"))
            gbnf = grammar.serialize()
            return CompileResponse(
                gbnf=gbnf,
                rule_count=1,
                rule_names=["root"],
                root_rule="root",
            )

        elif req.source_type == "pydantic":
            if not req.pydantic_source:
                raise HTTPException(status_code=400, detail="Missing 'pydantic_source'.")
            gbnf = compile_pydantic_source(req.pydantic_source, req.pydantic_class or "Model", config=config)
            return CompileResponse(
                gbnf=gbnf,
                rule_count=gbnf.count("::="),
                rule_names=["root"],
                root_rule="root",
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source_type: {req.source_type}")

    except Exception as err:
        raise HTTPException(status_code=422, detail=str(err))


@router.post("/validate", response_model=ValidateResponse)
async def api_validate(req: ValidateRequest) -> ValidateResponse:
    """Validate JSON payload against schema."""
    sm = JsonStateMachine(req.schema_data)
    status, err = sm.check_partial_json(req.payload)
    return ValidateResponse(
        status=status,
        is_valid=(status == ParserStatus.ACCEPTING),
        error=err,
    )


@router.post("/mask", response_model=MaskResponse)
async def api_mask(req: MaskRequest) -> MaskResponse:
    """Evaluate candidate tokens and measure latency."""
    trie = get_shared_vocab(req.vocab_size)
    masker = LogitMasker(req.schema_data, trie)
    _, res = masker.compute_mask(req.prefix)

    samples = [
        {"id": tid, "str": trie.get_token_str(tid)}
        for tid in res.allowed_tokens[:25]
    ]

    return MaskResponse(
        allowed_count=len(res.allowed_tokens),
        total_vocab=trie.vocab_size,
        is_accepting=res.is_accepting,
        latency_us=res.latency_us,
        sample_allowed_tokens=samples,
        status=res.status,
    )


@router.get("/examples")
async def api_examples() -> Dict[str, Any]:
    """Return curated production schemas."""
    return {
        "tool_call": {
            "title": "Agent Tool Call",
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["search_code", "read_file", "run_command"]},
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path": {"type": "string"},
                            "timeout_ms": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                },
                "required": ["tool", "parameters"]
            }
        },
        "user_profile": {
            "title": "User Profile (Strict Regex Formats)",
            "schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                    "username": {"type": "string", "pattern": "^[a-zA-Z0-9_]{3,16}$"},
                    "email": {"type": "string", "format": "email"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "role": {"type": "string", "enum": ["admin", "member", "guest"]},
                    "verified": {"type": "boolean"}
                },
                "required": ["id", "username", "email", "role"]
            }
        },
        "sql_generator": {
            "title": "Structured SQL Query Generator",
            "schema": {
                "type": "object",
                "properties": {
                    "dialect": {"type": "string", "enum": ["postgres", "sqlite", "clickhouse"]},
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "query": {"type": "string"},
                    "cost_estimate": {"type": "number"}
                },
                "required": ["dialect", "tables", "query"]
            }
        }
    }


class RepairApiRequest(BaseModel):
    partial_text: str
    schema_data: Optional[Dict[str, Any]] = None


@router.post("/repair")
async def api_repair(req: RepairApiRequest) -> Dict[str, Any]:
    """Repair and heal an incomplete JSON string."""
    res = heal_partial_json(req.partial_text, schema=req.schema_data)
    return res.to_dict() | {"repaired_text": res.repaired_text}


class SqlApiRequest(BaseModel):
    sql: str
    format_type: str = "json"


@router.post("/sql")
async def api_sql(req: SqlApiRequest) -> Dict[str, Any]:
    """Compile SQL DDL to GBNF."""
    try:
        gbnf = compile_sql_ddl_to_gbnf(req.sql, output_format=req.format_type)
        return {"gbnf": gbnf, "format": req.format_type}
    except Exception as err:
        raise HTTPException(status_code=422, detail=str(err))


class LintApiRequest(BaseModel):
    gbnf: str


@router.post("/lint")
async def api_lint(req: LintApiRequest) -> Dict[str, Any]:
    """Lint and analyze a GBNF grammar."""
    linter = GbnfLinter(req.gbnf)
    diags = linter.lint()
    return {
        "diagnostics": [d.to_dict() for d in diags],
        "is_valid": not any(d.severity == "error" for d in diags),
    }


@router.post("/fmt")
async def api_fmt(req: LintApiRequest) -> Dict[str, Any]:
    """Format and align GBNF grammar."""
    formatted = format_gbnf(req.gbnf)
    return {"formatted_gbnf": formatted}


class ExportCodeRequest(BaseModel):
    schema_data: Dict[str, Any]
    gbnf: str


@router.post("/export_code")
async def api_export_code(req: ExportCodeRequest) -> Dict[str, str]:
    """Generate production client code snippets for python, vLLM, and cURL."""
    schema_json = json.dumps(req.schema_data, indent=2)
    escaped_gbnf = json.dumps(req.gbnf)

    python_llama = f'''from llama_cpp import Llama, LlamaGrammar

grammar = LlamaGrammar.from_string({escaped_gbnf})
llm = Llama(model_path="./model.gguf")

response = llm(
    "Your prompt here:",
    grammar=grammar,
    max_tokens=512
)
print(response["choices"][0]["text"])
'''

    python_vllm = f'''from vllm import LLM, SamplingParams
from grammarforge import GrammarForgeLogitsProcessor, VocabularyTrie

schema = {schema_json}
vocab = VocabularyTrie.from_huggingface_json("path/to/tokenizer.json")
processor = GrammarForgeLogitsProcessor(schema=schema, vocab_trie=vocab)

sampling_params = SamplingParams(
    temperature=0.0,
    logits_processors=[processor],
    max_tokens=512
)

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
output = llm.generate(["Prompt here:"], sampling_params)
'''

    curl_gateway = f'''curl http://127.0.0.1:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "local-model",
    "messages": [{{"role": "user", "content": "Extract data"}}],
    "response_format": {{
      "type": "json_object",
      "schema": {schema_json}
    }}
  }}'
'''

    return {
        "python_llama_cpp": python_llama,
        "python_vllm": python_vllm,
        "curl_openai_proxy": curl_gateway,
    }

