<p align="center">
  <img src="docs/images/grammarforge_logo.png?raw=true" alt="GrammarForge Logo" width="130" style="border-radius: 28px;" />
</p>

<h1 align="center">GrammarForge</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=flat-square" alt="Python Version" />
  <img src="https://img.shields.io/badge/tests-67%20passed-brightgreen?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/coverage-85%25-green?style=flat-square" alt="Coverage" />
  <img src="https://img.shields.io/badge/license-MIT-purple?style=flat-square" alt="License" />
</p>

<p align="center">
  <b>High-throughput constrained decoding engine and JSON Schema to GBNF compiler with sub-millisecond logit masking for local LLM gateways.</b>
</p>

<p align="center">
  <img src="docs/images/grammarforge_demo.gif?raw=true" alt="GrammarForge Studio Demo" width="920" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.18);" />
</p>

## Overview

Local inference runtimes such as llama.cpp, vLLM, and InferOps need guarantees that model outputs match expected schemas. Without grammar enforcement, LLMs frequently emit malformed JSON, missing fields, or invalid types that break downstream code.

GrammarForge compiles JSON Schema (Draft 7 and 2020-12), Pydantic models, SQL DDL statements, and regular expressions directly into GGML BNF (GBNF) grammars. It provides a prefix trie token masking engine that computes valid token bitmasks in under 75 microseconds per generation step.

## Key Capabilities

- **JSON Schema to GBNF Compiler**: Converts primitive types, objects, arrays, nested `$ref` definitions, `anyOf`/`oneOf` unions, enums, and regex patterns to deterministic GBNF grammar rules.
- **SQL DDL to GBNF Compiler**: Converts `CREATE TABLE` statements with data types and `CHECK` constraints into structured JSON grammar rules for synthetic data generation and ETL.
- **Regex to GBNF Translation**: Compiles arbitrary regular expressions into GBNF character classes and sequences.
- **Pydantic Model Introspection**: Compiles Pydantic BaseModel classes, TypedDicts, and standard dataclasses into GBNF without manual schema serialization.
- **Token Vocabulary Prefix Trie**: Byte-level prefix trie indexing tokenizer vocabularies for instant token validation.
- **Sub-Millisecond Logit Masking**: Produces floating point bias vectors (`0.0` for allowed, `-1e9` for forbidden) across vocabulary sizes up to 128k tokens in microsecond latency.
- **Payload Auto-Repair**: Real-time healing for truncated or partially streamed JSON responses, closing unclosed brackets and strings cleanly.
- **LLM Gateway Integration**: Drop-in `LogitsProcessor` implementations for vLLM, HuggingFace Transformers, and llama.cpp, plus an OpenAI-compatible reverse proxy.
- **Developer CLI**: Command-line interface powered by Typer for compiling schemas, testing regexes, benchmarking masking latency, and validating outputs.
- **Interactive Web Studio**: Built-in FastAPI dashboard with real-time grammar compilation, token stepping simulator, SQL compiler, and one-click code export.

## Studio Interface Showcase

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Interactive Schema Compiler & Token Mask Stepper</b><br/>
      <img src="docs/images/grammarforge-1.png?raw=true" alt="Interactive Schema Compiler & Stepper" width="100%" style="border-radius: 8px;"/>
      <p align="left"><sub>Visualizes JSON Schema compilation into 11 GBNF rules alongside vocabulary trie search, microsecond latency metrics, and step-by-step candidate token chips.</sub></p>
    </td>
    <td width="50%" align="center">
      <b>SQL DDL to Constrained Grammar Compiler</b><br/>
      <img src="docs/images/grammarforge-2.png?raw=true" alt="SQL DDL Compiler" width="100%" style="border-radius: 8px;"/>
      <p align="left"><sub>Parses PostgreSQL / SQLite DDL schemas and emits strict GBNF rules verifying field types, timestamps, and CHECK constraint membership.</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Streaming JSON Auto-Repair & Validator</b><br/>
      <img src="docs/images/grammarforge-3.png?raw=true" alt="Streaming JSON Auto-Repair" width="100%" style="border-radius: 8px;"/>
      <p align="left"><sub>Real-time payload inspection and repair engine that synthesizes missing closing delimiters for interrupted LLM generations.</sub></p>
    </td>
    <td width="50%" align="center">
      <b>One-Click Client Code Export</b><br/>
      <img src="docs/images/grammarforge-4.png?raw=true" alt="One-Click Client Code Export" width="100%" style="border-radius: 8px;"/>
      <p align="left"><sub>Provides copy-paste production snippets for llama-cpp-python, vLLM LogitsProcessor, and OpenAI-compatible proxy gateway requests.</sub></p>
    </td>
  </tr>
</table>

## Architecture

```
+-----------------------------------------------------------------+
|                          Input Layer                            |
|  JSON Schema (Draft 7/2020) | Pydantic | SQL DDL | Regex        |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                     Compiler Pipeline                           |
|   - AST Normalizer & Reference Resolver                         |
|   - Regex to GBNF Subrule Generator                             |
|   - SQL DDL Lexer & Constraint Translator                       |
|   - Deterministic GBNF Rule Formatter & Linter                  |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                     Grammar Engine                              |
|   - GBNF Parser & State Stepper                                 |
|   - Byte-level Prefix Trie (Tokenizer Vocab)                    |
|   - Dynamic Bitmask & Logit Bias Calculator (<75us/step)        |
|   - JSON Stream Auto-Repair Engine                              |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                     Gateway Delivery                            |
|   - vLLM LogitsProcessor / HuggingFace Pipeline                 |
|   - llama.cpp Grammar Bridge                                    |
|   - OpenAI-Compatible Reverse Proxy Gateway                     |
|   - Typer CLI / Interactive FastAPI Studio                      |
+-----------------------------------------------------------------+
```

Detailed architectural diagrams and pipeline internals are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

### Installation

```bash
pip install -e .
```

### CLI Usage

Compile a JSON schema to GBNF:

```bash
grammarforge compile user_schema.json -o user.gbnf
```

Compile a Pydantic model:

```bash
grammarforge pydantic my_app.models:UserRequest -o request.gbnf
```

Compile a regular expression pattern directly into GBNF:

```bash
grammarforge regex "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
```

Compile a SQL DDL schema directly to GBNF:

```bash
grammarforge sql "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255) NOT NULL);"
```

Auto-repair a truncated JSON completion:

```bash
grammarforge repair '{"name": "Alex", "tags": ["admin", "dev"'
```

Lint and format GBNF grammar files:

```bash
grammarforge lint grammar.gbnf
grammarforge fmt grammar.gbnf --write
```

Start the OpenAI-compatible reverse proxy gateway:

```bash
grammarforge proxy --upstream http://127.0.0.1:11434 --port 8080
```

Validate a JSON payload against a schema:

```bash
grammarforge validate user_schema.json '{"name": "Alice", "age": 30}'
```

Benchmark token masking speed against a tokenizer vocabulary:

```bash
grammarforge bench --vocab-size 32000 --iterations 500
```

Start the interactive Web Studio:

```bash
grammarforge studio --host 127.0.0.1 --port 8000
```

## Python API Usage

### Direct GBNF Compilation

```python
from grammarforge import compile_json_schema, CompilerConfig, WhitespaceMode

schema = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["search", "read_file"]},
        "query": {"type": "string"}
    },
    "required": ["tool", "query"]
}

# Compile with lenient whitespace (standard GBNF)
gbnf = compile_json_schema(schema, config=CompilerConfig(whitespace_mode=WhitespaceMode.LENIENT))
print(gbnf)
```

### vLLM LogitsProcessor Integration

```python
from grammarforge import GrammarForgeLogitsProcessor, VocabularyTrie

# Load vocabulary from tokenizer file
trie = VocabularyTrie.from_huggingface_json("path/to/tokenizer.json")

# Instantiate processor
processor = GrammarForgeLogitsProcessor(schema=schema, vocab_trie=trie)

# Pass to vLLM SamplingParams(logits_processors=[processor])
```

## Documentation

- [Architecture & Internal Design](docs/ARCHITECTURE.md): Complete breakdown of IR AST, prefix trie indexing, and parser state machine.
- [GBNF Syntax & Rule Reference](docs/GBNF_GUIDE.md): Grammar syntax rules, character class escaping, and formatting guidelines.
- [Gateway Integration Guide](docs/GATEWAY_INTEGRATION.md): Integration tutorials for vLLM, llama-cpp-python, and streaming validation.

## License

MIT License. See [LICENSE](LICENSE) for details.
