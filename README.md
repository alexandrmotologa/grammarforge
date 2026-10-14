# GrammarForge

Constrained decoding engine and JSON Schema to GBNF grammar compiler with sub-millisecond logit masking for local LLM inference engines.

## Overview

Local inference runtimes such as llama.cpp, vLLM, and InferOps need guarantees that model outputs match expected schemas. Without grammar enforcement, LLMs frequently emit malformed JSON, missing fields, or hallucinated types that break downstream code.

GrammarForge compiles JSON Schema (Draft 7 and 2020-12) and Pydantic models directly into GGML BNF (GBNF) grammars. It provides a prefix-trie token masking engine that computes valid token bitmasks in under 50 microseconds per generation step.

## Key Capabilities

- **JSON Schema to GBNF Compiler**: Converts primitive types, objects, arrays, nested `$ref` definitions, `anyOf`/`oneOf` unions, enums, and regex patterns to deterministic GBNF grammar rules.
- **Regex to GBNF Translation**: Compiles arbitrary regular expressions into GBNF character classes and sequences.
- **Pydantic Model Introspection**: Compiles Pydantic BaseModel classes, TypedDicts, and standard dataclasses into GBNF without manual schema serialization.
- **Token Vocabulary Prefix Trie**: Byte-level prefix trie indexing tokenizer vocabularies for instant token validation.
- **Fast Logit Masking**: Produces floating point bias vectors (`0.0` for allowed, `-inf` for forbidden) across vocabulary sizes up to 128k tokens in sub-millisecond latency.
- **LLM Gateway Integration**: Drop-in `LogitsProcessor` implementations for vLLM, HuggingFace Transformers, and llama.cpp.
- **Developer CLI**: Full command-line interface powered by Typer for compiling schemas, testing regexes, benchmarking masking latency, and validating outputs.
- **Interactive Web Studio**: Built-in FastAPI dashboard with real-time grammar compilation, token stepping simulator, and visualizer.

## Architecture

```
+-----------------------------------------------------------+
|                        Input Layer                        |
|  JSON Schema (Draft 7/2020-12) | Pydantic Models | Regex  |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   Compiler Pipeline                       |
|   - AST Normalizer & Reference Resolver                   |
|   - Regex to GBNF Subrule Generator                       |
|   - Deterministic GBNF Rule Formatter                     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   Grammar Engine                          |
|   - GBNF Parser & State Stepper                           |
|   - Byte-level Prefix Trie (Tokenizer Vocab)              |
|   - Bitmask & Logit Bias Calculator (<50us/step)          |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   Gateway Delivery                        |
|   - vLLM LogitsProcessor                                  |
|   - llama.cpp Grammar Bridge                              |
|   - Typer CLI / Interactive FastAPI Studio                |
+-----------------------------------------------------------+
```

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

Benchmark token masking speed against a tokenizer vocabulary:

```bash
grammarforge bench --vocab-size 32000 --iterations 500
```

Start the interactive Web Studio:

```bash
grammarforge studio --host 127.0.0.1 --port 8000
```

## License

MIT License. See [LICENSE](LICENSE) for details.
