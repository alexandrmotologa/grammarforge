# GrammarForge Architecture

GrammarForge provides constrained decoding and grammar compilation for local large language model gateways.

## High-Level System Architecture

The pipeline consists of three core components:

1. **Compiler Layer**: Ingests JSON Schemas (Draft 7 and 2020-12), Python Pydantic models, or standard regular expressions. It translates these constraints into an intermediate representation and emits optimized GBNF (GGML BNF) grammar specifications.
2. **Engine Layer**: Indexes tokenizer vocabularies inside a byte-level prefix trie. It executes an incremental state machine tracking JSON structural balance, property keys, and value types to compute allowed token masks.
3. **Gateway Delivery Layer**: Connects with inference runtimes through drop-in adapters, including HuggingFace Transformers, vLLM LogitsProcessors, and llama.cpp grammar strings.

```
                  +---------------------------+
                  |  JSON Schema / Pydantic   |
                  +---------------------------+
                                |
                                v
+-----------------------------------------------------------+
|                      Compiler Layer                       |
|                                                           |
|   +-------------------+          +--------------------+   |
|   | RegexParser       | -------> | GBNF IR AST        |   |
|   +-------------------+          +--------------------+   |
|             ^                              |              |
|             |                              v              |
|   +-------------------+          +--------------------+   |
|   | SchemaCompiler    |          | Grammar Serializer |   |
|   +-------------------+          +--------------------+   |
+-----------------------------------------------------------+
                                |
                   (GBNF Text / Syntax Tree)
                                |
                                v
+-----------------------------------------------------------+
|                       Engine Layer                        |
|                                                           |
|   +-------------------+          +--------------------+   |
|   | VocabularyTrie    | <------> | LogitMasker        |   |
|   | (Byte-level index)|          | (Sub-ms evaluation)|   |
|   +-------------------+          +--------------------+   |
|                                            ^              |
|                                            |              |
|                                  +--------------------+   |
|                                  | JsonStateMachine   |   |
|                                  +--------------------+   |
+-----------------------------------------------------------+
                                |
                                v
                  +---------------------------+
                  |    vLLM / llama.cpp / HF   |
                  +---------------------------+
```

## Compiler Pipeline

### GBNF Intermediate Representation (IR)

The compiler models GBNF rules using explicit typed AST nodes:

- `Terminal`: Represents literal string tokens with escaping for quotes, newlines, tabs, and hex/unicode bytes.
- `CharClass`: Represents character sets and ranges, supporting both positive `[0-9a-z]` and inverted `[^"\\]` classes.
- `RuleRef`: Refers to non-terminal rules by sanitized identifier names matching `[a-zA-Z0-9_-]+`.
- `Sequence`: Concatenates ordered child expressions.
- `Alternative`: Represents disjunction (`|`), automatically flattening nested branches and deduplicating identical choices.
- `Repetition`: Represents quantifiers including zero-or-more (`*`), one-or-more (`+`), and optional (`?`).
- `Rule`: Holds the rule name, expression body, and optional comments.
- `Grammar`: Organizes rules, guarantees that the root rule appears first, and performs topological dependency traversal.

### Recursive Reference Resolution

JSON Schemas often contain recursive self-referential structures (e.g. linked lists, AST trees, or open JSON values). The compiler handles references through the following steps:

1. Detects `$ref` targets pointing to local `#/$defs/` or `#/definitions/`.
2. Generates a placeholder `RuleRef` forward reference before compiling target bodies to prevent recursion cycles.
3. Compiles the target schema once, associates it with the named rule, and links dependent references.

### Key Ordering Modes

- **Canonical Ordering**: Enforces a fixed property order matching the schema definition. This avoids combinatorial state explosion during logit masking, enabling sub-millisecond evaluation across large vocabularies.
- **Flexible Ordering**: Allows keys in arbitrary order through an alternation of property member rules.

## Vocabulary Prefix Trie & Masking Engine

Constrained decoding requires identifying which tokens in a tokenizer vocabulary can be legally sampled at step `t`.

A naive check across 32,000 or 128,000 tokens per generation step would introduce prohibitive latency. GrammarForge solves this through two optimizations:

1. **Byte-Level Prefix Trie**: The vocabulary is indexed in a compact byte-level trie (`VocabularyTrie`). Given a sequence of candidate bytes, the trie enables prefix lookups without scanning the entire vocabulary array.
2. **Prefix Transition Cache**: Partial JSON states are cached using an LRU policy. Consecutive tokens traversing common structures (such as object opening, string quotes, or delimiters) hit cached bitmasks directly, keeping step latency under 50 microseconds.

## Gateway Integration Architecture

### vLLM & HuggingFace LogitsProcessor

The `GrammarForgeLogitsProcessor` integrates with standard Python sampling pipelines:

```python
from grammarforge import GrammarForgeLogitsProcessor, VocabularyTrie

# Initialize vocabulary trie from HuggingFace tokenizer
trie = VocabularyTrie.from_huggingface_json("path/to/tokenizer.json")

# Create processor for target schema
processor = GrammarForgeLogitsProcessor(schema=user_schema, vocab_trie=trie)
processor.set_prompt_length(len(prompt_tokens))

# Pass directly to model.generate(logits_processor=[processor])
```

During each forward pass, the processor receives generated token IDs, extracts newly emitted tokens, calculates the allowed token set, and masks disallowed logits by setting their values to `-1e9`.
