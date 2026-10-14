# LLM Gateway Integration Guide

This guide describes how to connect GrammarForge with vLLM, llama.cpp, and HuggingFace Transformers.

## 1. vLLM Integration

vLLM supports custom logits processors via its sampling configuration.

```python
from vllm import LLM, SamplingParams
from grammarforge import GrammarForgeLogitsProcessor, VocabularyTrie

# Load vocabulary
vocab_trie = VocabularyTrie.from_huggingface_json("path/to/tokenizer.json")

# Define schema
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "confidence": {"type": "number"}
    },
    "required": ["summary", "sentiment", "confidence"]
}

# Initialize processor
logits_processor = GrammarForgeLogitsProcessor(schema, vocab_trie)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    logits_processors=[logits_processor]
)

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
outputs = llm.generate(["Classify the following text..."], sampling_params)
```

## 2. llama.cpp / llama-cpp-python

llama.cpp accepts precompiled GBNF grammar strings natively:

```python
from llama_cpp import Llama, LlamaGrammar
from grammarforge import compile_json_schema

schema = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "arguments": {"type": "object"}
    },
    "required": ["function", "arguments"]
}

# Compile schema to GBNF
gbnf_string = compile_json_schema(schema)

# Load grammar in llama-cpp-python
grammar = LlamaGrammar.from_string(gbnf_string)

llm = Llama(model_path="./models/llama-3-8b.Q4_K_M.gguf")
response = llm(
    "Generate a database backup command in JSON format:",
    grammar=grammar,
    max_tokens=200
)
print(response["choices"][0]["text"])
```

## 3. Streaming Output Validation

For real-time streaming interfaces (e.g. Server-Sent Events, WebSockets, or UI dashboards), use `StreamingValidator` to detect schema deviations on each token chunk:

```python
from grammarforge import StreamingValidator

validator = StreamingValidator(schema)

for chunk in llm_stream:
    status, error = validator.feed_token(chunk)
    if status == "rejected":
        print(f"Schema violation detected: {error}")
        break
    elif status == "accepting":
        print("Valid complete payload received.")
```
