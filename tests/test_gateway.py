"""Unit tests for Gateway adapters (LogitsProcessor, StreamingValidator)."""

import pytest
from grammarforge.engine.gateway import (
    GrammarForgeLogitsProcessor,
    LlamaCppGrammarConnector,
    StreamingValidator,
)
from grammarforge.engine.trie import VocabularyTrie


def test_streaming_validator() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }
    validator = StreamingValidator(schema)

    # Stream chunks
    status1, err1 = validator.feed_token('{"query"')
    assert status1 == "valid"
    assert not validator.is_complete()

    status2, err2 = validator.feed_token(': "test"}')
    assert status2 == "accepting"
    assert validator.is_complete()


def test_logits_processor_call() -> None:
    schema = {"type": "string"}
    trie = VocabularyTrie.create_synthetic_vocab(size=400)
    processor = GrammarForgeLogitsProcessor(schema, trie)

    # Initial input_ids
    input_ids = [1]  # BOS
    scores = [0.0] * 400

    masked_scores = processor(input_ids, scores)
    assert len(masked_scores) == 400
    # Disallowed tokens should have large negative bias
    assert min(masked_scores) <= -1e8


def test_llama_cpp_connector() -> None:
    schema = {"type": "boolean"}
    params = LlamaCppGrammarConnector.create_llama_cpp_params(schema)
    assert "grammar" in params
    assert "root ::= json-bool" in params["grammar"]
