"""Unit tests for Logit Masking Engine."""

import pytest
from grammarforge.engine.masking import LogitMasker
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie


def test_state_machine_partial_json() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    sm = JsonStateMachine(schema)

    # Valid incomplete prefix
    status, err = sm.check_partial_json('{"name":')
    assert status == ParserStatus.VALID
    assert err is None

    # Invalid structural prefix
    status_bad, err_bad = sm.check_partial_json('{"name": ]')
    assert status_bad == ParserStatus.REJECTED

    # Complete valid payload
    status_done, _ = sm.check_partial_json('{"name": "Alice"}')
    assert status_done == ParserStatus.ACCEPTING


def test_logit_masker_evaluation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
        },
        "required": ["status"],
    }
    trie = VocabularyTrie.create_synthetic_vocab(size=500)
    masker = LogitMasker(schema, trie)

    # Empty prefix: must start with '{' or whitespace
    mask, res = masker.compute_mask("")
    assert len(res.allowed_tokens) > 0
    assert not res.is_accepting
    assert res.latency_us > 0.0

    # Complete valid json: EOS must be allowed
    mask_done, res_done = masker.compute_mask('{"status": "ok"}')
    assert res_done.is_accepting
    assert masker.config.eos_token_id in res_done.allowed_tokens
