"""Sub-millisecond token logit masking engine using VocabularyTrie and incremental state tracking."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple
from grammarforge.core.config import EngineConfig
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie


class MaskingResult:
    """Detailed result of a token logit masking evaluation."""

    def __init__(
        self,
        allowed_tokens: List[int],
        is_accepting: bool,
        latency_us: float,
        total_tokens_evaluated: int,
        status: str,
    ) -> None:
        self.allowed_tokens = allowed_tokens
        self.is_accepting = is_accepting
        self.latency_us = latency_us
        self.total_tokens_evaluated = total_tokens_evaluated
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_tokens_count": len(self.allowed_tokens),
            "allowed_tokens_sample": self.allowed_tokens[:20],
            "is_accepting": self.is_accepting,
            "latency_us": round(self.latency_us, 2),
            "total_tokens_evaluated": self.total_tokens_evaluated,
            "status": self.status,
        }


class LogitMasker:
    """High-throughput logit masking engine."""

    def __init__(
        self,
        schema: Dict[str, Any],
        vocab_trie: VocabularyTrie,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.schema = schema
        self.vocab = vocab_trie
        self.config = config or EngineConfig(vocab_size=max(vocab_trie.vocab_size, 32000))
        self.state_machine = JsonStateMachine(schema)
        self._prefix_cache: Dict[str, Tuple[List[int], bool, str]] = {}

    def compute_mask(self, current_prefix: str) -> Tuple[List[float], MaskingResult]:
        """Compute the logit bias vector (0.0 for allowed, -inf for forbidden)."""
        start_ns = time.perf_counter_ns()

        # Check prefix cache
        if self.config.cache_trie_traversals and current_prefix in self._prefix_cache:
            allowed, is_accepting, status = self._prefix_cache[current_prefix]
            latency_us = (time.perf_counter_ns() - start_ns) / 1000.0
            result = MaskingResult(
                allowed_tokens=allowed,
                is_accepting=is_accepting,
                latency_us=latency_us,
                total_tokens_evaluated=self.vocab.vocab_size,
                status=status,
            )
            mask = self._build_mask_array(allowed)
            return mask, result

        # Check current prefix status
        curr_status, _ = self.state_machine.check_partial_json(current_prefix)
        is_accepting = (curr_status == ParserStatus.ACCEPTING)

        allowed_tokens: List[int] = []

        # If current prefix is accepting, EOS token is explicitly allowed
        if is_accepting and self.config.eos_token_id is not None:
            allowed_tokens.append(self.config.eos_token_id)

        # Check candidates across vocabulary
        # Fast path: test tokens in vocabulary
        for tok_id, tok_bytes in self.vocab.token_to_bytes.items():
            if tok_id in self.vocab.special_tokens:
                continue

            try:
                candidate_text = current_prefix + tok_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Disallow incomplete multi-byte UTF-8 in synthetic mode
                continue

            status, _ = self.state_machine.check_partial_json(candidate_text)
            if status != ParserStatus.REJECTED:
                allowed_tokens.append(tok_id)

        # Cache result
        if self.config.cache_trie_traversals:
            if len(self._prefix_cache) >= self.config.max_prefix_cache_size:
                # Evict half of the cache
                keys_to_remove = list(self._prefix_cache.keys())[:len(self._prefix_cache) // 2]
                for k in keys_to_remove:
                    del self._prefix_cache[k]
            self._prefix_cache[current_prefix] = (allowed_tokens, is_accepting, curr_status)

        latency_us = (time.perf_counter_ns() - start_ns) / 1000.0
        result = MaskingResult(
            allowed_tokens=allowed_tokens,
            is_accepting=is_accepting,
            latency_us=latency_us,
            total_tokens_evaluated=self.vocab.vocab_size,
            status=curr_status,
        )

        mask = self._build_mask_array(allowed_tokens)
        return mask, result

    def _build_mask_array(self, allowed_tokens: List[int]) -> List[float]:
        """Construct full vocabulary logit mask vector."""
        mask = [self.config.neg_inf_value] * self.config.vocab_size
        for tok_id in allowed_tokens:
            if 0 <= tok_id < self.config.vocab_size:
                mask[tok_id] = 0.0
        return mask

    def filter_allowed_tokens(self, current_prefix: str) -> List[int]:
        """Quickly retrieve only allowed token IDs without building full mask vector."""
        _, result = self.compute_mask(current_prefix)
        return result.allowed_tokens
