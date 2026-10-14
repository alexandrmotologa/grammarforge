"""LLM Gateway integration adapters for vLLM, HuggingFace, and llama.cpp."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.core.config import CompilerConfig, EngineConfig
from grammarforge.engine.masking import LogitMasker
from grammarforge.engine.state_machine import JsonStateMachine, ParserStatus
from grammarforge.engine.trie import VocabularyTrie


class GrammarForgeLogitsProcessor:
    """Drop-in LogitsProcessor compatible with HuggingFace, vLLM, and InferOps."""

    def __init__(
        self,
        schema: Dict[str, Any],
        vocab_trie: VocabularyTrie,
        tokenizer_decode_fn: Optional[Callable[[List[int]], str]] = None,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.schema = schema
        self.vocab = vocab_trie
        self.decode_fn = tokenizer_decode_fn or self._default_decode
        self.masker = LogitMasker(schema, vocab_trie, config=config)
        self.prompt_length: Optional[int] = None

    def _default_decode(self, token_ids: List[int]) -> str:
        """Decode token sequence using indexed trie byte mappings."""
        raw_bytes = b"".join(self.vocab.get_token_bytes(tid) for tid in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def set_prompt_length(self, length: int) -> None:
        """Set prompt token count so only newly generated completion tokens are decoded."""
        self.prompt_length = length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Apply logit masking in-place or return masked scores."""
        # Normalize input_ids to List[int]
        if hasattr(input_ids, "tolist"):
            ids_list = input_ids.tolist()
            if ids_list and isinstance(ids_list[0], list):
                ids_list = ids_list[0]
        else:
            ids_list = list(input_ids)

        if self.prompt_length is not None and len(ids_list) >= self.prompt_length:
            generated_ids = ids_list[self.prompt_length:]
        else:
            generated_ids = ids_list

        current_text = self.decode_fn(generated_ids)
        mask, _ = self.masker.compute_mask(current_text)

        # Apply mask to scores
        if hasattr(scores, "shape") and hasattr(scores, "__getitem__"):
            # Torch / Numpy array
            for idx, val in enumerate(mask):
                if val < 0.0 and idx < scores.shape[-1]:
                    scores[..., idx] = val
            return scores
        elif isinstance(scores, list):
            # Pure Python list
            return [s + m for s, m in zip(scores, mask)]
        return scores


class StreamingValidator:
    """Token-by-token stream validator that enforces schema compliance during live LLM streaming."""

    def __init__(
        self,
        schema: Dict[str, Any],
        vocab_trie: Optional[VocabularyTrie] = None,
    ) -> None:
        self.schema = schema
        self.vocab = vocab_trie
        self.state_machine = JsonStateMachine(schema)
        self.accumulated_text = ""
        self.token_history: List[int] = []

    def feed_token(self, token: Union[int, str]) -> Tuple[str, Optional[str]]:
        """Feed a new token (ID or string chunk) into the validator.

        Returns:
            Tuple of (status, error_message).
            status is one of: 'valid', 'accepting', or 'rejected'.
        """
        if isinstance(token, int):
            if not self.vocab:
                raise ValueError("VocabularyTrie required when feeding integer token IDs.")
            chunk = self.vocab.get_token_str(token)
            self.token_history.append(token)
        else:
            chunk = token

        self.accumulated_text += chunk
        status, err = self.state_machine.check_partial_json(self.accumulated_text)
        return status, err

    def is_complete(self) -> bool:
        """Check if generated text forms a valid complete JSON target."""
        status, _ = self.state_machine.check_partial_json(self.accumulated_text)
        return status == ParserStatus.ACCEPTING

    def reset(self) -> None:
        """Reset validator state for new generation."""
        self.accumulated_text = ""
        self.token_history.clear()


class LlamaCppGrammarConnector:
    """Helper connecting GrammarForge schemas directly with llama.cpp."""

    @staticmethod
    def get_grammar_string(schema: Dict[str, Any], config: Optional[CompilerConfig] = None) -> str:
        """Compile schema into llama.cpp compatible GBNF grammar string."""
        return compile_json_schema(schema, config=config)

    @staticmethod
    def create_llama_cpp_params(schema: Dict[str, Any], config: Optional[CompilerConfig] = None) -> Dict[str, Any]:
        """Generate kwargs ready to be passed to llama-cpp-python."""
        grammar_str = compile_json_schema(schema, config=config)
        return {
            "grammar": grammar_str,
        }
