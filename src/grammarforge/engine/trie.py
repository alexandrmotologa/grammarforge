"""Byte-level prefix trie indexing tokenizer vocabulary for sub-millisecond constrained token search."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union


class TrieNode:
    """A single node in the vocabulary byte trie."""

    __slots__ = ("children", "token_ids", "is_terminal")

    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = {}
        self.token_ids: List[int] = []
        self.is_terminal: bool = False

    def add_child(self, byte_val: int) -> TrieNode:
        if byte_val not in self.children:
            node = TrieNode()
            self.children[byte_val] = node
            return node
        return self.children[byte_val]


class VocabularyTrie:
    """Prefix trie indexing full tokenizer vocabulary for fast token search."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.token_to_bytes: Dict[int, bytes] = {}
        self.bytes_to_tokens: Dict[bytes, List[int]] = {}
        self.vocab_size: int = 0
        self.special_tokens: Set[int] = set()

    def insert(self, token_id: int, token_bytes: Union[bytes, str], is_special: bool = False) -> None:
        """Insert a token into the trie."""
        if isinstance(token_bytes, str):
            b_val = token_bytes.encode("utf-8")
        else:
            b_val = token_bytes

        self.token_to_bytes[token_id] = b_val
        if b_val not in self.bytes_to_tokens:
            self.bytes_to_tokens[b_val] = []
        self.bytes_to_tokens[b_val].append(token_id)

        if is_special:
            self.special_tokens.add(token_id)
            # Special tokens are often handled out-of-band (e.g. EOS/BOS)
            return

        curr = self.root
        for b in b_val:
            curr = curr.add_child(b)

        curr.is_terminal = True
        curr.token_ids.append(token_id)

        if token_id >= self.vocab_size:
            self.vocab_size = token_id + 1

    def get_token_bytes(self, token_id: int) -> bytes:
        """Return the raw byte sequence for a token ID."""
        return self.token_to_bytes.get(token_id, b"")

    def get_token_str(self, token_id: int) -> str:
        """Return the decoded string for a token ID, replacing invalid utf-8 sequences."""
        raw = self.token_to_bytes.get(token_id, b"")
        return raw.decode("utf-8", errors="replace")

    def find_node(self, byte_seq: bytes) -> Optional[TrieNode]:
        """Navigate to the TrieNode corresponding to byte_seq, if it exists."""
        curr = self.root
        for b in byte_seq:
            if b not in curr.children:
                return None
            curr = curr.children[b]
        return curr

    def tokens_with_prefix(self, prefix: Union[bytes, str]) -> List[int]:
        """Find all token IDs whose byte representations start with the given prefix."""
        b_prefix = prefix.encode("utf-8") if isinstance(prefix, str) else prefix
        node = self.find_node(b_prefix)
        if not node:
            return []

        results: List[int] = []
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr.is_terminal:
                results.extend(curr.token_ids)
            for child in curr.children.values():
                stack.append(child)
        return results

    def collect_all_tokens(self) -> List[int]:
        """Collect all indexed token IDs."""
        return list(self.token_to_bytes.keys())

    @classmethod
    def from_dict(cls, vocab: Dict[Union[int, str], Union[str, bytes]]) -> VocabularyTrie:
        """Construct trie from a vocabulary mapping dictionary."""
        trie = cls()
        for k, v in vocab.items():
            token_id = int(k)
            trie.insert(token_id, v)
        return trie

    @classmethod
    def from_list(cls, token_list: Iterable[Union[str, bytes]]) -> VocabularyTrie:
        """Construct trie from an ordered list of tokens where index is token_id."""
        trie = cls()
        for idx, tok in enumerate(token_list):
            trie.insert(idx, tok)
        return trie

    @classmethod
    def from_huggingface_json(cls, file_path: str) -> VocabularyTrie:
        """Load vocabulary from HuggingFace tokenizer.json or vocab.json."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trie = cls()
        if "model" in data and "vocab" in data["model"]:
            # tokenizer.json format: {"model": {"vocab": {"tok": id, ...}}}
            vocab = data["model"]["vocab"]
            for tok_str, tok_id in vocab.items():
                trie.insert(int(tok_id), tok_str)
        elif isinstance(data, dict):
            # vocab.json format: {"tok": id, ...} or {id: "tok", ...}
            for k, v in data.items():
                if isinstance(v, int):
                    trie.insert(v, k)
                else:
                    trie.insert(int(k), str(v))
        return trie

    @classmethod
    def create_synthetic_vocab(cls, size: int = 32000) -> VocabularyTrie:
        """Generate a realistic synthetic vocabulary for benchmarking and unit tests."""
        trie = cls()
        # 1. Special tokens
        trie.insert(0, "<unk>", is_special=True)
        trie.insert(1, "<s>", is_special=True)
        trie.insert(2, "</s>", is_special=True)

        # 2. Byte tokens for all 256 byte values
        for b in range(256):
            trie.insert(3 + b, bytes([b]))

        # 3. Common JSON structural characters & tokens
        common_tokens = [
            '{"', '"}', '":', '",', '": "', '": [', '": {', '",\n  "',
            "true", "false", "null", "\n  ", "    ", "\n", "\t",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "id", "name", "email", "status", "count", "type", "description",
            "active", "pending", "success", "error", "items", "data", "results",
            "title", "user", "role", "admin", "age", "tags", "created_at"
        ]
        curr_id = 3 + 256
        for tok in common_tokens:
            trie.insert(curr_id, tok)
            curr_id += 1

        # 4. Fill remaining with synthetic subwords
        syllables = ["al", "ex", "an", "der", "gra", "mmar", "for", "ge", "con", "strain", "ed", "de", "co", "ding"]
        while curr_id < size:
            idx1 = curr_id % len(syllables)
            idx2 = (curr_id // len(syllables)) % len(syllables)
            tok_str = f" {syllables[idx1]}{syllables[idx2]}_{curr_id % 100}"
            trie.insert(curr_id, tok_str)
            curr_id += 1

        return trie
