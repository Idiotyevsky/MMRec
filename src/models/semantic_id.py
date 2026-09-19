"""Semantic ID vocabulary, collision handling and constrained decoding.

A Semantic ID is a tuple of ``K`` codes, one per residual-quantisation level.
Two things routinely go wrong in generative recommendation and both are handled
explicitly here:

1. **Collisions.** Different items can quantise to the same code tuple. We do
   not pretend they do not exist: the mapper stores ``sid -> [item ids]`` and the
   retriever either re-ranks the colliding candidates by their own embedding or
   appends a disambiguation token when configured to.
2. **Illegal generations.** A language model can emit a code tuple that is not a
   real item. The trie restricts every decoding step to codes that extend a valid
   prefix, so the decoder can only ever produce a real item — or nothing, never a
   random one.

Token layout::

    0                      PAD
    1                      BOS
    2                      EOS
    3 + level * C + code   code token          (level < K, code < C)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3


class SemanticIDMapper:
    def __init__(self, codes: np.ndarray, num_items: int, codebook_size: int) -> None:
        """``codes`` has shape ``(num_items, K)`` with row ``i`` for internal item ``i+1``."""
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2:
            raise ValueError(f"codes must be (num_items, K), got {codes.shape}")
        if codes.shape[0] != num_items:
            raise ValueError(f"expected {num_items} rows of codes, got {codes.shape[0]}")
        self.codes = codes
        self.num_items = int(num_items)
        self.num_levels = int(codes.shape[1])
        self.codebook_size = int(codebook_size)

        self.sid_to_items: dict[tuple, list[int]] = defaultdict(list)
        for i in range(num_items):
            self.sid_to_items[tuple(int(c) for c in codes[i])].append(i + 1)
        self.item_to_sid = {i + 1: tuple(int(c) for c in codes[i]) for i in range(num_items)}

        # prefix -> set of allowed next codes, used by constrained decoding
        self._prefix: dict[tuple, set[int]] = defaultdict(set)
        for sid in self.sid_to_items:
            for k in range(self.num_levels):
                self._prefix[sid[:k]].add(sid[k])
        self._prefix = {k: sorted(v) for k, v in self._prefix.items()}

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + self.num_levels * self.codebook_size

    def token(self, level: int, code: int) -> int:
        return N_SPECIAL + level * self.codebook_size + int(code)

    def decode_token(self, token: int) -> tuple[int, int] | None:
        if token < N_SPECIAL:
            return None
        t = token - N_SPECIAL
        return divmod(t, self.codebook_size)

    def item_tokens(self, item_id: int) -> list[int]:
        sid = self.item_to_sid[item_id]
        return [self.token(k, c) for k, c in enumerate(sid)]

    def sid_from_tokens(self, tokens: list[int]) -> tuple | None:
        sid = []
        for pos, t in enumerate(tokens):
            dc = self.decode_token(t)
            if dc is None or dc[0] != pos:
                return None
            sid.append(dc[1])
        return tuple(sid) if len(sid) == self.num_levels else None

    # ------------------------------------------------------------------
    def allowed_next(self, prefix: tuple) -> list[int]:
        """Token ids that can legally extend ``prefix`` (empty if impossible)."""
        if len(prefix) >= self.num_levels:
            return []
        return [self.token(len(prefix), c) for c in self._prefix.get(tuple(prefix), ())]

    # ------------------------------------------------------------------
    def statistics(self) -> dict:
        n_unique = len(self.sid_to_items)
        collisions = {sid: items for sid, items in self.sid_to_items.items() if len(items) > 1}
        n_colliding_items = sum(len(v) for v in collisions.values())
        return {
            "num_items": self.num_items,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "unique_semantic_ids": n_unique,
            "collision_rate": 1.0 - n_unique / max(self.num_items, 1),
            "num_colliding_sids": len(collisions),
            "num_items_in_collisions": n_colliding_items,
            "max_items_per_sid": max((len(v) for v in self.sid_to_items.values()), default=0),
        }

    def disambiguation_codes(self) -> np.ndarray:
        """Extra integer per item that breaks ties inside a colliding SID."""
        extra = np.zeros(self.num_items, dtype=np.int64)
        for items in self.sid_to_items.values():
            if len(items) > 1:
                for j, it in enumerate(sorted(items)):
                    extra[it - 1] = j
        return extra


class TrieNode:
    """One node of the Semantic-ID trie."""

    __slots__ = ("children", "items", "depth")

    def __init__(self, depth: int) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.items: list[int] = []  # items whose SID *ends* here
        self.depth = depth


class SemanticTrie:
    """Explicit prefix tree over Semantic IDs.

    Constrained decoding needs one question answered at every step: *which codes
    may legally follow this prefix?*  A flat prefix->codes dict answers it, but
    building the tree makes the structure inspectable (depth, branching, leaf
    sizes) and lets ``validate`` prove that the trie and the mapper agree.
    """

    def __init__(self, mapper: SemanticIDMapper) -> None:
        self.mapper = mapper
        self.root = TrieNode(0)
        self._num_nodes = 1
        self._num_leaves = 0
        for sid, items in mapper.sid_to_items.items():
            node = self.root
            for code in sid:
                child = node.children.get(int(code))
                if child is None:
                    child = TrieNode(node.depth + 1)
                    node.children[int(code)] = child
                    self._num_nodes += 1
                node = child
            if not node.items:
                self._num_leaves += 1
            node.items.extend(int(i) for i in items)

    # ------------------------------------------------------------------
    def node(self, prefix: tuple | list) -> TrieNode | None:
        cur = self.root
        for code in prefix:
            cur = cur.children.get(int(code))
            if cur is None:
                return None
        return cur

    def allowed_next(self, prefix: tuple | list) -> list[int]:
        """Token ids that can legally extend ``prefix``."""
        if len(prefix) >= self.mapper.num_levels:
            return []
        node = self.node(prefix)
        if node is None:
            return []
        level = len(prefix)
        return [self.mapper.token(level, c) for c in sorted(node.children)]

    def is_complete(self, prefix: tuple | list) -> bool:
        if len(prefix) != self.mapper.num_levels:
            return False
        node = self.node(prefix)
        return node is not None and bool(node.items)

    def items(self, prefix: tuple | list) -> list[int]:
        node = self.node(prefix)
        return list(node.items) if node else []

    def items_at_most(self, prefix: tuple | list, max_items: int) -> list[int]:
        """Items reachable under ``prefix``, truncated to ``max_items``."""
        node = self.node(prefix)
        if node is None:
            return []
        out: list[int] = []
        stack = [node]
        while stack and len(out) < max_items:
            cur = stack.pop()
            out.extend(cur.items[: max_items - len(out)])
            stack.extend(cur.children.values())
        return out

    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    def statistics(self) -> dict:
        depth_counts = [0] * (self.mapper.num_levels + 1)
        stack = [self.root]
        while stack:
            cur = stack.pop()
            depth_counts[cur.depth] += 1
            stack.extend(cur.children.values())
        return {
            "num_nodes": self._num_nodes,
            "num_leaves": self._num_leaves,
            "num_levels": self.mapper.num_levels,
            "nodes_per_depth": depth_counts,
            "branching_factor_root": len(self.root.children),
        }

    def validate(self) -> None:
        """Assert the trie and the mapper describe the same catalogue."""
        for sid, items in self.mapper.sid_to_items.items():
            node = self.node(sid)
            assert node is not None, f"missing SID {sid}"
            assert sorted(node.items) == sorted(items), f"item mismatch at {sid}"
        for item, sid in self.mapper.item_to_sid.items():
            assert item in self.items(sid), f"item {item} not stored under {sid}"
        for level in range(self.mapper.num_levels):
            for node in self._nodes_at(level):
                for code in node.children:
                    assert 0 <= code < self.mapper.codebook_size, "code out of range"

    def _nodes_at(self, depth: int) -> list[TrieNode]:
        out, stack = [], [self.root]
        while stack:
            cur = stack.pop()
            if cur.depth == depth:
                out.append(cur)
            elif cur.depth < depth:
                stack.extend(cur.children.values())
        return out


class PrefixConstraint:
    """Trie-backed constraint used during autoregressive decoding."""

    def __init__(self, mapper: SemanticIDMapper, trie: SemanticTrie | None = None) -> None:
        self.mapper = mapper
        self.trie = trie if trie is not None else SemanticTrie(mapper)

    def allowed(self, prefix: list[int]) -> list[int]:
        codes = []
        for pos, t in enumerate(prefix):
            dc = self.mapper.decode_token(t)
            if dc is None or dc[0] != pos:
                return []
            codes.append(dc[1])
        return self.trie.allowed_next(codes)

    def is_complete(self, tokens: list[int]) -> bool:
        return self.mapper.sid_from_tokens(tokens) is not None


def sid_to_items(mapper: SemanticIDMapper, sid: tuple) -> list[int]:
    return mapper.sid_to_items.get(tuple(sid), [])
