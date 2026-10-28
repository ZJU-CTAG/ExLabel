"""
Single-mode retrievers for the SimpleRAG baseline.

Unlike the main pipeline's hybrid search, each SimpleRAG run picks exactly one
retrieval mode:
- dense: cosine similarity between query embedding and stored key embeddings
- bm25:  lexical BM25 matching over serialized key texts

Both read from the same cached index (a KnowledgeBase JSON file); entries are
returned sorted by relevance without any score fusion.
"""

import re
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from knowledge import KnowledgeBase

# Fields that make up a stored sample's retrieval key text. SimpleRAG keys are
# built from raw task input (see baseline.rag_utils.INPUT_KEY_FIELDS), so these
# are exactly the possible key components.
KEY_TEXT_FIELDS = ("problem_statement", "test_patch")

# Character cap for serialized key text. Test-task keys embed test_patch, which
# can reach ~200k chars and exceed the embedding model's max_model_len (40960
# tokens) -> vLLM rejects such inputs with HTTP 400. Capping here keeps every
# key safely under the token limit; both the index build and query-time
# retrieval call this function, so index/query keys stay truncated identically.
MAX_KEY_CHARS = 30000


def serialize_input_key(key: Dict[str, Any]) -> str:
    """
    Serialize a raw-input key dict into a single-line text.

    Mirrors KnowledgeBase.serialize_memory_key so that queries are embedded /
    tokenized consistently with what was stored at index build time.
    """
    parts = [f"{k.capitalize()}: {v}" for k, v in key.items()]
    serialized = " | ".join(p.replace("\n", " ") for p in parts)
    serialized = re.sub(r"\s+", " ", serialized).strip()
    return serialized[:MAX_KEY_CHARS]


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 (same scheme as KnowledgeBase._tokenize)."""
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


def _clean(entry: Dict[str, Any], score_field: str, score: float) -> Dict[str, Any]:
    entry["_score"] = score
    entry[f"_{score_field}"] = score
    return entry


class DenseRetriever:
    """
    Vector retrieval over the SimpleRAG index.

    Embeds the serialized query and ranks stored samples by cosine similarity.
    """

    def __init__(self, kb: KnowledgeBase, embedder):
        if embedder is None:
            raise ValueError("Embedder is required for DenseRetriever")
        self.kb = kb
        self.embedder = embedder

    async def retrieve(self, key: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.kb.memories:
            return []

        query_text = serialize_input_key(key)
        query_embedding = await self.embedder.embed(query_text)
        query_vector = np.array(query_embedding, dtype=float).reshape(1, -1)

        candidates = []
        for mem in self.kb.memories:
            emb = mem.get("embedding")
            if emb:
                mem_vector = np.array(emb, dtype=float).reshape(1, -1)
                denom = np.linalg.norm(query_vector[0]) * np.linalg.norm(mem_vector[0]) + 1e-12
                score = float(max(0.0, float(np.dot(query_vector[0], mem_vector[0])) / denom))
            else:
                score = 0.0
            candidates.append(_clean(dict(mem), "dense_score", score))

        candidates.sort(key=lambda x: x["_score"], reverse=True)
        return candidates[:top_k]


class BM25Retriever:
    """
    Lexical retrieval over the SimpleRAG index.

    Builds a BM25 index over the serialized key texts of all stored samples
    (lazily, on first retrieval) and scores the tokenized query against it.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._bm25_index = None

    def _memory_text(self, mem: Dict[str, Any]) -> str:
        """Serialize a stored sample's key portion into its BM25 document."""
        key = {k: mem.get(k, "") for k in KEY_TEXT_FIELDS if k in mem}
        return serialize_input_key(key)

    def _ensure_index(self):
        if self._bm25_index is not None:
            return
        documents = [self._memory_text(mem) for mem in self.kb.memories]
        self._bm25_index = BM25Okapi([_tokenize(doc) for doc in documents])

    async def retrieve(self, key: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.kb.memories:
            return []
        self._ensure_index()

        scores = self._bm25_index.get_scores(_tokenize(serialize_input_key(key)))

        candidates = [
            _clean(dict(mem), "bm25_score", float(score))
            for mem, score in zip(self.kb.memories, scores)
        ]
        candidates.sort(key=lambda x: x["_score"], reverse=True)
        return candidates[:top_k]


def get_retriever(retrieval_mode: str, kb: KnowledgeBase, embedder=None):
    """
    Factory for SimpleRAG retrievers.

    Args:
        retrieval_mode: 'dense' or 'bm25'
        kb: Loaded SimpleRAG index
        embedder: Embedder instance (required for 'dense')

    Returns:
        A retriever exposing `async retrieve(key_dict, top_k)`
    """
    if retrieval_mode == "dense":
        return DenseRetriever(kb, embedder)
    elif retrieval_mode == "bm25":
        return BM25Retriever(kb)
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}. Available: dense, bm25")
