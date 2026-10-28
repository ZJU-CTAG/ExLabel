import asyncio
import os
import json
import uuid
import re
import logging
import copy
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


def generate_id() -> str:
    """Generate a unique 32-character hexadecimal UUID without hyphens."""
    return uuid.uuid4().hex


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving memories using hybrid search.

    Combines vector similarity (cosine) and text relevance (BM25) for retrieval.
    Provides persistent storage with automatic embedding computation.

    Attributes:
        storage_path: Path to the knowledge base JSON file
        embedder: Text embedding model instance
        memories: List of all memory entries with embeddings
    """

    VECTOR_WEIGHT = 0.5
    TEXT_WEIGHT = 0.5

    def __init__(self, storage_path: str, embedder):
        """
        Initialize the knowledge base.

        Args:
            storage_path: Path to the knowledge base JSON file
            embedder: Text embedder that implements async embed(text: str) method

        Raises:
            ValueError: If embedder is None
        """
        if embedder is None:
            raise ValueError("Embedder is required for KnowledgeBase")
        
        self.storage_path = storage_path
        self.embedder = embedder
        self.memories: List[Dict[str, Any]] = []
        self._write_lock = asyncio.Lock()

        self._bm25_index: Optional[BM25Okapi] = None
        self._tokenized_docs: Optional[List[List[str]]] = None

        if os.path.exists(self.storage_path):
            self.load()
        self._validate_ids()

    def _validate_ids(self):
        ids = [m.get("id") for m in self.memories]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(
                "Bank requires nonempty unique memory IDs; rebuild rather than "
                "guessing which duplicate to retain"
            )

    @staticmethod
    def _validate_fields(key, value):
        if {"id", "embedding"} & (set(key or {}) | set(value)):
            raise ValueError("Memory IDs and embeddings are managed by the bank")
        if set(key or {}) & set(value):
            raise ValueError("Memory keys and content fields must not overlap")

    async def _new_entry(self, memory_id, key, value):
        self._validate_fields(key, value)
        embedding = np.asarray(
            await self.embedder.embed(self.serialize_memory_key(key)), dtype=float
        ).reshape(-1)
        if not embedding.size or not np.isfinite(embedding).all():
            raise ValueError("Invalid embedding")
        return {
            "id": memory_id, **copy.deepcopy(key), **copy.deepcopy(value),
            "embedding": embedding.tolist(),
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the knowledge base to a JSON file.

        Args:
            path: Save path (defaults to storage_path)
            
        Raises:
            Exception: If file write operation fails
        """
        self._validate_ids()
        save_path = path or self.storage_path
        
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = {"memories": self.memories}

        tmp_path = f"{save_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, save_path)
            logging.info("Knowledge base saved to %s", save_path)
        except Exception as e:
            logging.exception("Failed to save knowledge base: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self) -> None:
        """
        Load the knowledge base from file.
        
        Raises:
            Exception: If file read or JSON parsing fails
        """
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.memories = data.get("memories", [])            
            self._invalidate_cache()
            
            logging.info(
                "Loaded knowledge base from %s (total=%d memories)",
                self.storage_path,
                len(self.memories)
            )
        except Exception as e:
            logging.exception("Failed to load knowledge base: %s", e)
            raise

    def clear(self) -> None:
        """Clear all memories from the knowledge base."""
        self.memories.clear()
        self._invalidate_cache()
        logging.info("Knowledge base cleared")

    def _invalidate_cache(self) -> None:
        """Invalidate BM25 index and tokenized docs cache."""
        self._bm25_index = None
        self._tokenized_docs = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def serialize_memory_key(self, key: Dict[str, Any]) -> str:
        """
        Serialize structured key dictionary into a single-line string.

        Args:
            key: Structured key dictionary

        Returns:
            Serialized string representation of the key
        """
        parts = [f"{k.capitalize()}: {v}" for k, v in key.items()]
        serialized = " | ".join(p.replace("\n", " ") for p in parts)
        serialized = re.sub(r"\s+", " ", serialized).strip()
        return serialized

    async def add_memory(
        self,
        key: Dict[str, Any],
        value: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Add a new memory entry.

        Args:
            key: Memory key (used for retrieval)
            value: Memory value (additional information)

        Returns:
            (success flag, memory ID or None)
        """
        async with self._write_lock:
            memory_id = generate_id()
            entry = await self._new_entry(memory_id, key, value)
            self.memories.append(entry)
            self._invalidate_cache()
            return True, memory_id

    async def update_memory(
        self, 
        memory_id: str, 
        value: Dict[str, Any], 
        key: Dict[str, Any] = None
    ) -> bool:
        """
        Update an existing memory entry.

        Args:
            memory_id: Memory ID to update
            value: Values to update (perspective, experience, mistakes)
            key: Optional new key to update (triggers embedding recalculation)

        Returns:
            Whether update was successful
        """
        self._validate_fields(key, value)
        async with self._write_lock:
            index = next((i for i, m in enumerate(self.memories) if m["id"] == memory_id), None)
            if index is None:
                # Preserve the original explicit-key upsert behavior.
                if key is None:
                    return False
                entry = await self._new_entry(generate_id(), key, value)
                self.memories.append(entry)
            elif key is not None:
                # Compute completely before publishing; replace, never append.
                entry = await self._new_entry(memory_id, key, value)
                self.memories[index] = entry
            else:
                entry = copy.deepcopy(self.memories[index])
                for field in ("perspective", "experience", "mistakes"):
                    if field in value:
                        entry[field] = copy.deepcopy(value[field])
                    else:
                        entry.pop(field, None)
                self.memories[index] = entry
            self._invalidate_cache()
            return True

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Whether deletion was successful
        """
        async with self._write_lock:
            for i, entry in enumerate(self.memories):
                if entry.get("id") == memory_id:
                    self.memories.pop(i)
                    self._invalidate_cache()
                    return True

            logging.warning("Memory with id=%s not found", memory_id)
            return False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _build_bm25_index(self, documents: List[str]) -> BM25Okapi:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document strings

        Returns:
            BM25Okapi index
        """
        self._tokenized_docs = [self._tokenize(doc) for doc in documents]
        return BM25Okapi(self._tokenized_docs)

    async def retrieve_memory(
        self,
        query: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using hybrid search.

        Combines cosine similarity (vector search) and BM25 (text search)
        with configurable weights for optimal retrieval performance.

        Args:
            query: Structured query dictionary
            top_k: Number of top candidates to return

        Returns:
            List of candidate memories with '_score' field indicating relevance
        """
        query_key = [k for k, v in query.items()]
        query_text = self.serialize_memory_key(query)
        
        query_embedding = await self.embedder.embed(query_text)
        query_vector = np.array(query_embedding, dtype=float).reshape(1, -1)
        
        if not self.memories:
            return []

        memory_vectors = []
        
        for mem in self.memories:
            emb = mem.get("embedding")
            if emb:
                memory_vectors.append(np.array(emb, dtype=float).reshape(1, -1))
            else:
                memory_vectors.append(np.zeros_like(query_vector))
        
        vector_scores = []
        for mem_vec in memory_vectors:
            sim = cosine_similarity(query_vector, mem_vec)[0][0]
            vector_scores.append(max(0, sim))
        
        if self._bm25_index is None:
            memory_texts = [
                self.serialize_memory_key(
                    {
                        k: mem.get(k, "") for k in query_key
                    }
                ) for mem in self.memories
            ]
            self._bm25_index = self._build_bm25_index(memory_texts)
        
        query_tokens = self._tokenize(query_text)
        bm25_scores = self._bm25_index.get_scores(query_tokens)

        max_vector_score = max(vector_scores) if max(vector_scores) > 0 else 1.0
        max_bm25_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        
        normalized_vector_scores = [s / max_vector_score for s in vector_scores]
        normalized_bm25_scores = [s / max_bm25_score for s in bm25_scores]
        
        candidates = []
        for i, mem in enumerate(self.memories):
            hybrid_score = (
                self.VECTOR_WEIGHT * normalized_vector_scores[i] +
                self.TEXT_WEIGHT * normalized_bm25_scores[i]
            )
            
            candidate = copy.deepcopy(mem)
            candidate["_score"] = hybrid_score
            candidate["_vector_score"] = vector_scores[i]
            candidate["_bm25_score"] = bm25_scores[i]
            candidates.append(candidate)
        
        candidates.sort(key=lambda x: x["_score"], reverse=True)
        
        selected_candidates = []
        seen_ids = set()
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id not in seen_ids:
                selected_candidates.append(candidate)
                seen_ids.add(candidate_id)
                if len(selected_candidates) >= top_k:
                    break
        
        return selected_candidates

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary containing statistics
        """
        return {
            "total_memories": len(self.memories),
            "storage_path": self.storage_path
        }
