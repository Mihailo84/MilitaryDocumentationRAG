from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.embedding_service import EmbeddingService
from src.index_store import IndexStore
from src.models import ChunkRecord, RetrievedChunk


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_value = float(values.max())
    min_value = float(values.min())
    if max_value == min_value:
        if max_value == 0:
            return np.zeros_like(values, dtype=np.float32)
        return np.ones_like(values, dtype=np.float32)
    normalized = (values - min_value) / (max_value - min_value)
    return normalized.astype(np.float32)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        vector_store: IndexStore,
        embedding_service: EmbeddingService,
        bm25_weight: float,
        vector_weight: float,
    ) -> None:
        self.chunks = chunks
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if chunks else None
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or self.bm25 is None:
            return []

        query_embedding = self.embedding_service.encode_query(query)
        candidate_count = min(len(self.chunks), max(top_k * 4, top_k, 12))
        vector_matches = self.vector_store.query_similar(query_embedding, n_results=candidate_count)
        vector_scores_by_id = {match.record.chunk_id: match.score for match in vector_matches}
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
        candidate_ids: list[str] = []
        seen_ids: set[str] = set()

        for match in vector_matches:
            chunk_id = match.record.chunk_id
            if chunk_id not in seen_ids:
                candidate_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        for index in bm25_top_indices:
            chunk_id = self.chunks[index].chunk_id
            if chunk_id not in seen_ids:
                candidate_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        if not candidate_ids:
            return []

        vector_values = np.array([vector_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_ids], dtype=np.float32)
        bm25_values = np.array(
            [bm25_scores[self.chunk_index_by_id[chunk_id]] for chunk_id in candidate_ids],
            dtype=np.float32,
        )

        vector_norm = normalize_scores(vector_values)
        bm25_norm = normalize_scores(bm25_values)
        combined = (vector_norm * self.vector_weight) + (bm25_norm * self.bm25_weight)

        top_indices = np.argsort(combined)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for index in top_indices:
            chunk_id = candidate_ids[index]
            results.append(
                RetrievedChunk(
                    record=self.chunk_by_id[chunk_id],
                    vector_score=float(vector_values[index]),
                    bm25_score=float(bm25_values[index]),
                    combined_score=float(combined[index]),
                )
            )
        return results
