from __future__ import annotations

import json
from pathlib import Path

import chromadb
import numpy as np

from src.models import ChunkRecord, VectorMatch


class IndexStore:
    COLLECTION_NAME = "rag_chunks"
    CHROMA_DIR_NAME = "chroma"

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chroma_dir = index_dir / self.CHROMA_DIR_NAME
        self.metadata_path = index_dir / "metadata.json"
        self.legacy_chunks_path = index_dir / "chunks.jsonl"
        self.legacy_embeddings_path = index_dir / "embeddings.npy"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

    def save(self, chunks: list[ChunkRecord], embeddings: np.ndarray, metadata: dict) -> None:
        collection = self._reset_collection()
        self._cleanup_legacy_files()

        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            if not batch_chunks:
                continue
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                embeddings=batch_embeddings.tolist(),
                metadatas=[self._metadata_from_chunk(chunk) for chunk in batch_chunks],
            )

        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_chunks(self) -> list[ChunkRecord]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        payload = collection.get(include=["documents", "metadatas"])
        rows = list(
            zip(
                payload.get("ids", []),
                payload.get("documents", []),
                payload.get("metadatas", []),
            )
        )
        rows.sort(key=lambda row: row[0])
        return [self._record_from_payload(chunk_id, document, metadata) for chunk_id, document, metadata in rows]

    def load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def query_similar(self, query_embedding: np.ndarray, n_results: int) -> list[VectorMatch]:
        collection = self._get_collection()
        if n_results <= 0 or collection.count() == 0:
            return []

        payload = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            record = self._record_from_payload(chunk_id, document, metadata)
            matches.append(
                VectorMatch(
                    record=record,
                    score=self._distance_to_similarity(distance),
                )
            )
        return matches

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def _reset_collection(self):
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        return self._get_collection()

    def _metadata_from_chunk(self, chunk: ChunkRecord) -> dict:
        return {
            "source_path": chunk.source_path,
            "source_name": chunk.source_name,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page if chunk.page is not None else -1,
        }

    def _record_from_payload(self, chunk_id: str, document: str, metadata: dict | None) -> ChunkRecord:
        payload = metadata or {}
        page = payload.get("page", -1)
        return ChunkRecord(
            chunk_id=chunk_id,
            source_path=str(payload.get("source_path", "")),
            source_name=str(payload.get("source_name", "")),
            text=document,
            chunk_index=int(payload.get("chunk_index", 0)),
            page=None if page in (-1, None) else int(page),
        )

    def _distance_to_similarity(self, distance: float | int | None) -> float:
        if distance is None:
            return 0.0
        similarity = 1.0 - float(distance)
        return max(-1.0, min(1.0, similarity))

    def _cleanup_legacy_files(self) -> None:
        for path in (self.legacy_chunks_path, self.legacy_embeddings_path):
            if path.exists():
                path.unlink()
