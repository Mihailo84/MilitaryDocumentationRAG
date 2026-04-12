from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SourceDocument:
    source_path: str
    source_name: str
    text: str
    page: Optional[int] = None


@dataclass
class ChunkRecord:
    chunk_id: str
    source_path: str
    source_name: str
    text: str
    chunk_index: int
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ChunkRecord":
        return cls(**payload)


@dataclass
class QueryRewrite:
    original_query: str
    rewritten_query: str
    retrieval_query: str
    reasoning: str


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    vector_score: float
    bm25_score: float
    combined_score: float


@dataclass
class VectorMatch:
    record: ChunkRecord
    score: float
