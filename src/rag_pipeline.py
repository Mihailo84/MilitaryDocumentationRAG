from __future__ import annotations

from datetime import datetime

from src.chunking import chunk_text
from src.config import INDEX_DIR, MEMORY_FILE, RAW_DATA_DIR, Settings
from src.document_loader import load_documents
from src.embedding_service import EmbeddingService
from src.index_store import IndexStore
from src.llm_client import LLMClient
from src.memory import MemoryStore
from src.models import ChunkRecord, RetrievedChunk
from src.retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.index_store = IndexStore(INDEX_DIR)
        self.memory_store = MemoryStore(MEMORY_FILE)
        self.embedding_service = EmbeddingService(self.settings)
        self.llm_client = LLMClient(self.settings)
        self.retriever: HybridRetriever | None = None
        self.metadata: dict = {}
        self.load_index()

    def load_index(self) -> None:
        chunks = self.index_store.load_chunks()
        self.metadata = self.index_store.load_metadata()
        if chunks:
            self.retriever = HybridRetriever(
                chunks=chunks,
                vector_store=self.index_store,
                embedding_service=self.embedding_service,
                bm25_weight=self.settings.bm25_weight,
                vector_weight=self.settings.vector_weight,
            )
        else:
            self.retriever = None

    def reindex(self) -> dict:
        documents = load_documents(RAW_DATA_DIR)
        chunk_records: list[ChunkRecord] = []
        for document in documents:
            chunks = chunk_text(
                text=document.text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for chunk_index, chunk in enumerate(chunks):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=self._build_chunk_id(document.source_name, document.page, chunk_index),
                        source_path=document.source_path,
                        source_name=document.source_name,
                        text=chunk,
                        chunk_index=chunk_index,
                        page=document.page,
                    )
                )

        embeddings = self.embedding_service.encode_documents([chunk.text for chunk in chunk_records])
        metadata = {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "documents_loaded": len(documents),
            "chunks_indexed": len(chunk_records),
            "embedding_model": self.settings.embedding_model,
            "vector_db": "Chroma",
        }
        self.index_store.save(chunk_records, embeddings, metadata)
        self.load_index()
        return metadata

    def answer_query(self, query: str) -> dict:
        if not query.strip():
            raise ValueError("Please enter a question.")
        if query.strip().lower().startswith("/remember "):
            note = query.strip()[10:]
            added = self.memory_store.append_note(note)
            return {
                "mode": "memory",
                "message": "Note saved to memory." if added else "That note already exists or was empty.",
            }
        if self.retriever is None:
            raise RuntimeError("No index found. Add documents to data/raw and click Reindex Documents.")

        memory_text = self.memory_store.load()
        rewrite = self.llm_client.rewrite_query(query, memory_text)
        retrieved = self.retriever.retrieve(rewrite.retrieval_query, top_k=self.settings.top_k)
        context_blocks = self._format_context_blocks(retrieved)
        answer = self.llm_client.generate_answer(
            original_query=query,
            rewritten_query=rewrite.rewritten_query,
            memory_text=memory_text,
            context_blocks=context_blocks,
        )
        return {
            "mode": "answer",
            "rewrite": rewrite,
            "retrieved": retrieved,
            "answer": answer,
        }

    def save_note(self, note: str) -> bool:
        return self.memory_store.append_note(note)

    def get_memory(self) -> str:
        return self.memory_store.load()

    def _format_context_blocks(self, retrieved: list[RetrievedChunk]) -> list[str]:
        blocks: list[str] = []
        for index, item in enumerate(retrieved, start=1):
            page_label = f", page {item.record.page}" if item.record.page else ""
            blocks.append(
                f"[Chunk {index}] Source: {item.record.source_name}{page_label}\n"
                f"Text: {item.record.text}"
            )
        return blocks

    @staticmethod
    def _build_chunk_id(source_name: str, page: int | None, chunk_index: int) -> str:
        page_value = page if page is not None else 0
        safe_source = source_name.replace(" ", "_")
        return f"{safe_source}-p{page_value}-c{chunk_index}"
