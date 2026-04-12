from __future__ import annotations

import numpy as np
import requests

from src.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model_name = settings.embedding_model
        self.batch_size = 96

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if batch:
                batches.append(self._create_embeddings(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._create_embeddings([text])[0]

    def _create_embeddings(self, inputs: list[str]) -> np.ndarray:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before reindexing documents.")

        url = f"{self.base_url}/embeddings"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": inputs,
                "encoding_format": "float",
            },
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(_build_openai_error(response, "embedding")) from exc

        payload = response.json()
        data = payload.get("data", [])
        if len(data) != len(inputs):
            raise RuntimeError("OpenAI embedding response did not match the number of requested inputs.")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = np.array([item["embedding"] for item in ordered], dtype=np.float32)
        return _l2_normalize(vectors)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _build_openai_error(response: requests.Response, action: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    detail = str(payload.get("error", {}).get("message", "")).strip()
    if detail:
        return f"OpenAI {action} request failed: {detail}"
    return f"OpenAI {action} request failed with status {response.status_code}."
