from __future__ import annotations

import json
from typing import Any

import requests

from src.config import Settings
from src.models import QueryRewrite


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rewrite_query(self, query: str, memory_text: str) -> QueryRewrite:
        messages = [
            {
                "role": "developer",
                "content": (
                    "You rewrite user questions for retrieval in a RAG pipeline. "
                    "Use the memory notes as persistent behavior rules. "
                    "Return only valid JSON with keys: rewritten_query, retrieval_query, reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Memory notes:\n{memory_text}\n\n"
                    f"Original user question:\n{query}\n\n"
                    "Rewrite the question so retrieval becomes more precise. "
                    "Keep military terminology, equipment names, dates, and versions if present."
                ),
            },
        ]

        try:
            content = self.chat(
                messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            payload = _extract_json(content)
            rewritten_query = str(payload.get("rewritten_query", query)).strip() or query
            retrieval_query = str(payload.get("retrieval_query", rewritten_query)).strip() or rewritten_query
            reasoning = str(payload.get("reasoning", "Fallback rewrite used.")).strip()
            return QueryRewrite(
                original_query=query,
                rewritten_query=rewritten_query,
                retrieval_query=retrieval_query,
                reasoning=reasoning,
            )
        except Exception:
            fallback = query.strip()
            return QueryRewrite(
                original_query=query,
                rewritten_query=fallback,
                retrieval_query=fallback,
                reasoning="The rewrite model was unavailable, so the original query was used.",
            )

    def generate_answer(
        self,
        original_query: str,
        rewritten_query: str,
        memory_text: str,
        context_blocks: list[str],
    ) -> str:
        context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a military documentation assistant inside a RAG system. "
                    "Follow the memory notes carefully. "
                    "Answer only from the retrieved context. "
                    "If the context is insufficient, say what is missing. "
                    "Do not claim facts that are not supported by the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Memory notes:\n{memory_text}\n\n"
                    f"Original user question:\n{original_query}\n\n"
                    f"Rewritten query:\n{rewritten_query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Write the answer in English. "
                    "If multiple equipment versions are mentioned, prefer the newest version supported by the context. "
                    "Reference the chunk numbers in square brackets when you rely on them."
                ),
            },
        ]
        return self.chat(messages, temperature=0.2)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
    ) -> str:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before sending questions.")

        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(_build_openai_error(response)) from exc

        response_payload = response.json()
        content = _extract_text_content(response_payload)
        if not content:
            raise RuntimeError("OpenAI chat response did not include text content.")
        return content.strip()


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _extract_text_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _build_openai_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    detail = str(payload.get("error", {}).get("message", "")).strip()
    if detail:
        return f"OpenAI chat request failed: {detail}"
    return f"OpenAI chat request failed with status {response.status_code}."
