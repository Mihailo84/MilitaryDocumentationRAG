from __future__ import annotations

from pathlib import Path


class MemoryStore:
    def __init__(self, memory_path: Path) -> None:
        self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("# RAG Memory\n\n", encoding="utf-8")

    def load(self) -> str:
        return self.memory_path.read_text(encoding="utf-8")

    def append_note(self, note: str) -> bool:
        cleaned = note.strip().lstrip("-").strip()
        if not cleaned:
            return False
        existing = self.load()
        bullet = f"- {cleaned}"
        if bullet.lower() in existing.lower():
            return False
        with self.memory_path.open("a", encoding="utf-8") as handle:
            if not existing.endswith("\n"):
                handle.write("\n")
            handle.write(f"{bullet}\n")
        return True

