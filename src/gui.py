from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

from src.config import Settings
from src.models import QueryRewrite, RetrievedChunk
from src.rag_pipeline import RAGPipeline


class RAGDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Military Documentation RAG")
        self.root.geometry("1080x760")
        self.root.minsize(900, 650)

        self.settings = Settings()
        self.pipeline = RAGPipeline(self.settings)
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()
        self._show_welcome()

    def _build_layout(self) -> None:
        self.root.configure(bg="#e6e2d3")

        header = tk.Frame(self.root, bg="#1f3b4d", padx=16, pady=12)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="Military Documentation RAG",
            font=("Georgia", 20, "bold"),
            fg="#f6f1e9",
            bg="#1f3b4d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text=f"LLM: {self.settings.active_model_name} | Embeddings: {self.settings.embedding_model}",
            font=("Georgia", 10),
            fg="#d9e2ec",
            bg="#1f3b4d",
        )
        subtitle.pack(anchor="w")

        container = tk.Frame(self.root, bg="#e6e2d3", padx=16, pady=16)
        container.pack(fill="both", expand=True)

        input_label = tk.Label(
            container,
            text="Question or Note",
            font=("Georgia", 12, "bold"),
            bg="#e6e2d3",
            fg="#1d2d35",
        )
        input_label.pack(anchor="w")

        self.input_box = scrolledtext.ScrolledText(
            container,
            height=5,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#fffdf7",
            fg="#1d2d35",
            insertbackground="#1d2d35",
        )
        self.input_box.pack(fill="x", pady=(6, 12))

        button_row = tk.Frame(container, bg="#e6e2d3")
        button_row.pack(fill="x", pady=(0, 12))

        self.ask_button = self._build_button(button_row, "Ask", self.ask_question)
        self.ask_button.pack(side="left", padx=(0, 8))

        self.reindex_button = self._build_button(button_row, "Reindex Documents", self.reindex_documents)
        self.reindex_button.pack(side="left", padx=(0, 8))

        self.remember_button = self._build_button(button_row, "Remember Note", self.remember_note)
        self.remember_button.pack(side="left", padx=(0, 8))

        self.clear_button = self._build_button(button_row, "Clear", self.clear_output)
        self.clear_button.pack(side="left")

        output_label = tk.Label(
            container,
            text="Response",
            font=("Georgia", 12, "bold"),
            bg="#e6e2d3",
            fg="#1d2d35",
        )
        output_label.pack(anchor="w")

        self.output_box = scrolledtext.ScrolledText(
            container,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#f7f4ec",
            fg="#1d2d35",
            state="disabled",
        )
        self.output_box.pack(fill="both", expand=True)

        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padx=12,
            pady=6,
            bg="#8d6e63",
            fg="#fffaf3",
            font=("Georgia", 10),
        )
        status_bar.pack(fill="x", side="bottom")

    def _build_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Georgia", 10, "bold"),
            bg="#b14f3a",
            fg="#fffaf3",
            activebackground="#8f3d2c",
            activeforeground="#fffaf3",
            relief=tk.FLAT,
            padx=12,
            pady=8,
            cursor="hand2",
        )

    def _show_welcome(self) -> None:
        metadata = self.pipeline.metadata
        index_line = (
            f"Indexed chunks: {metadata.get('chunks_indexed', 0)} "
            f"| Documents: {metadata.get('documents_loaded', 0)} "
            f"| Vector DB: {metadata.get('vector_db', 'Chroma')}"
        )
        welcome_text = (
            "This desktop app runs a full RAG pipeline for military documentation.\n\n"
            "Workflow:\n"
            "1. Put files into data/raw\n"
            "2. Click Reindex Documents\n"
            "3. Ask a question\n"
            "4. Review the rewritten query, answer, and retrieved sources\n\n"
            "To save a persistent rule, type /remember your rule here or use the Remember Note button.\n\n"
            f"{index_line}"
        )
        self._set_output(welcome_text)

    def ask_question(self) -> None:
        query = self._get_input()
        self._run_background_task("Running prompt rewrite, retrieval, and answer generation...", self._answer_task, query)

    def reindex_documents(self) -> None:
        self._run_background_task("Reindexing documents and rebuilding embeddings...", self._reindex_task)

    def remember_note(self) -> None:
        note = self._get_input()
        if not note.strip():
            messagebox.showwarning("Missing note", "Enter a note before saving it.")
            return
        added = self.pipeline.save_note(note)
        message = "Note saved to memory." if added else "That note already exists or was empty."
        self.status_var.set(message)
        self._append_output(f"\n\nMemory update:\n{message}")

    def clear_output(self) -> None:
        self._set_output("")
        self.status_var.set("Output cleared.")

    def _answer_task(self, query: str) -> None:
        result = self.pipeline.answer_query(query)
        if result["mode"] == "memory":
            self.root.after(0, lambda: self._finish_memory_action(result["message"]))
            return

        rewrite: QueryRewrite = result["rewrite"]
        retrieved: list[RetrievedChunk] = result["retrieved"]
        formatted = self._format_answer(rewrite, result["answer"], retrieved)
        self.root.after(0, lambda: self._finish_success(formatted, "Answer ready."))

    def _reindex_task(self) -> None:
        metadata = self.pipeline.reindex()
        message = (
            f"Index rebuilt successfully.\n"
            f"Documents loaded: {metadata['documents_loaded']}\n"
            f"Chunks indexed: {metadata['chunks_indexed']}\n"
            f"Vector DB: {metadata.get('vector_db', 'Chroma')}\n"
            f"Embedding model: {metadata['embedding_model']}\n"
            f"Built at: {metadata['built_at']}"
        )
        self.root.after(0, lambda: self._finish_success(message, "Reindex complete."))

    def _run_background_task(self, status_message: str, target, *args) -> None:
        self._set_busy(True)
        self.status_var.set(status_message)

        def runner() -> None:
            try:
                target(*args)
            except Exception as exc:
                self.root.after(0, lambda: self._finish_error(str(exc)))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def _finish_success(self, output_text: str, status_message: str) -> None:
        self._set_output(output_text)
        self.status_var.set(status_message)
        self._set_busy(False)

    def _finish_error(self, error_message: str) -> None:
        self.status_var.set("The last action failed.")
        self._append_output(f"\n\nError:\n{error_message}")
        self._set_busy(False)

    def _finish_memory_action(self, message: str) -> None:
        self.status_var.set(message)
        self._append_output(f"\n\nMemory update:\n{message}")
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.ask_button.configure(state=state)
        self.reindex_button.configure(state=state)
        self.remember_button.configure(state=state)

    def _get_input(self) -> str:
        return self.input_box.get("1.0", tk.END).strip()

    def _set_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, text)
        self.output_box.configure(state="disabled")

    def _append_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.insert(tk.END, text)
        self.output_box.see(tk.END)
        self.output_box.configure(state="disabled")

    def _format_answer(self, rewrite: QueryRewrite, answer: str, retrieved: list[RetrievedChunk]) -> str:
        source_lines = []
        for index, item in enumerate(retrieved, start=1):
            page_label = f", page {item.record.page}" if item.record.page else ""
            source_lines.append(
                f"[{index}] {item.record.source_name}{page_label} | "
                f"combined={item.combined_score:.3f}, vector={item.vector_score:.3f}, bm25={item.bm25_score:.3f}"
            )
        sources_text = "\n".join(source_lines) if source_lines else "No sources retrieved."
        return (
            f"Original question:\n{rewrite.original_query}\n\n"
            f"Rewritten query:\n{rewrite.rewritten_query}\n\n"
            f"Retrieval query:\n{rewrite.retrieval_query}\n\n"
            f"Rewrite reasoning:\n{rewrite.reasoning}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Retrieved sources:\n{sources_text}"
        )


def launch_app() -> None:
    root = tk.Tk()
    RAGDesktopApp(root)
    root.mainloop()
