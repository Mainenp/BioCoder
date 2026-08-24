from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.services.file_content import extract_file
from app.services.llm import create_embeddings

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".json", ".docx"}


class KnowledgeStore:
    """Rebuildable local vector index backed by files on disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vector_store: InMemoryVectorStore | None = None
        self._documents: list[Document] = []
        self._files: list[str] = []
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        return self._vector_store is not None

    def _iter_files(self) -> list[Path]:
        paths: list[Path] = []
        excluded = {
            name.strip().casefold()
            for name in self.settings.knowledge_exclude_files.split(",")
            if name.strip()
        }
        for folder in (self.settings.knowledge_dir, self.settings.uploads_dir):
            folder.mkdir(parents=True, exist_ok=True)
            paths.extend(
                path
                for path in folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and path.name.casefold() not in excluded
            )
        return sorted(paths)

    @staticmethod
    def _read_file(path: Path) -> str:
        return extract_file(path.name, path.read_bytes()).text

    def rebuild(self) -> dict[str, Any]:
        files = self._iter_files()
        raw_documents: list[Document] = []
        for path in files:
            text = self._read_file(path).strip()
            if not text:
                continue
            raw_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "path": str(path),
                        "source_type": "local_knowledge",
                    },
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=140,
            separators=["\n## ", "\n### ", "\n\n", "。", ". ", " "],
        )
        chunks = splitter.split_documents(raw_documents)
        if not chunks:
            with self._lock:
                self._vector_store = None
                self._documents = []
                self._files = []
            return self.status()

        store = InMemoryVectorStore(create_embeddings(self.settings))
        store.add_documents(chunks)
        with self._lock:
            self._vector_store = store
            self._documents = chunks
            self._files = [path.name for path in files]
        return self.status()

    def ensure_ready(self) -> None:
        if not self.ready:
            self.rebuild()

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._lock:
            store = self._vector_store
        if store is None:
            return []
        results = store.similarity_search_with_score(query, k=max(1, min(k, 10)))
        return [
            {
                "title": doc.metadata.get("source", "本地知识库"),
                "url": None,
                "source_type": doc.metadata.get("source_type", "local_knowledge"),
                "snippet": doc.page_content[:1000],
                "metadata": {**doc.metadata, "score": round(float(score), 4)},
            }
            for doc, score in results
        ]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self.ready,
                "documents": len(set(self._files)),
                "chunks": len(self._documents),
                "files": list(self._files),
            }
