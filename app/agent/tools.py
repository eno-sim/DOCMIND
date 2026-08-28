"""Safe, small tools exposed to the DocMind agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from app.backend import KnowledgeBase

TOOLS = [
    {"name": "semantic_search", "description": "Search local documents by meaning.", "input": {"query": "string", "top_k": "integer 1-10"}},
    {"name": "keyword_search", "description": "Search local documents by exact keywords using BM25.", "input": {"query": "string", "top_k": "integer 1-10"}},
    {"name": "read_document", "description": "Read a specific local document or one chunk in full.", "input": {"source": "document filename or path", "chunk_id": "optional exact chunk id"}},
    {"name": "web_search", "description": "Search Wikipedia for information unavailable in local documents. Use only when local evidence is insufficient.", "input": {"query": "string"}},
    {"name": "decompose", "description": "Split a complex question into focused sub-questions.", "input": {"questions": "list of strings"}},
    {"name": "synthesize", "description": "Return a final answer using collected evidence.", "input": {"answer": "string"}},
    {"name": "clarify", "description": "Ask the user a clarification question when the request is ambiguous.", "input": {"question": "string"}},
]


def compact(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": r.get("chunk_id", r.get("id")), "source": r.get("source"), "text": str(r.get("text", ""))[:1800], "metadata": r.get("metadata", {})} for r in results]


class AgentTools:
    def __init__(self, kb: KnowledgeBase, web_enabled: bool = True) -> None:
        self.kb, self.web_enabled = kb, web_enabled

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.kb.vector is None:
            return []
        return compact(self.kb.vector.search(query, top_k=max(1, min(top_k, 10))))

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return compact(self.kb.sparse.search(query, top_k=max(1, min(top_k, 10))))

    def read_document(self, source: str, chunk_id: str | None = None) -> list[dict[str, Any]]:
        if chunk_id:
            for chunk in self.kb.sparse.chunks:
                if chunk.chunk_id == chunk_id:
                    return compact([{"id": chunk.chunk_id, "chunk_id": chunk.chunk_id, "source": chunk.source, "text": chunk.text, "metadata": chunk.metadata}])
        requested = Path(source).name
        for document in self.kb.documents.values():
            path = Path(str(document["source"]))
            if path.name == requested or str(path) == source:
                return [{"source": str(path), "text": path.read_text(encoding="utf-8", errors="replace")[:12000]}]
        return []

    def web_search(self, query: str) -> list[dict[str, Any]]:
        if not self.web_enabled:
            return [{"error": "Web search is disabled by DOCMIND_WEB_SEARCH_ENABLED."}]
        try:
            url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + quote(query) + "&format=json&srlimit=3"
            with urlopen(url, timeout=8) as response:  # nosec B310: fixed HTTPS endpoint
                rows = json.loads(response.read().decode("utf-8"))["query"]["search"]
            return [{"source": "Wikipedia", "title": row["title"], "text": row["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")} for row in rows]
        except Exception as exc:
            return [{"error": f"web search failed: {exc}"}]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "semantic_search": return self.semantic_search(**arguments)
        if name == "keyword_search": return self.keyword_search(**arguments)
        if name == "read_document": return self.read_document(**arguments)
        if name == "web_search": return self.web_search(**arguments)
        if name in {"decompose", "synthesize", "clarify"}: return arguments
        return {"error": f"Unknown tool: {name}"}
