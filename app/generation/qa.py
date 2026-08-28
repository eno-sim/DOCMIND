"""Question answering over retrieved context using Anthropic Claude.

The generator deliberately accepts retrieval result dictionaries rather than
index-specific objects.  This keeps the boundary between retrieval and
writing small, testable, and easy to replace with another provider later.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

SYSTEM_PROMPT = """You answer questions using only the supplied context.
Every factual claim must be supported by the context. Cite the supporting
chunk immediately after the claim using exactly [source, page/chunk]. For a
text chunk without a page, use its chunk number (for example
[notes.txt, chunk-2]). For audio or video use its timestamp when available.
If the context does not contain the answer, say exactly that you do not know;
do not guess or use outside knowledge. Keep the answer concise."""


def _citation(result: dict[str, Any]) -> str:
    """Create the citation label shown to the model for one result."""
    metadata = result.get("metadata") or {}
    source = str(result.get("source") or metadata.get("source") or "unknown")
    page = metadata.get("page")
    if page is not None:
        location = f"page-{page}"
    elif metadata.get("start_seconds") is not None:
        start = metadata["start_seconds"]
        end = metadata.get("end_seconds", start)
        location = f"{start}s-{end}s"
    else:
        index = metadata.get("chunk_index", "?")
        location = f"chunk-{index}"
    return f"{source}, {location}"


def format_context(context: list[dict[str, Any]]) -> str:
    """Format retrieval results as a numbered, citation-aware prompt."""
    if not isinstance(context, list):
        raise TypeError("context must be a list of dictionaries")
    blocks: list[str] = []
    for number, result in enumerate(context, start=1):
        if not isinstance(result, dict):
            raise TypeError("context items must be dictionaries")
        text = str(result.get("text", "")).strip()
        if not text:
            continue
        blocks.append(f"<chunk id=\"{number}\" citation=\"[{_citation(result)}]\">\n{text}\n</chunk>")
    return "\n\n".join(blocks) or "(No context was retrieved.)"


def _text_from_response(response: Any) -> str:
    """Extract text from an Anthropic response, including simple test doubles."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content", [])
    texts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def answer_question(
    question: str,
    context: list[dict[str, Any]],
    *,
    client: Any | None = None,
    model: str | None = None,
    max_tokens: int = 700,
) -> dict[str, object]:
    """Generate a grounded answer with Claude and return answer metadata.

    ``client`` is injectable so notebooks and tests need not make a network
    request.  When omitted, an :class:`anthropic.Anthropic` client uses
    ``ANTHROPIC_API_KEY`` from the environment.  The result contains the
    answer, citations detected in that answer, the model, and API usage when
    available.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(context, list):
        raise TypeError("context must be a list")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    prompt = (
        "Context:\n" + format_context(context) +
        "\n\nQuestion:\n" + question.strip() +
        "\n\nAnswer with inline citations when the context supports it."
    )
    # Load before reading the model so values in the repository .env are
    # available when uvicorn was started without exporting them first.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is an optional convenience
        pass
    provider = os.getenv("DOCMIND_LLM_PROVIDER", "claude").lower()
    selected_model = model or os.getenv("OLLAMA_MODEL" if provider == "ollama" else "ANTHROPIC_MODEL", "llama3.2" if provider == "ollama" else "claude-3-5-sonnet-latest")
    usage: Any = None
    if client is None and provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        request = Request(base + "/api/chat", data=json.dumps({"model": selected_model, "stream": False, "options": {"num_predict": max_tokens}, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:  # nosec B310: configured Ollama URL
            payload = json.loads(response.read().decode())
        answer = str(payload.get("message", {}).get("content", "")).strip()
        usage = {"input_tokens": payload.get("prompt_eval_count"), "output_tokens": payload.get("eval_count")}
    else:
        if client is None:
            if provider != "claude":
                raise RuntimeError("DOCMIND_LLM_PROVIDER must be claude or ollama")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install anthropic to use Claude generation") from exc
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is required for Claude generation")
            client = anthropic.Anthropic()
        response = client.messages.create(
            model=selected_model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = _text_from_response(response)
        usage = getattr(response, "usage", None)
    if usage is not None and not isinstance(usage, dict):
        usage = {key: getattr(usage, key) for key in ("input_tokens", "output_tokens") if hasattr(usage, key)}
    citations = sorted(set(re.findall(r"\[([^\]]+,\s*(?:page-[^\]]+|chunk-[^\]]+|[^\]]*s-[^\]]*s))\]", answer)))
    return {"answer": answer, "citations": citations, "model": selected_model, "usage": usage}
