"""Claude-powered question answering over retrieved context."""

SYSTEM_PROMPT = """Answer using only the supplied context. Cite sources as [source, page/chunk].
If the context does not contain the answer, say that you do not know."""


def answer_question(question: str, context: list[dict]) -> dict[str, object]:
    """Generate a cited answer with the Claude API."""
    raise NotImplementedError("Claude API integration is not configured yet")
