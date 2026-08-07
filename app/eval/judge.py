"""LLM-as-a-judge evaluation pipeline."""


def evaluate_answer(question: str, answer: str, reference: str, context: list[dict]) -> dict[str, object]:
    """Score correctness, relevance, and faithfulness with Claude."""
    raise NotImplementedError("Evaluation pipeline is not configured yet")
