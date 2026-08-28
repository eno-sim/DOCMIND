"""Provider-neutral ReAct loop for multi-step document research."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

from app.backend import KnowledgeBase, kb as default_kb
from app.generation.qa import SYSTEM_PROMPT, format_context
from app.agent.tools import AgentTools, TOOLS

PLANNER_PROMPT = """You are DocMind's retrieval planner. Choose exactly one next action from the supplied tools.
Return JSON only, with {"action":"tool_name","arguments":{...},"reason":"short operational reason"}.
Do not reveal private chain-of-thought. Use decompose only for multiple independent facts. Search local documents before web_search. Use synthesize only when evidence answers the request, clarify only when ambiguity prevents searching.
Tools: {tools}
Question: {question}
Scratchpad (actions and observations): {scratchpad}
"""

@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    sources: list[dict[str, Any]]
    retrieved: int
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    clarification_needed: bool = False
    mode: str = "agentic"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _chat(system: str, prompt: str) -> str:
    """Run a plain text turn against either configured Claude or Ollama."""
    _load_env()
    provider = os.getenv("DOCMIND_LLM_PROVIDER", "claude").lower()
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        request = Request(base + "/api/chat", data=json.dumps({"model": model, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:  # nosec B310: user-configured local Ollama endpoint
            return str(json.loads(response.read().decode())["message"]["content"]).strip()
    if provider != "claude":
        raise RuntimeError("DOCMIND_LLM_PROVIDER must be claude or ollama")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required when DOCMIND_LLM_PROVIDER=claude")
    import anthropic
    response = anthropic.Anthropic().messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), max_tokens=900, system=system, messages=[{"role": "user", "content": prompt}])
    return "\n".join(block.text for block in response.content if getattr(block, "text", None)).strip()


def _plan(question: str, scratchpad: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _chat("Return valid JSON only.", PLANNER_PROMPT.format(tools=json.dumps(TOOLS), question=question, scratchpad=json.dumps(scratchpad[-6:])[:12000]))
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("Planner did not return JSON")
    plan = json.loads(match.group())
    if plan.get("action") not in {tool["name"] for tool in TOOLS} or not isinstance(plan.get("arguments"), dict):
        raise ValueError("Planner selected an invalid action")
    return plan


def _final_answer(question: str, evidence: list[dict[str, Any]]) -> tuple[str, list[str]]:
    text = _chat(SYSTEM_PROMPT, "Context:\n" + format_context(evidence) + "\n\nQuestion:\n" + question + "\n\nAnswer using only context. Cite claims inline.")
    citations = sorted(set(re.findall(r"\[([^\]]+?)\]", text)))
    return text or "I do not know.", citations


def answer_agentically(question: str, knowledge_base: KnowledgeBase | None = None, max_iterations: int | None = None) -> AgentResult:
    if not question.strip():
        raise ValueError("question must be non-empty")
    kb = knowledge_base or default_kb
    limit = max_iterations or int(os.getenv("AGENT_MAX_ITERATIONS", "4"))
    tools = AgentTools(kb, web_enabled=os.getenv("DOCMIND_WEB_SEARCH_ENABLED", "false").lower() == "true")
    scratchpad: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    pending_questions = [question]
    for _ in range(limit):
        current = pending_questions.pop(0) if pending_questions else question
        try:
            plan = _plan(current, scratchpad)
        except Exception:
            # Safe fallback retains useful behavior if a local Ollama model
            # produces malformed tool JSON.
            plan = {"action": "semantic_search" if kb.vector else "keyword_search", "arguments": {"query": current, "top_k": 5}, "reason": "fallback local search"}
        action, arguments = plan["action"], plan["arguments"]
        fingerprint = action + ":" + json.dumps(arguments, sort_keys=True)
        if fingerprint in seen and action not in {"synthesize", "clarify"}:
            scratchpad.append({"action": action, "observation": "skipped repeated action"})
            continue
        seen.add(fingerprint)
        if action == "decompose":
            questions = [str(q) for q in arguments.get("questions", []) if str(q).strip()]
            pending_questions.extend(q for q in questions if q not in pending_questions)
            observation: Any = {"sub_questions": questions}
        elif action == "clarify":
            return AgentResult(answer=str(arguments.get("question", "Could you clarify your question?")), citations=[], sources=list(evidence.values()), retrieved=len(evidence), agent_steps=scratchpad + [{"action": action, "arguments": arguments}], clarification_needed=True)
        elif action == "synthesize":
            answer, citations = _final_answer(question, list(evidence.values()))
            return AgentResult(answer=answer, citations=citations, sources=list(evidence.values()), retrieved=len(evidence), agent_steps=scratchpad + [{"action": action, "arguments": arguments}])
        else:
            observation = tools.execute(action, arguments)
            if isinstance(observation, list):
                for item in observation:
                    if item.get("text"):
                        evidence[str(item.get("id") or item.get("source") + item["text"][:50])] = item
        scratchpad.append({"question": current, "action": action, "arguments": arguments, "observation": observation})
    answer, citations = _final_answer(question, list(evidence.values())) if evidence else ("I do not know.", [])
    return AgentResult(answer=answer, citations=citations, sources=list(evidence.values()), retrieved=len(evidence), agent_steps=scratchpad)
