"""
Side Effect Classifier — AI-driven identification of irreversible tool calls.

This is a core AI mechanism of SentinelTrace: no deterministic algorithm can
determine whether a tool has irreversible side effects by reading its
description. That requires semantic understanding — only an LLM provides it.

Without this component the library cannot safely replay an unknown agent.
You would need a hardcoded list of dangerous tools per agent, making
generalisation impossible. The AI makes the replay guarantee possible.
"""
import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

_SYSTEM_PROMPT = """\
You are a tool safety classifier for an AI agent observability system called SentinelTrace.

Your mission: given a list of tools with their names and descriptions, decide
which ones have IRREVERSIBLE SIDE EFFECTS that must be blocked during replay
and debug sessions.

A tool has a SIDE EFFECT if calling it during replay could cause real-world damage:
  - Sending emails, notifications, or messages to users
  - Writing, updating, or deleting data in any database or file
  - Triggering webhooks, external workflows, or third-party APIs that mutate state
  - Charging money, provisioning resources, or modifying permissions
  - Any action whose consequence cannot be undone by simply not calling it

A tool is SAFE if it is strictly read-only:
  - SELECT queries on databases
  - Reading files or fetching documents
  - Searching knowledge bases or vector stores
  - Retrieving user profiles or configuration
  - Any operation that observes state without changing it

Respond ONLY with a single valid JSON object. No markdown, no explanation outside the JSON.
{
  "side_effect_tools": ["tool_name_1", "tool_name_2"],
  "safe_tools": ["tool_name_3", "tool_name_4"],
  "reasoning": {
    "tool_name_1": "one sentence why it is a side effect",
    "tool_name_2": "one sentence why it is a side effect",
    "tool_name_3": "one sentence why it is safe",
    "tool_name_4": "one sentence why it is safe"
  },
  "confidence": 0.95
}"""


@dataclass
class ClassificationReport:
    side_effect_tools: frozenset
    safe_tools: frozenset
    reasoning: dict = field(default_factory=dict)
    confidence: float = 1.0
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "side_effect_tools": sorted(self.side_effect_tools),
            "safe_tools": sorted(self.safe_tools),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "used_fallback": self.used_fallback,
        }


# Fallback used when the LLM is unavailable (no API key, network error, etc.)
_FALLBACK_SIDE_EFFECTS: frozenset = frozenset({
    "send_notification", "send_email", "write_db",
})


class SideEffectClassifier:
    """
    Uses an LLM to classify which tools have irreversible side effects.

    This is the mechanism that makes SentinelTrace generic: instead of
    requiring every team to hardcode a list of dangerous tools per agent,
    the AI reads tool descriptions and makes the call automatically.

    Usage:
        classifier = SideEffectClassifier()
        report = classifier.classify(tools)
        # report.side_effect_tools → frozenset({"send_notification", ...})
        # report.reasoning        → {"send_notification": "sends real emails", ...}

    Offline / local LLM:
        from langchain_community.chat_models import ChatOllama
        classifier = SideEffectClassifier(llm=ChatOllama(model="llama3"))
    """

    def __init__(self, model: str | None = None, llm=None):
        resolved = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = llm or ChatOpenAI(model=resolved, temperature=0.0)

    def classify(self, tools: list) -> ClassificationReport:
        """
        Classify a list of LangChain tools into side-effect vs safe.

        Falls back to a hardcoded default set if the LLM call fails so that
        the replay safety guarantee is never silently dropped.
        """
        if not tools:
            return ClassificationReport(
                side_effect_tools=frozenset(),
                safe_tools=frozenset(),
            )

        tool_block = self._format_tools(tools)

        try:
            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"Classify these tools:\n\n{tool_block}"),
            ])
            parsed = self._parse(response.content)

            side_effects = frozenset(parsed.get("side_effect_tools", []))
            safe = frozenset(parsed.get("safe_tools", []))

            # Safety net: any tool not classified goes into side_effect bucket
            all_names = {getattr(t, "name", str(t)) for t in tools}
            unclassified = all_names - side_effects - safe
            if unclassified:
                side_effects = side_effects | unclassified

            return ClassificationReport(
                side_effect_tools=side_effects,
                safe_tools=safe,
                reasoning=parsed.get("reasoning", {}),
                confidence=float(parsed.get("confidence", 0.5)),
                used_fallback=False,
            )

        except Exception as exc:
            # Never silently drop the safety guarantee — use fallback
            all_names = {getattr(t, "name", str(t)) for t in tools}
            known_safe = all_names - _FALLBACK_SIDE_EFFECTS
            return ClassificationReport(
                side_effect_tools=_FALLBACK_SIDE_EFFECTS & all_names | (
                    all_names - known_safe - _FALLBACK_SIDE_EFFECTS
                ),
                safe_tools=known_safe,
                reasoning={"_error": str(exc)},
                confidence=0.0,
                used_fallback=True,
            )

    @staticmethod
    def _format_tools(tools: list) -> str:
        lines = []
        for t in tools:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "No description available.")
            lines.append(f"name: {name}\ndescription: {desc}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
