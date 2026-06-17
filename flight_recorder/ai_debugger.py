"""
AI Root Cause Analyzer — the core AI mechanism of SentinelTrace.

This is NOT the cobaye agent under observation. This is a dedicated LLM subagent
whose sole responsibility is to diagnose failures in recorded execution traces.

Design rationale:
  Microsoft Research, Debug2Fix (Garg & Huang, ACM Feb 2026) demonstrates that
  a dedicated debugging subagent — one that receives a full trace as context and
  is specialized for failure diagnosis — outperforms a general-purpose agent asked
  to debug its own output by +21%. Specialization and separation of concerns are
  the key drivers.

  This subagent is called AFTER a run completes (or fails). It never runs live
  alongside the agent under observation. It reads only recorded data.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

Category = Literal["Exception", "RootCause", "VariableInspection", "Divergence", "SideEffect"]

_SYSTEM_PROMPT = """\
You are the SentinelTrace Root Cause Analyzer — a specialized debugging subagent.

You receive two inputs:
1. The full execution trace of a LangChain agent run that failed or behaved unexpectedly.
2. A set of similar past runs from the SentinelTrace database (few-shot pattern context).
   Use these to identify whether this is a known recurring pattern or a new failure mode.

Your task:
1. Identify the exact step where the failure or anomaly occurred.
2. Explain the root cause in plain language: what variable, what value, what decision was wrong.
3. If a similar past run exists, note whether this matches a known pattern.
4. Propose a concrete, actionable fix: a corrected prompt fragment, a patched variable, or SQL fix.
5. Classify the failure into exactly one category.

Categories:
  Exception          — a tool or LLM call raised an unhandled error
  RootCause          — the agent produced wrong output due to bad reasoning or bad input
  VariableInspection — a variable held an unexpected value that propagated
  Divergence         — replay output differs from original at a specific step
  SideEffect         — a side-effect tool was called when it should not have been

Respond ONLY with a single valid JSON object. No markdown, no explanation outside the JSON.
{
  "root_cause":       "<plain language, 1-3 sentences>",
  "failed_step":      <integer step index, or null>,
  "failed_variable":  "<variable name, or null>",
  "failed_value":     "<actual bad value, or null>",
  "suggested_fix":    "<concrete fix>",
  "category":         "<one of the five categories>",
  "confidence":       <float 0.0-1.0>,
  "is_known_pattern": <true if a similar past run had the same failure mode>,
  "pattern_note":     "<'same pattern as run XYZ' or 'new failure mode' or null>"
}"""


@dataclass
class RootCauseReport:
    root_cause: str
    failed_step: int | None
    failed_variable: str | None
    failed_value: str | None
    suggested_fix: str
    category: Category
    confidence: float
    raw_trace_steps: int
    is_known_pattern: bool = False
    pattern_note: str | None = None
    similar_runs_count: int = 0

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "failed_step": self.failed_step,
            "failed_variable": self.failed_variable,
            "failed_value": self.failed_value,
            "suggested_fix": self.suggested_fix,
            "category": self.category,
            "confidence": self.confidence,
            "is_known_pattern": self.is_known_pattern,
            "pattern_note": self.pattern_note,
            "similar_runs_used": self.similar_runs_count,
            "trace_steps_analyzed": self.raw_trace_steps,
        }


class RootCauseSubAgent:
    """
    Dedicated LLM subagent for diagnosing SentinelTrace execution traces.

    Receives a serialized trace (list of recorded steps) and returns a structured
    RootCauseReport. Uses temperature=0 for deterministic, reproducible diagnoses —
    the same trace always yields the same analysis.

    To run fully locally (zero API cost):
        from langchain_community.chat_models import ChatOllama
        agent = RootCauseSubAgent(llm=ChatOllama(model="llama3"))
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        llm: object | None = None,
    ):
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Allow injecting a custom LLM (e.g. ChatOllama) for local/offline use
        self.llm = llm or ChatOpenAI(model=resolved_model, temperature=temperature)

    def analyze(
        self,
        run_id: str,
        steps: list[dict],
        error_hint: str = "",
        similar_runs_context: str = "",
    ) -> RootCauseReport:
        """
        Analyze an execution trace and return a structured root cause report.

        Args:
            run_id:                The run identifier for traceability.
            steps:                 Full list of recorded steps from FlightRecorder.get_steps().
            error_hint:            Optional free-text hint from the operator.
            similar_runs_context:  Formatted output from PatternStore.format_for_context().
                                   Injected as few-shot context before the trace.
        """
        trace_text = self._serialize_trace(steps)
        hint_line = f"\nOperator hint: {error_hint}" if error_hint else ""
        pattern_section = f"\n\n{similar_runs_context}" if similar_runs_context else \
            "\n\n(No similar past runs found — first occurrence of this pattern.)"

        user_content = (
            f"Run ID: {run_id}\n"
            f"Total steps recorded: {len(steps)}"
            f"{hint_line}"
            f"{pattern_section}\n\n"
            f"--- Execution trace ---\n{trace_text}\n--- End of trace ---"
        )

        response = self.llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        parsed = self._parse_response(response.content)
        similar_count = similar_runs_context.count("[Pattern ")
        return RootCauseReport(
            root_cause=parsed.get("root_cause", "Unable to determine root cause."),
            failed_step=parsed.get("failed_step"),
            failed_variable=parsed.get("failed_variable"),
            failed_value=parsed.get("failed_value"),
            suggested_fix=parsed.get("suggested_fix", "No fix suggested."),
            category=parsed.get("category", "RootCause"),
            confidence=float(parsed.get("confidence", 0.5)),
            raw_trace_steps=len(steps),
            is_known_pattern=bool(parsed.get("is_known_pattern", False)),
            pattern_note=parsed.get("pattern_note"),
            similar_runs_count=similar_count,
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        """Extract JSON from the model response, tolerating light wrapping."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}

    @staticmethod
    def _serialize_trace(steps: list[dict]) -> str:
        """
        Convert steps to a compact, LLM-readable format.
        Truncates long values to keep the context window manageable.
        """
        lines: list[str] = []
        for s in steps:
            header = f"[{s['step_index']:02d}] {s['step_type'].upper()}"
            if s.get("name"):
                header += f" · {s['name']}"
            if s.get("model_name"):
                header += f" · model={s['model_name']}"
            if s.get("tokens_in") or s.get("tokens_out"):
                header += f" · tokens={s.get('tokens_in', '?')}/{s.get('tokens_out', '?')}"
            lines.append(header)

            inp = str(s.get("input_data", ""))
            lines.append(f"  IN:  {inp[:400]}" + (" …" if len(inp) > 400 else ""))

            out = s.get("output_data")
            if out:
                out_str = str(out)
                lines.append(f"  OUT: {out_str[:400]}" + (" …" if len(out_str) > 400 else ""))

            if s.get("anomaly"):
                lines.append("  *** ANOMALY FLAGGED ***")

        return "\n".join(lines)
