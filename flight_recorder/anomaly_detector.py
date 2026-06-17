"""
Silent Failure Detector — heuristic detection of runs that look successful but aren't.

A silent failure is an agent run that completed without raising an exception
but produced a wrong, incomplete, or hallucinated result. Standard log-based
observability misses these entirely because there is no error to catch.

This detector runs on recorded steps WITHOUT calling an LLM — it is fast,
deterministic, and costs zero tokens. It flags suspicious patterns so the
RootCauseSubAgent can be invoked with specific anomaly context.

Six failure modes detected:

  MissingToolCall       — an expected tool was never called (agent skipped a step)
  ToolLoop              — same tool called 3+ times with similar input (stuck)
  IgnoredToolResult     — tool_result content not referenced in next LLM output
  PrematureTermination  — Final Answer reached too early (few steps, no key tools)
  UncertainCompletion   — Final Answer contains hedging language ("I think", "probably")
  HallucinationSignal   — Final Answer mentions entities not present in any tool result
"""
import re
from dataclasses import dataclass, field
from enum import Enum


class AnomalyType(str, Enum):
    MISSING_TOOL_CALL = "MissingToolCall"
    TOOL_LOOP = "ToolLoop"
    IGNORED_TOOL_RESULT = "IgnoredToolResult"
    EMPTY_TOOL_RESULT = "EmptyToolResult"        # DB ERROR, [], "No articles found"
    PREMATURE_TERMINATION = "PrematureTermination"
    UNCERTAIN_COMPLETION = "UncertainCompletion"
    HALLUCINATION_SIGNAL = "HallucinationSignal"


@dataclass
class Anomaly:
    type: AnomalyType
    step_index: int | None
    description: str
    severity: str  # "high" | "medium" | "low"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "step_index": self.step_index,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class DetectionReport:
    run_id: str
    anomalies: list[Anomaly] = field(default_factory=list)
    is_silent_failure: bool = False
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "is_silent_failure": self.is_silent_failure,
            "confidence": self.confidence,
            "summary": self.summary,
            "anomaly_count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


# Words that indicate hedging / uncertainty in a final answer
_HEDGE_PATTERNS = [
    r"i (think|believe|assume|suppose)",
    r"(probably|possibly|maybe|perhaps)",
    r"i (couldn't|could not|was unable to|failed to)",
    r"i (don't|do not) (know|have)",
    r"(not sure|uncertain|unclear)",
    r"i (was not able|wasn't able)",
    r"(it seems|it appears|it looks like)",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)


def _extract_final_answer(steps: list[dict]) -> tuple[str, int | None]:
    """Return (final_answer_text, step_index) from the last llm_result, or ('', None)."""
    for s in reversed(steps):
        if s["step_type"] == "llm_result":
            gens = (s.get("input_data") or {}).get("generations", [[]])
            text = ""
            if gens and gens[0]:
                text = gens[0][0] if isinstance(gens[0], list) else str(gens[0])
            if "Final Answer:" in text:
                return text.split("Final Answer:", 1)[-1].strip(), s["step_index"]
    return "", None


def _tool_calls(steps: list[dict]) -> list[dict]:
    return [s for s in steps if s["step_type"] == "tool_call"]


def _tool_results(steps: list[dict]) -> list[dict]:
    return [s for s in steps if s["step_type"] == "tool_result"]


def _llm_results(steps: list[dict]) -> list[dict]:
    return [s for s in steps if s["step_type"] == "llm_result"]


class SilentFailureDetector:
    """
    Heuristic detector for silent failures in recorded agent runs.

    Usage:
        detector = SilentFailureDetector(
            expected_tools=["query_db", "send_notification"]
        )
        report = detector.detect(run_id, steps)
        if report.is_silent_failure:
            # escalate to RootCauseSubAgent with anomaly context
    """

    def __init__(self, expected_tools: list[str] | None = None):
        # Tools that MUST appear in a successful run; missing = silent failure
        self.expected_tools = expected_tools or []

    def detect(self, run_id: str, steps: list[dict]) -> DetectionReport:
        report = DetectionReport(run_id=run_id)

        checks = [
            self._check_missing_tools,
            self._check_tool_loops,
            self._check_empty_tool_results,
            self._check_ignored_results,
            self._check_premature_termination,
            self._check_uncertain_completion,
            self._check_hallucination_signal,
        ]
        for check in checks:
            report.anomalies.extend(check(steps))

        high = sum(1 for a in report.anomalies if a.severity == "high")
        medium = sum(1 for a in report.anomalies if a.severity == "medium")

        # Confidence: weighted score — high anomaly = 0.4, medium = 0.2
        raw_conf = min(1.0, high * 0.4 + medium * 0.2 + len(report.anomalies) * 0.05)
        report.confidence = round(raw_conf, 2)
        report.is_silent_failure = report.confidence >= 0.4 or high >= 1

        if not report.anomalies:
            report.summary = "No silent failure indicators detected."
        else:
            types = ", ".join(sorted({a.type for a in report.anomalies}))
            report.summary = (
                f"{len(report.anomalies)} anomaly/ies detected: {types}. "
                f"Silent failure confidence: {report.confidence:.0%}."
            )
        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_missing_tools(self, steps: list[dict]) -> list[Anomaly]:
        """Flag tools that were expected but never called."""
        called = {s["name"] for s in _tool_calls(steps) if s.get("name")}
        missing = [t for t in self.expected_tools if t not in called]
        return [
            Anomaly(
                type=AnomalyType.MISSING_TOOL_CALL,
                step_index=None,
                description=f"Expected tool '{t}' was never called — agent may have skipped a required step.",
                severity="high",
            )
            for t in missing
        ]

    def _check_tool_loops(self, steps: list[dict]) -> list[Anomaly]:
        """Detect same tool called 3+ times in a row with similar input."""
        anomalies = []
        calls = _tool_calls(steps)
        counts: dict[str, list[int]] = {}
        for s in calls:
            name = s.get("name", "unknown")
            counts.setdefault(name, []).append(s["step_index"])

        for name, indices in counts.items():
            if len(indices) >= 3:
                anomalies.append(Anomaly(
                    type=AnomalyType.TOOL_LOOP,
                    step_index=indices[0],
                    description=f"'{name}' called {len(indices)} times (steps {indices}). Possible reasoning loop.",
                    severity="high",
                ))
        return anomalies

    def _check_empty_tool_results(self, steps: list[dict]) -> list[Anomaly]:
        """
        Flag tool calls that returned an empty, error, or useless response.
        These are the most common source of silent failures: the agent received
        nothing useful but continued anyway and hallucinated the rest.
        """
        # Per-tool rules: what counts as a real failure vs a normal empty response
        _REAL_ERRORS: dict[str, list[str]] = {
            "query_db":        ["db error", "error:", "[]"],   # empty result IS an error
            "get_user_info":   ["not found", "error:"],
            "send_notification": ["error:"],
            # search_kb: "No articles found" is normal — not flagged
        }
        anomalies = []
        for s in steps:
            if s["step_type"] != "tool_result":
                continue
            # Fallback: output_data (new runs) then input_data (old runs)
            output = str(
                (s.get("output_data") or s.get("input_data") or {}).get("output", "")
            ).strip()
            tool_call = next(
                (tc for tc in steps if tc["step_index"] == s["step_index"] - 1
                 and tc["step_type"] == "tool_call"),
                None,
            )
            tool_name = tool_call.get("name", "unknown") if tool_call else "unknown"
            tool_input = str((tool_call.get("input_data") or {}).get("input", "")) if tool_call else ""

            error_signals = _REAL_ERRORS.get(tool_name, [])
            is_empty = (
                (not output and tool_name != "search_kb")
                or any(output.lower().startswith(sig) for sig in error_signals)
                or (output == "[]" and tool_name == "query_db")
            )
            if is_empty:
                anomalies.append(Anomaly(
                    type=AnomalyType.EMPTY_TOOL_RESULT,
                    step_index=s["step_index"],
                    description=(
                        f"'{tool_name}' returned empty/error at step {s['step_index']}: "
                        f"'{output[:120]}'. "
                        f"Input was: '{tool_input[:120]}'. "
                        f"Agent may have continued with hallucinated data."
                    ),
                    severity="high",
                ))
        return anomalies

    def _check_ignored_results(self, steps: list[dict]) -> list[Anomaly]:
        """
        Detect tool results that are not referenced in the subsequent LLM output.
        Heuristic: extract 3+ char tokens from tool result and check overlap with
        the next llm_result text.
        """
        anomalies = []
        for i, s in enumerate(steps):
            if s["step_type"] != "tool_result":
                continue
            result_text = str(
                (s.get("output_data") or s.get("input_data") or {}).get("output", "")
            )
            if not result_text or result_text.startswith("[MOCK"):
                continue

            # Find the next llm_result after this tool_result
            next_llm = next(
                (r for r in steps[i + 1:] if r["step_type"] == "llm_result"),
                None,
            )
            if not next_llm:
                continue

            llm_gens = (next_llm.get("input_data") or {}).get("generations", [[]])
            llm_text = ""
            if llm_gens and llm_gens[0]:
                llm_text = llm_gens[0][0] if isinstance(llm_gens[0], list) else str(llm_gens[0])

            # Token overlap — meaningful tokens only (len >= 4)
            result_tokens = set(re.findall(r'\b\w{4,}\b', result_text.lower()))
            llm_tokens = set(re.findall(r'\b\w{4,}\b', llm_text.lower()))
            stop_words = {"that", "this", "with", "from", "have", "will", "been",
                          "were", "they", "what", "when", "where", "which", "then"}
            meaningful = result_tokens - stop_words
            if meaningful and not (meaningful & llm_tokens):
                anomalies.append(Anomaly(
                    type=AnomalyType.IGNORED_TOOL_RESULT,
                    step_index=s["step_index"],
                    description=(
                        f"Tool result at step {s['step_index']} has no token overlap "
                        f"with the next LLM output — the agent may have ignored it and hallucinated."
                    ),
                    severity="medium",
                ))
        return anomalies

    def _check_premature_termination(self, steps: list[dict]) -> list[Anomaly]:
        """
        Flag a Final Answer reached with very few steps or before key tools were called.
        Threshold: fewer than 3 tool calls for a multi-step workflow.
        """
        calls = _tool_calls(steps)
        final_answer, fa_step = _extract_final_answer(steps)
        if not final_answer or len(calls) >= 3:
            return []
        return [Anomaly(
            type=AnomalyType.PREMATURE_TERMINATION,
            step_index=fa_step,
            description=(
                f"Agent reached Final Answer after only {len(calls)} tool call(s). "
                f"Expected at least 3 for this workflow."
            ),
            severity="medium",
        )]

    def _check_uncertain_completion(self, steps: list[dict]) -> list[Anomaly]:
        """Detect hedging language in the Final Answer."""
        final_answer, fa_step = _extract_final_answer(steps)
        if not final_answer:
            return []
        match = _HEDGE_RE.search(final_answer)
        if match:
            return [Anomaly(
                type=AnomalyType.UNCERTAIN_COMPLETION,
                step_index=fa_step,
                description=(
                    f"Final Answer contains hedging language: '{match.group()}'. "
                    f"Agent may have completed without confidence in its result."
                ),
                severity="medium",
            )]
        return []

    def _check_hallucination_signal(self, steps: list[dict]) -> list[Anomaly]:
        """
        Check if the Final Answer mentions capitalized proper nouns (names, IDs)
        that do not appear in any tool result — potential hallucination.
        """
        final_answer, fa_step = _extract_final_answer(steps)
        if not final_answer:
            return []

        all_tool_content = " ".join(
            str((s.get("output_data") or {}).get("output", ""))
            for s in _tool_results(steps)
        )

        # Capitalized words are likely names/IDs — check if present in tool data
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', final_answer)
        excluded = {"Final", "Answer", "The", "This", "Done", "Thank", "Please",
                    "Note", "Based", "According", "Dear", "Hello", "Ticket"}
        suspicious = [
            w for w in proper_nouns
            if w not in excluded and w not in all_tool_content
        ]
        # Deduplicate
        suspicious = list(dict.fromkeys(suspicious))

        if len(suspicious) >= 2:
            return [Anomaly(
                type=AnomalyType.HALLUCINATION_SIGNAL,
                step_index=fa_step,
                description=(
                    f"Final Answer references {len(suspicious)} term(s) not found "
                    f"in any tool result: {', '.join(suspicious[:5])}. "
                    f"Possible hallucination."
                ),
                severity="high" if len(suspicious) >= 3 else "medium",
            )]
        return []
