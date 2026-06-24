"""
Silent Failure Detector — heuristic + LLM-based detection of runs that look
successful but aren't.

A silent failure is an agent run that completed without raising an exception
but produced a wrong, incomplete, or hallucinated result. Standard log-based
observability misses these entirely because there is no error to catch.

Two-layer detection:

Layer 1 — Heuristics (zero LLM cost, runs always):
  MissingToolCall       — an expected tool was never called
  ToolLoop              — same tool called 3+ times (stuck)
  EmptyToolResult       — tool returned error / empty response
  MalformedToolInput    — bad SQL or hostile tone in tool input
  IgnoredToolResult     — tool result not referenced in next LLM output
  PrematureTermination  — Final Answer reached too early
  UncertainCompletion   — hedging language ("I think", "probably")
  HallucinationSignal   — proper nouns in answer not found in tool data

Layer 2 — SemanticValidator (1 LLM call, only when heuristic confidence < 0.4):
  Catches subtle hallucinations, semantically ignored tool data, and implicit
  uncertainty that regex patterns miss. Replaces the three weakest heuristics
  with a single grounded LLM judgment.
"""
import json
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
    MALFORMED_TOOL_INPUT = "MalformedToolInput"


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

# Patterns that indicate a malformed SQL input (unquoted string identifier in WHERE)
_SQL_BARE_ID_RE = re.compile(r"WHERE\s+\w+\s*=\s*[A-Z][A-Z0-9\-_]+\b(?!['\"])", re.IGNORECASE)

# Keywords that suggest hostile/aggressive tone in a notification payload
_AGGRESSIVE_PATTERNS = [
    r"unacceptable", r"incompetent", r"immediately or face",
    r"consequences", r"completely wrong", r"disgrace",
    r"you (must|have to|need to) fix", r"(threatening|hostile)",
]
_AGGRESSIVE_RE = re.compile("|".join(_AGGRESSIVE_PATTERNS), re.IGNORECASE)


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


class SemanticValidator:
    """
    LLM-based fallback detector for subtle failures that evade heuristics.

    Called only when heuristic confidence < 0.4 — costs exactly 1 LLM call.
    Catches: subtle hallucination, semantically ignored tool results, and
    implicit uncertainty that regex patterns miss.

    Replaces the three weakest heuristics (token-overlap IgnoredToolResult,
    regex UncertainCompletion, proper-noun HallucinationSignal) with a single
    grounded LLM judgment that understands semantics, not just surface patterns.
    """

    _SYSTEM = (
        "You are a quality auditor for AI agent runs. "
        "Your job is to find subtle issues in a Final Answer given the actual data "
        "the agent received from its tools. Be strict but fair — only flag real issues."
    )

    def validate(
        self,
        final_answer: str,
        tool_results_text: str,
    ) -> list["Anomaly"]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError:
            return []

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            f"Tool results the agent received:\n"
            f"---\n{tool_results_text[:2500]}\n---\n\n"
            f"Final Answer the agent produced:\n"
            f"---\n{final_answer[:800]}\n---\n\n"
            "Identify subtle issues. Respond ONLY with valid JSON:\n"
            '{"issues":['
            '{"type":"HALLUCINATION|IGNORED_DATA|IMPLICIT_UNCERTAINTY",'
            '"description":"one sentence","severity":"high|medium"}'
            ']}\n'
            'If no issues, return {"issues":[]}. '
            "HALLUCINATION = answer mentions specific facts/names/numbers absent from tool results. "
            "IGNORED_DATA = important tool result data completely absent from answer. "
            "IMPLICIT_UNCERTAINTY = answer hedges subtly without explicit keywords."
        )
        try:
            resp = llm.invoke([
                SystemMessage(content=self._SYSTEM),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
        except Exception:
            return []

        _type_map = {
            "HALLUCINATION":       AnomalyType.HALLUCINATION_SIGNAL,
            "IGNORED_DATA":        AnomalyType.IGNORED_TOOL_RESULT,
            "IMPLICIT_UNCERTAINTY": AnomalyType.UNCERTAIN_COMPLETION,
        }
        anomalies = []
        for issue in data.get("issues", []):
            atype = _type_map.get(issue.get("type"), AnomalyType.HALLUCINATION_SIGNAL)
            anomalies.append(Anomaly(
                type=atype,
                step_index=None,
                description=f"[SemanticValidator] {issue.get('description', '')}",
                severity=issue.get("severity", "medium"),
            ))
        return anomalies


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

    def detect(self, run_id: str, steps: list[dict], semantic: bool = True) -> DetectionReport:
        report = DetectionReport(run_id=run_id)

        # ── Layer 1: fast heuristics (zero LLM cost) ─────────────────────────
        checks = [
            self._check_missing_tools,
            self._check_tool_loops,
            self._check_empty_tool_results,
            self._check_tool_call_inputs,
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

        # ── Layer 2: SemanticValidator (1 LLM call, only when heuristics find nothing) ──
        # The three weakest heuristics (token-overlap, regex uncertainty, proper-noun
        # hallucination) have high false-positive and false-negative rates. When they
        # find nothing, the SemanticValidator does a grounded LLM judgment instead.
        if semantic and report.confidence < 0.4:
            final_answer, _ = _extract_final_answer(steps)
            tool_results_text = "\n\n".join(
                f"[{s.get('name','tool')}]: "
                + str((s.get("output_data") or s.get("input_data") or {}).get("output", ""))
                for s in steps
                if s["step_type"] == "tool_result"
            )
            if final_answer and tool_results_text:
                semantic_anomalies = SemanticValidator().validate(final_answer, tool_results_text)
                report.anomalies.extend(semantic_anomalies)
                # Recalculate with semantic findings
                high   = sum(1 for a in report.anomalies if a.severity == "high")
                medium = sum(1 for a in report.anomalies if a.severity == "medium")
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

    def _check_tool_call_inputs(self, steps: list[dict]) -> list[Anomaly]:
        """
        Inspect what the agent SENT to each tool — not just what came back.
        Catches two classes of input-side problems:
          1. Malformed SQL: unquoted string identifier in WHERE clause
             (e.g. WHERE id = DB-1193 instead of WHERE id = 'DB-1193')
          2. Aggressive/hostile content in send_notification payloads
        Both are detectable before the tool result arrives — earlier signal,
        zero LLM cost.
        """
        anomalies = []
        for s in _tool_calls(steps):
            tool_name = s.get("name", "")
            raw_input = str((s.get("input_data") or {}).get("input", ""))

            if tool_name == "query_db" and _SQL_BARE_ID_RE.search(raw_input):
                anomalies.append(Anomaly(
                    type=AnomalyType.MALFORMED_TOOL_INPUT,
                    step_index=s["step_index"],
                    description=(
                        f"query_db call at step {s['step_index']} contains an unquoted "
                        f"string identifier in the WHERE clause: '{raw_input[:120]}'. "
                        f"Likely cause of DB ERROR in subsequent tool_result."
                    ),
                    severity="high",
                ))

            if tool_name == "send_notification":
                match = _AGGRESSIVE_RE.search(raw_input)
                if match:
                    anomalies.append(Anomaly(
                        type=AnomalyType.MALFORMED_TOOL_INPUT,
                        step_index=s["step_index"],
                        description=(
                            f"send_notification at step {s['step_index']} contains "
                            f"potentially hostile language: '{match.group()}'. "
                            f"Payload: '{raw_input[:160]}'."
                        ),
                        severity="high",
                    ))
        return anomalies

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
