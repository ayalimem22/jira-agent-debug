"""
Replay Engine — deterministic agent simulation over recorded traces.

Two replay modes:

1. Static replay (ReplayEngine) — reads stored steps, returns recorded responses,
   never invokes the LLM. Used to validate the Mock Injector guarantee.

2. Live divergence replay (ToolResponseQueue) — the LLM actually re-runs.
   Tool calls are intercepted by mock tools that return recorded responses,
   except at the patched step where the injected value is returned instead.
   The LLM receives this new data and may choose a completely different trajectory.

The Mock Injector guarantee holds in both modes:
side-effect tools are blocked before any recorded data is consulted.

ToolResponseQueue uses per-tool name deques (dict[tool_name → deque[str]]).
When the LLM calls search_kb it gets the recorded search_kb response regardless
of call order. This avoids the step-explosion bug caused by a changed prompt
re-ordering tool calls and receiving incoherent FIFO responses.
"""
import copy
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from flight_recorder.recorder import FlightRecorder

# Default fallback — used only when SideEffectClassifier is unavailable.
# In normal operation this set is replaced by the AI-classified result.
SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({"send_notification", "send_email", "write_db"})


@dataclass
class ReplayResult:
    run_id: str
    patched_step: int | None
    steps_replayed: int
    divergence_detected: bool
    divergence_step: int | None
    divergence_details: dict | None
    outputs: list[dict] = field(default_factory=list)


class MockInjector:
    """
    Returns recorded tool responses during simulation.

    Architectural guarantee: there is no code path that reaches a live endpoint.
    Side-effect tools are blocked by name before any recorded data is consulted.
    This guarantee holds regardless of replay configuration or environment flags.

    side_effect_tools is injected by SideEffectClassifier at runtime so the
    blocked set reflects the actual agent under observation, not a hardcoded list.
    """

    def __init__(
        self,
        recorded_steps: list[dict],
        side_effect_tools: frozenset[str] | None = None,
    ):
        # Build an ordered list of tool_result outputs (one per tool_call in order)
        self._tool_results: list[str] = [
            (s.get("output_data") or s.get("input_data") or {}).get("output", "")
            for s in recorded_steps
            if s["step_type"] == "tool_result"
        ]
        self._cursor = 0
        # AI-classified set takes precedence; fall back to hardcoded default
        self._side_effects = side_effect_tools if side_effect_tools is not None \
            else SIDE_EFFECT_TOOLS

    def get_response(self, tool_name: str, _tool_input: str) -> str:
        # Hard block — evaluated before any other logic
        if tool_name in self._side_effects:
            return (
                f"[MOCK-BLOCKED] {tool_name} is disabled in simulation mode. "
                f"Recorded call intercepted. No live call was made."
            )
        if self._cursor < len(self._tool_results):
            result = self._tool_results[self._cursor]
            self._cursor += 1
            return f"[MOCK] {result}"
        return "[MOCK] No recorded response available for this step."


class DivergenceEngine:
    """
    Compares replay output to original trace step-by-step.
    Reports the index and details of the first step where behavior differs.
    """

    def compare(
        self,
        original: list[dict],
        replayed: list[dict],
    ) -> tuple[bool, int | None, dict | None]:
        for i, (orig, rep) in enumerate(zip(original, replayed)):
            if orig["step_type"] != rep.get("step_type"):
                return True, i, {
                    "field": "step_type",
                    "original": orig["step_type"],
                    "replayed": rep.get("step_type"),
                }
            if orig["step_type"] == "tool_result":
                orig_out = str((orig.get("output_data") or {}).get("output", ""))
                rep_out = str((rep.get("output_data") or {}).get("output", ""))
                # Strip simulation prefixes before comparing content
                rep_clean = rep_out.removeprefix("[MOCK] ").removeprefix("[MOCK-BLOCKED] ")
                if orig_out != rep_clean and not rep_out.startswith("[MOCK-BLOCKED]"):
                    return True, i, {
                        "field": "tool_output",
                        "original": orig_out[:300],
                        "replayed": rep_out[:300],
                    }
        return False, None, None


class ToolResponseQueue:
    """
    Name-keyed store of pre-recorded tool responses for live divergence replay.

    Responses are indexed by tool name, not by position. When the LLM calls
    search_kb it always gets a recorded search_kb response — even if the new
    prompt caused it to call tools in a different order than the original run.
    This prevents the step-explosion bug: a FIFO queue delivers incoherent
    responses when call order changes, causing the LLM to loop endlessly.

    Each tool name maps to a deque of its recorded outputs (in original order).
    Repeated calls to the same tool consume responses one by one from that deque.
    If the agent makes more calls than recorded, a fallback message is returned.

    At the patched step index, patch_value replaces the recorded response for
    that specific call to that specific tool.

    Side-effect tools are blocked before any other logic runs.
    """

    def __init__(
        self,
        recorded_steps: list[dict],
        patch_step_index: int | None = None,
        patch_value: str | None = None,
        side_effect_tools: frozenset[str] | None = None,
    ):
        # Per-tool deques: tool_name → deque of recorded outputs (in call order)
        self._queues: dict[str, deque] = {}
        self._total_recorded = 0
        self._patch_tool: str | None = None
        self._patch_call_pos: int | None = None  # 0-based index within that tool's calls

        # Per-tool call counter to track position for patch injection
        _tool_counters: dict[str, int] = {}

        for s in recorded_steps:
            if s["step_type"] != "tool_call":
                continue

            tool_name = s.get("name", "unknown")
            call_pos = _tool_counters.get(tool_name, 0)
            _tool_counters[tool_name] = call_pos + 1

            # Recorded output from the tool_result that follows this tool_call
            result_output = next(
                (
                    (r.get("output_data") or r.get("input_data") or {}).get("output", "")
                    for r in recorded_steps
                    if r["step_index"] == s["step_index"] + 1
                    and r["step_type"] == "tool_result"
                ),
                "",
            )

            is_patched = (
                patch_step_index is not None
                and s["step_index"] == patch_step_index
            )
            if is_patched:
                self._patch_tool = tool_name
                self._patch_call_pos = call_pos

            response = patch_value if is_patched else result_output

            if tool_name not in self._queues:
                self._queues[tool_name] = deque()
            self._queues[tool_name].append(response)
            self._total_recorded += 1

        self._side_effects = side_effect_tools if side_effect_tools is not None \
            else SIDE_EFFECT_TOOLS

    def pop(self, tool_name: str) -> str:
        """Return the next recorded response for this tool, or block if side-effect."""
        if tool_name in self._side_effects:
            return (
                f"[MOCK-BLOCKED] {tool_name} is disabled in divergence replay. "
                f"No live call was made."
            )
        q = self._queues.get(tool_name)
        if q:
            return f"[MOCK] {q.popleft()}"
        # Tool was never called in the original OR all recorded responses consumed
        return f"[MOCK] No recorded response for {tool_name} — tool not in original trace."

    def summary(self) -> dict:
        return {
            "total_recorded_tool_calls": self._total_recorded,
            "patch_tool": self._patch_tool,
            "patch_call_position": self._patch_call_pos,
        }


class TrajectoryDriftAnalyzer:
    """
    LLM-based semantic comparison between original and diverged trajectories.

    Computes a drift_score (0.0–1.0) and a human-readable summary of what
    changed — beyond the simple tool-call sequence comparison that only
    catches structural differences.

    A low drift score means the agent reached a similar conclusion by a similar
    path. A high drift score means the patch caused a fundamentally different
    reasoning trajectory, even if the tool call sequence looks identical.
    """

    def analyze(
        self,
        orig_steps: list[dict],
        div_steps: list[dict],
    ) -> dict:
        try:
            import json as _json
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError:
            return {"semantic_drift": None, "drift_summary": "langchain not available"}

        def _final(steps):
            for s in reversed(steps):
                if s["step_type"] == "agent_finish":
                    return ((s.get("input_data") or {}).get("final_output") or {}).get("output", "")
            return ""

        def _llm_thoughts(steps):
            out = []
            for s in steps:
                if s["step_type"] != "llm_result":
                    continue
                gens = ((s.get("input_data") or {}).get("generations") or [[]])[0]
                text = gens[0] if isinstance(gens, list) and gens else str(gens)
                th = re.search(r"Thought:(.*?)(?=Action:|Final Answer:|$)", text, re.S)
                if th:
                    out.append(th.group(1).strip()[:200])
            return out

        orig_tools  = [s.get("name") for s in orig_steps if s["step_type"] == "tool_call"]
        div_tools   = [s.get("name") for s in div_steps  if s["step_type"] == "tool_call"]
        orig_final  = _final(orig_steps)[:500]
        div_final   = _final(div_steps)[:500]
        orig_thoughts = _llm_thoughts(orig_steps)[:3]
        div_thoughts  = _llm_thoughts(div_steps)[:3]

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            f"Original tool sequence : {orig_tools}\n"
            f"Diverged tool sequence : {div_tools}\n\n"
            f"Original reasoning (first 3 thoughts): {orig_thoughts}\n"
            f"Diverged reasoning (first 3 thoughts): {div_thoughts}\n\n"
            f"Original final answer:\n{orig_final}\n\n"
            f"Diverged final answer:\n{div_final}\n\n"
            "Rate the SEMANTIC drift between the two runs. "
            "Respond ONLY with valid JSON:\n"
            '{"drift_score": 0.0-1.0, '
            '"drift_summary": "one sentence — what changed in behavior/decision", '
            '"decision_point": "at which step did reasoning diverge"}\n'
            "drift_score 0.0 = identical outcome, 1.0 = completely different trajectory."
        )
        try:
            resp = llm.invoke([
                SystemMessage(content="You are an expert at comparing AI agent execution traces."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = _json.loads(raw)
            return {
                "semantic_drift": float(data.get("drift_score", 0.0)),
                "drift_summary":  data.get("drift_summary", ""),
                "decision_point": data.get("decision_point", ""),
            }
        except Exception:
            # Fallback: structural diff score (Jaccard on tool sequences)
            s1, s2 = set(orig_tools), set(div_tools)
            jaccard = 1.0 - (len(s1 & s2) / max(len(s1 | s2), 1))
            return {
                "semantic_drift": round(jaccard, 2),
                "drift_summary":  "Structural tool sequence difference (LLM comparison unavailable)",
                "decision_point": None,
            }


class ReplayEngine:
    """
    Replays a recorded run deterministically, with optional step patching.

    replay()              — exact replay, validates Mock Injector guarantee
    replay_with_patch()   — replay with one step overridden (test a fix)
    """

    def __init__(
        self,
        recorder: FlightRecorder,
        side_effect_tools: frozenset[str] | None = None,
    ):
        self.recorder = recorder
        self.divergence = DivergenceEngine()
        self._side_effects = side_effect_tools

    def replay(self, run_id: str) -> ReplayResult:
        """Replay a run exactly as recorded. Validates that no live calls occur."""
        return self._execute(run_id, patched_step=None, patch_data=None)

    def replay_with_patch(
        self,
        run_id: str,
        step_index: int,
        patch_data: dict,
    ) -> ReplayResult:
        """
        Replay with a specific step's input_data overridden.
        Use this to verify that a proposed fix (from RootCauseSubAgent) resolves
        the failure without rerunning the live agent.
        """
        return self._execute(run_id, patched_step=step_index, patch_data=patch_data)

    def _execute(
        self,
        run_id: str,
        patched_step: int | None,
        patch_data: dict | None,
    ) -> ReplayResult:
        original_steps = self.recorder.get_steps(run_id)
        injector = MockInjector(original_steps, side_effect_tools=self._side_effects)
        replayed: list[dict] = []

        for step in original_steps:
            sim_step = copy.deepcopy(step)

            # Apply patch at the designated step index
            if patched_step is not None and step["step_index"] == patched_step:
                sim_step["input_data"] = patch_data or step["input_data"]
                sim_step["_patched"] = True

            # Intercept all tool calls — no live endpoint reachable in simulation
            if step["step_type"] == "tool_call":
                tool_name = step.get("name") or "unknown"
                tool_input = str((step.get("input_data") or {}).get("input", ""))
                mock_out = injector.get_response(tool_name, tool_input)
                sim_step["output_data"] = {"output": mock_out}
                sim_step["_simulated"] = True

            replayed.append(sim_step)

        diverged, div_step, div_details = self.divergence.compare(original_steps, replayed)

        return ReplayResult(
            run_id=run_id,
            patched_step=patched_step,
            steps_replayed=len(replayed),
            divergence_detected=diverged,
            divergence_step=div_step,
            divergence_details=div_details,
            outputs=replayed,
        )
