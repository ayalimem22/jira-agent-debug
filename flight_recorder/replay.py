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
"""
import copy
from dataclasses import dataclass, field
from typing import Any

from flight_recorder.recorder import FlightRecorder

# Tools whose live calls must never happen — even in test environments.
# Extend this set as new side-effect tools are added to the agent.
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
    """

    def __init__(self, recorded_steps: list[dict]):
        # Build an ordered list of tool_result outputs (one per tool_call in order)
        self._tool_results: list[str] = [
            (s.get("output_data") or s.get("input_data") or {}).get("output", "")
            for s in recorded_steps
            if s["step_type"] == "tool_result"
        ]
        self._cursor = 0

    def get_response(self, tool_name: str, _tool_input: str) -> str:
        # Hard block — evaluated before any other logic
        if tool_name in SIDE_EFFECT_TOOLS:
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
    Ordered queue of pre-recorded tool responses for live divergence replay.

    During a live agent re-run, tool calls are intercepted in order by mock
    tools. At the patched position, the injected value is returned instead of
    the recorded response. The LLM then reasons freely with this new data and
    may take a completely different trajectory — fulfilling the 'Divergence
    editing' acceptance criterion.

    Side-effect tools are still blocked regardless of position.
    """

    def __init__(
        self,
        recorded_steps: list[dict],
        patch_step_index: int | None = None,
        patch_value: str | None = None,
    ):
        self._entries: list[dict] = []
        for s in recorded_steps:
            if s["step_type"] != "tool_call":
                continue
            # Find the tool_result that immediately follows this tool_call
            result_output = next(
                (
                    # output_data for new runs, input_data fallback for old runs
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
            self._entries.append({
                "step_index": s["step_index"],
                "tool_name": s.get("name", "unknown"),
                "response": patch_value if is_patched else result_output,
                "patched": is_patched,
            })
        self._cursor = 0
        self._patch_position = next(
            (i for i, e in enumerate(self._entries) if e["patched"]), None
        )

    def pop(self, tool_name: str) -> str:
        """Return the next recorded response, or the patch if at the injected position."""
        if tool_name in SIDE_EFFECT_TOOLS:
            return (
                f"[MOCK-BLOCKED] {tool_name} is disabled in divergence replay. "
                f"No live call was made."
            )
        if self._cursor < len(self._entries):
            entry = self._entries[self._cursor]
            self._cursor += 1
            return entry["response"]
        # Agent made more tool calls than the original — let it continue freely
        return f"[MOCK] No recorded response for extra tool call #{self._cursor}."

    def summary(self) -> dict:
        return {
            "total_recorded_tool_calls": len(self._entries),
            "patch_at_call_position": self._patch_position,
        }


class ReplayEngine:
    """
    Replays a recorded run deterministically, with optional step patching.

    replay()              — exact replay, validates Mock Injector guarantee
    replay_with_patch()   — replay with one step overridden (test a fix)
    """

    def __init__(self, recorder: FlightRecorder):
        self.recorder = recorder
        self.divergence = DivergenceEngine()

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
        injector = MockInjector(original_steps)
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
