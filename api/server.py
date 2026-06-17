"""
SentinelTrace REST API — FastAPI endpoints for trace inspection, replay, and AI analysis.

The /analyze endpoint is the entry point to the AI Root Cause Analyzer,
SentinelTrace's core AI mechanism. It returns a structured diagnosis of any
recorded run without restarting the live agent.

Start with: uvicorn api.server:app --reload
Docs at:    http://localhost:8000/docs
"""
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from flight_recorder.ai_debugger import RootCauseSubAgent
from flight_recorder.anomaly_detector import SilentFailureDetector
from flight_recorder.pattern_store import PatternStore
from flight_recorder.recorder import FlightRecorder
from flight_recorder.replay import ReplayEngine

app = FastAPI(
    title="SentinelTrace",
    version="0.1.0",
    description="Agent Execution Tracer with AI Root Cause Analysis",
)

recorder = FlightRecorder()
replay_engine = ReplayEngine(recorder)
analyzer = RootCauseSubAgent()
pattern_store = PatternStore(recorder)
detector = SilentFailureDetector(
    expected_tools=["search_kb", "query_db", "get_user_info", "send_notification"]
)


# ── Request/response models ───────────────────────────────────────────────────

class RunRequest(BaseModel):
    ticket_id: str


class ReplayRequest(BaseModel):
    patched_step: int | None = None
    patch_data: dict | None = None


class AnalyzeRequest(BaseModel):
    hint: str = ""


# ── Runs ─────────────────────────────────────────────────────────────────────

@app.get("/runs", summary="List all recorded runs")
def list_runs() -> list[dict]:
    conn = sqlite3.connect(recorder.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/runs/{run_id}", summary="Get run metadata")
def get_run(run_id: str) -> dict:
    conn = sqlite3.connect(recorder.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return dict(row)


@app.get("/runs/{run_id}/steps", summary="Get all steps of a run (decompressed)")
def get_steps(run_id: str) -> list[dict]:
    steps = recorder.get_steps(run_id)
    if not steps:
        raise HTTPException(404, f"No steps found for run '{run_id}'")
    return steps


@app.get("/runs/{run_id}/steps/{step_index}", summary="Inspect a single step")
def get_step(run_id: str, step_index: int) -> dict:
    """
    Returns the full recorded state at a specific step: input, output,
    model_name, temperature, tokens_in, tokens_out, cost_usd, hmac_sig.
    Satisfies the 'State inspection' acceptance criterion.
    """
    steps = recorder.get_steps(run_id)
    matches = [s for s in steps if s["step_index"] == step_index]
    if not matches:
        raise HTTPException(404, f"Step {step_index} not found in run '{run_id}'")
    return matches[0]


@app.get("/runs/{run_id}/anomalies", summary="Silent failure detection (heuristic, no LLM)")
def detect_anomalies(run_id: str) -> dict:
    """
    Scans a completed run for silent failure patterns without invoking an LLM.
    Detects: missing tool calls, tool loops, ignored results, premature termination,
    uncertain completion, hallucination signals.
    """
    steps = recorder.get_steps(run_id)
    if not steps:
        raise HTTPException(404, f"No steps for run '{run_id}'")
    report = detector.detect(run_id, steps)
    return report.to_dict()


@app.get("/runs/{run_id}/integrity", summary="Verify HMAC integrity of all steps")
def verify_integrity(run_id: str) -> dict:
    """
    Verifies the HMAC-SHA256 signature of every step in the trace.
    Returns {'clean': true} if the audit vault is untampered.
    Used for compliance export scenarios (Scenario C — SEC-0412).
    """
    tampered = recorder.verify_integrity(run_id)
    return {
        "run_id": run_id,
        "clean": len(tampered) == 0,
        "tampered_steps": tampered,
    }


# ── Agent trigger ─────────────────────────────────────────────────────────────

@app.post("/agent/run", summary="Trigger a live agent run (recorded)")
def trigger_run(req: RunRequest) -> dict:
    """Runs the Jira Triage Agent live and records the full trace."""
    from agent.jira_triage import run_live
    try:
        run_id = run_live(req.ticket_id)
        return {"run_id": run_id, "status": "recorded"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Replay ───────────────────────────────────────────────────────────────────

@app.post("/runs/{run_id}/replay", summary="Replay a run (deterministic, no live calls)")
def replay_run(run_id: str, req: ReplayRequest) -> dict:
    """
    Replays a recorded run. All tool calls return stored mock responses.
    send_notification and other side-effect tools are hard-blocked.

    Optionally patch a specific step to test a proposed fix.
    Satisfies 'Deterministic replay' and 'Divergence editing' acceptance criteria.
    """
    if not recorder.get_steps(run_id):
        raise HTTPException(404, f"No steps found for run '{run_id}'")

    if req.patched_step is not None:
        result = replay_engine.replay_with_patch(
            run_id, req.patched_step, req.patch_data or {}
        )
    else:
        result = replay_engine.replay(run_id)

    return {
        "run_id": result.run_id,
        "patched_step": result.patched_step,
        "steps_replayed": result.steps_replayed,
        "divergence_detected": result.divergence_detected,
        "divergence_step": result.divergence_step,
        "divergence_details": result.divergence_details,
    }


# ── AI Root Cause Analyzer ────────────────────────────────────────────────────

@app.post("/runs/{run_id}/analyze", summary="AI Root Cause Analysis (SentinelTrace core)")
def analyze_run(run_id: str, req: AnalyzeRequest) -> dict:
    """
    Invokes the RootCauseSubAgent on the recorded trace of the given run.

    This is the core AI mechanism of SentinelTrace — a dedicated LLM subagent
    that diagnoses failures without rerunning the live agent. Returns a
    structured report: category, confidence, failed step, root cause, and a
    concrete suggested fix.

    Inspired by Debug2Fix (Microsoft Research, ACM 2026): dedicated debugging
    subagents improve resolution rate by +21% over general-purpose approaches.
    """
    steps = recorder.get_steps(run_id)
    if not steps:
        raise HTTPException(404, f"No steps found for run '{run_id}'")

    similar = pattern_store.find_similar(run_id, top_k=3)
    context = pattern_store.format_for_context(similar)

    report = analyzer.analyze(run_id, steps, error_hint=req.hint, similar_runs_context=context)
    return {
        "run_id": run_id,
        "similar_runs_found": len(similar),
        "similar_run_ids": [r.run_id for r in similar],
        **report.to_dict(),
    }
