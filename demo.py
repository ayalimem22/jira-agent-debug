#!/usr/bin/env python3
"""
SentinelTrace — Live Pipeline Demo
Chains everything end-to-end with a single command:

  python demo.py                   # runs on PROD-2847
  python demo.py DB-1193           # specify ticket
  python demo.py --use RUN_ID      # replay an existing run (no LLM cost)
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── Terminal colours ──────────────────────────────────────────────────────────
def _c(t, code): return f"\033[{code}m{t}\033[0m"
B  = lambda t: _c(t, '1')       # bold
DIM= lambda t: _c(t, '2')       # dim
BLU= lambda t: _c(t, '34')      # blue
CYN= lambda t: _c(t, '36')      # cyan
GRN= lambda t: _c(t, '32')      # green
RED= lambda t: _c(t, '31')      # red
YLW= lambda t: _c(t, '33')      # yellow
MAG= lambda t: _c(t, '35')      # magenta

W = 64
def banner(title):
    print(f"\n{'━'*W}\n  {B(title)}\n{'━'*W}")

def section(n, label):
    print(f"\n{CYN(f'  [{n}/6]')}  {B(label)}")
    print(f"         {'─'*50}")

def ok(msg):   print(f"         {GRN('✓')}  {msg}")
def warn(msg): print(f"         {YLW('⚠')}  {msg}")
def bad(msg):  print(f"         {RED('✕')}  {msg}")
def info(msg): print(f"         {DIM('·')}  {msg}")
def sub(msg):  print(f"              {DIM(msg)}")

# ── Args ──────────────────────────────────────────────────────────────────────
TICKET   = "PROD-2847"
USE_RUN  = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--use" and i + 1 < len(sys.argv[1:]):
        USE_RUN = sys.argv[i + 2]
    elif not arg.startswith("--"):
        TICKET = arg

# ── Prerequisites check ───────────────────────────────────────────────────────
banner("SENTINELTRACE  ·  Live Pipeline Demo")
print(f"  {DIM('Agent Run  →  Detection  →  Replay  →  AI Analysis  →  Divergence')}")
print()

if not os.getenv("OPENAI_API_KEY") and not USE_RUN:
    bad("OPENAI_API_KEY missing in .env")
    bad("Add it: OPENAI_API_KEY=sk-...  then retry")
    bad("Or replay an existing run:  python demo.py --use <RUN_ID>")
    sys.exit(1)

DB_PATH = ROOT / "agent" / "fixtures" / "tickets.db"
if not DB_PATH.exists():
    warn("tickets.db not found — seeding fixtures…")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "agent" / "fixtures" / "seed_db.py")], check=True)
    ok("Fixtures seeded")
else:
    ok(f"Fixtures: {DB_PATH.name}")

ok(f"Model   : {os.getenv('OPENAI_MODEL','gpt-4o-mini')}")

# ── Late imports (after sys.path is ready) ────────────────────────────────────
from flight_recorder.recorder import FlightRecorder
from flight_recorder.replay import ReplayEngine
from flight_recorder.anomaly_detector import SilentFailureDetector
from flight_recorder.ai_debugger import RootCauseSubAgent
from flight_recorder.pattern_store import PatternStore
from flight_recorder.side_effect_classifier import SideEffectClassifier
from agent.jira_triage import TOOLS, JIRA_EXPECTED_TOOLS, run_live, run_divergence, DB_PATH as JIRA_DB

recorder = FlightRecorder()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — AI TOOL SAFETY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
section(1, "AI Tool Safety Classification")
info("SideEffectClassifier reads tool descriptions and decides what to block.")
info("No hardcoded list — the LLM makes the call.")
print()

classifier = SideEffectClassifier()
cls_report = classifier.classify(TOOLS)

if cls_report.used_fallback:
    warn(f"LLM unavailable — fallback set: {sorted(cls_report.side_effect_tools)}")
else:
    ok(f"Confidence: {cls_report.confidence:.0%}")
    for name, reason in cls_report.reasoning.items():
        if name == "_error":
            continue
        tag = f"{YLW('⚠ BLOCK')}" if name in cls_report.side_effect_tools else f"{GRN('✓ SAFE ')}"
        print(f"         {tag}  {B(name):<22} {DIM(reason[:62])}")

SIDE_EFFECTS = cls_report.side_effect_tools
time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LIVE AGENT RUN  (or reuse existing)
# ══════════════════════════════════════════════════════════════════════════════
section(2, "Live Agent Run  (FlightRecorderCallback recording every step)")

if USE_RUN:
    run_id = USE_RUN
    steps  = recorder.get_steps(run_id)
    if not steps:
        bad(f"No steps found for run '{run_id}'")
        sys.exit(1)
    ok(f"Reusing existing run: {B(run_id)}")
    ok(f"Steps loaded        : {len(steps)}")
else:
    info(f"Ticket  : {B(TICKET)}")
    info("The LLM is running live. Every call is CBOR-compressed and HMAC-signed.")
    print()
    run_id = run_live(TICKET)
    steps  = recorder.get_steps(run_id)
    print()

ok(f"Run ID  : {B(run_id)}")

tool_calls = [s for s in steps if s["step_type"] == "tool_call"]
llm_steps  = [s for s in steps if s["step_type"] in ("llm_start", "llm_result")]
ok(f"Steps   : {len(steps)} total  ({len(llm_steps)} LLM,  {len(tool_calls)} tool calls)")
info(f"Tool sequence: {' → '.join(s.get('name','?') for s in tool_calls) or '(none)'}")
time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SILENT FAILURE DETECTION  (0 tokens)
# ══════════════════════════════════════════════════════════════════════════════
section(3, "Silent Failure Detection  (7 heuristics, 0 LLM calls)")
info("Scanning for: ToolLoop, MissingToolCall, EmptyToolResult,")
info("              IgnoredToolResult, PrematureTermination,")
info("              UncertainCompletion, HallucinationSignal")
print()

detector   = SilentFailureDetector(expected_tools=JIRA_EXPECTED_TOOLS)
det_report = detector.detect(run_id, steps)

high_ct   = sum(1 for a in det_report.anomalies if a.severity == "high")
medium_ct = sum(1 for a in det_report.anomalies if a.severity == "medium")

sf_label  = f"{RED('YES — silent failure')} ({det_report.confidence:.0%} confidence)" \
            if det_report.is_silent_failure else f"{GRN('NO — run looks healthy')}"
print(f"         Result: {sf_label}")
print(f"         Anomalies: {RED(str(high_ct))} high  {YLW(str(medium_ct))} medium\n")

if det_report.anomalies:
    for a in det_report.anomalies:
        sev_tag = RED(f"[{a.severity.upper():6}]") if a.severity == "high" \
                  else YLW(f"[{a.severity.upper():6}]")
        print(f"         {sev_tag}  {B(a.type)}  {DIM(f'step {a.step_index}')}")
        sub(a.description[:90])
else:
    ok("No anomalies — nothing suspicious detected")

time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — STATIC REPLAY  (no LLM, no live calls)
# ══════════════════════════════════════════════════════════════════════════════
section(4, "Static Replay  (deterministic, Mock Injector guarantee)")
info("Replaying all steps with recorded responses.")
info(f"Side-effect tools blocked by AI classifier: {sorted(SIDE_EFFECTS)}")
print()

engine = ReplayEngine(recorder, side_effect_tools=SIDE_EFFECTS)
replay = engine.replay(run_id)

ok(f"Steps replayed : {replay.steps_replayed}")
ok(f"Live calls     : {GRN('0')}  (Mock Injector — no live endpoint reachable)")

blocked = [
    s for s in replay.outputs
    if str((s.get("output_data") or {}).get("output", "")).startswith("[MOCK-BLOCKED]")
]
if blocked:
    warn(f"Side-effect blocks: {len(blocked)}")
    for b in blocked:
        sub(f"step {b['step_index']}: {b.get('name','?')}  → {YLW('[MOCK-BLOCKED]')}")

if replay.divergence_detected:
    warn(f"Divergence at step {replay.divergence_step}: {replay.divergence_details}")
else:
    ok("Divergence check: exact match with original")

time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — AI ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section(5, "AI Root Cause Analysis  (RootCauseSubAgent, temperature=0)")
info("Dedicated LLM subagent diagnoses the trace.")
info("Few-shot context from similar past runs via LCS similarity.")
print()

store   = PatternStore(recorder)
similar = store.find_similar(run_id, top_k=3)
if similar:
    info(f"Similar runs found: {', '.join(f'{r.run_id[:16]}… ({r.similarity_score:.0%})' for r in similar)}")
else:
    info("No similar past runs — first occurrence of this pattern")

error_hint = det_report.summary if det_report.anomalies else ""
ai = RootCauseSubAgent()
ai_report = ai.analyze(run_id, steps, error_hint=error_hint,
                       similar_runs_context=store.format_for_context(similar))
print()
print(f"         Category   : {CYN(ai_report.category)}")
print(f"         Confidence : {B(f'{ai_report.confidence:.0%}')}")
print(f"         Failed step: {ai_report.failed_step}")
print(f"         Variable   : {ai_report.failed_variable}")
print(f"         Root cause : {ai_report.root_cause}")
print(f"         Fix        : {GRN(ai_report.suggested_fix[:80])}")
if ai_report.is_known_pattern:
    info(f"Known pattern: {ai_report.pattern_note}")
time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — DIVERGENCE REPLAY  (real LLM + patched step)
# ══════════════════════════════════════════════════════════════════════════════
section(6, "Divergence Replay  (inject fix → watch agent re-reason)")

can_diverge = (
    det_report.is_silent_failure
    and ai_report.failed_step is not None
    and any(s["step_index"] == ai_report.failed_step and s["step_type"] == "tool_call"
            for s in steps)
)

if not can_diverge:
    if not det_report.is_silent_failure:
        info("Run was healthy — no fix to inject. Showing manual usage instead:")
    else:
        info(f"No tool_call at step {ai_report.failed_step}. Manual usage:")
    print()
    print(f"  {DIM('Patch a tool result:')}")
    print(f"  python agent/jira_triage.py . --diverge {run_id} \\")
    print(f"    --patch-step <N> --patch-value '<json>'")
    print()
    print(f"  {DIM('Patch the initial prompt:')}")
    print(f"  python agent/jira_triage.py . --diverge {run_id} \\")
    print(f"    --patch-prompt 'New instruction — investigate only, no notifications'")
else:
    target_step = next(s for s in steps
                       if s["step_index"] == ai_report.failed_step
                       and s["step_type"] == "tool_call")
    tool_name  = target_step.get("name", "unknown")
    fix_value  = ai_report.suggested_fix

    info(f"Patching step {ai_report.failed_step} ({B(tool_name)}) with AI-suggested fix")
    info(f"Fix: {DIM(fix_value[:72])}")

    # Execute the fix locally to get real data
    corrected_result = fix_value
    if tool_name == "query_db" and fix_value.strip().upper().startswith("SELECT"):
        try:
            conn = sqlite3.connect(JIRA_DB)
            cur  = conn.execute(fix_value.strip())
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            conn.close()
            corrected_result = json.dumps([dict(zip(cols, r)) for r in rows])
            ok(f"Fix executed locally → {len(rows)} row(s) returned")
            sub(corrected_result[:80])
        except Exception as e:
            warn(f"Could not execute locally ({e}) — injecting raw SQL as value")

    print()
    info("Re-running agent with real LLM + patched step…")
    div_run_id = run_divergence(
        run_id,
        patch_step=ai_report.failed_step,
        patch_value=corrected_result,
    )
    print()

    if div_run_id:
        div_steps  = recorder.get_steps(div_run_id)
        orig_calls = [s.get("name","?") for s in steps     if s["step_type"] == "tool_call"]
        div_calls  = [s.get("name","?") for s in div_steps if s["step_type"] == "tool_call"]

        changed = orig_calls != div_calls
        label   = f"{YLW('⤢ TRAJECTORY CHANGED')}" if changed else f"{GRN('= same trajectory')}"
        print(f"         {label}")
        print(f"         Original : {' → '.join(orig_calls) or '(none)'}")
        print(f"         Diverged : {' → '.join(div_calls)  or '(none)'}")

        if changed:
            added   = set(div_calls)  - set(orig_calls)
            removed = set(orig_calls) - set(div_calls)
            if added:   ok(f"New tool calls:     {', '.join(added)}")
            if removed: warn(f"Dropped tool calls: {', '.join(removed)}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
banner("PIPELINE COMPLETE")
print(f"  Run ID     : {B(run_id)}")
print(f"  Steps      : {len(steps)}")
print(f"  Anomalies  : {len(det_report.anomalies)}  ({det_report.summary.split('.')[0]})")
print(f"  Analysis   : {ai_report.category}  ·  {ai_report.confidence:.0%} confidence")
print()
print(f"  {DIM('Explore further (CLI):')}")
print(f"  python agent/jira_triage.py .  --list-steps  {run_id}")
print(f"  python agent/jira_triage.py .  --analyze     {run_id}")
print(f"  python agent/jira_triage.py .  --auto-fix    {run_id}")
print()
print(f"  {DIM('Open the dashboard:')}")
print(f"  uvicorn api.server:app --reload   →  http://localhost:8000")
print()
