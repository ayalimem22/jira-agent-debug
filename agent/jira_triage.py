"""
Jira Triage Agent — the agent under observation (cobaye).

This file contains the agent being monitored by SentinelTrace. The agent
itself is not the innovation — SentinelTrace's recorder, replay engine, and
AI Root Cause Analyzer are. The Jira agent is the subject of observation.

4 local tools replace Jira/Confluence/LDAP/SMTP so the demo runs with zero
external accounts. All data is served from fixtures/ (created by seed_db.py).
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

_REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)

# SentinelTrace instrumentation
sys.path.insert(0, str(Path(__file__).parent.parent))
from flight_recorder.ai_debugger import RootCauseSubAgent
from flight_recorder.anomaly_detector import SilentFailureDetector
from flight_recorder.pattern_store import PatternStore
from flight_recorder.recorder import FlightRecorder, FlightRecorderCallback
from flight_recorder.replay import ReplayEngine, ToolResponseQueue
from flight_recorder.side_effect_classifier import SideEffectClassifier

FIXTURES = Path(__file__).parent / "fixtures"
DB_PATH = FIXTURES / "tickets.db"
KB_PATH = FIXTURES / "kb_articles.json"
USERS_PATH = FIXTURES / "users.json"
NOTIFICATIONS_LOG = Path(__file__).parent.parent / "notifications.log"

recorder = FlightRecorder()

# Expected tools for a complete Jira triage run
JIRA_EXPECTED_TOOLS = ["search_kb", "query_db", "get_user_info", "send_notification"]


# ── Tool definitions ──────────────────────────────────────────────────────────
# These tools use local fixtures only. No external service is ever contacted.

@tool
def search_kb(query: str) -> str:
    """Search the knowledge base for articles relevant to a query."""
    if not KB_PATH.exists():
        return "ERROR: knowledge base not found — run seed_db.py first"
    articles = json.loads(KB_PATH.read_text())
    hits = [a for a in articles if query.lower() in a["content"].lower()]
    return json.dumps(hits[:3], ensure_ascii=False) if hits else "No articles found."


@tool
def query_db(sql: str) -> str:
    """Run a SELECT query against the tickets database. Only SELECT is permitted."""
    if not sql.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are permitted in this tool."
    if not DB_PATH.exists():
        return "ERROR: tickets.db not found — run seed_db.py first"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        conn.close()
        return json.dumps([dict(zip(cols, r)) for r in rows], ensure_ascii=False)
    except Exception as exc:
        return f"DB ERROR: {exc}"


@tool
def get_user_info(user_id: str) -> str:
    """Retrieve a user's profile from the directory by their user ID."""
    if not USERS_PATH.exists():
        return "ERROR: users.json not found — run seed_db.py first"
    users = json.loads(USERS_PATH.read_text())
    match = next((u for u in users if u["id"] == user_id), None)
    return json.dumps(match, ensure_ascii=False) if match else f"User '{user_id}' not found."


@tool
def send_notification(input: str) -> str:
    """
    Send a notification to a user. Input must be JSON with keys: user_id, subject, body.
    Example: {"user_id": "u_alice", "subject": "Ticket update", "body": "Please review PROD-2847."}

    SIDE EFFECT — SentinelTrace hard-blocks this tool during all replay and
    simulation runs at the Mock Injector layer. This function body is never
    reached during simulation, regardless of configuration.
    """
    try:
        data = json.loads(input)
        user_id = data["user_id"]
        subject = data["subject"]
        body = data["body"]
    except (json.JSONDecodeError, KeyError) as e:
        return f"ERROR: input must be JSON with user_id, subject, body. Got: {e}"
    entry = {"user_id": user_id, "subject": subject, "body": body}
    with NOTIFICATIONS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Notification sent to {user_id}: '{subject}'"


TOOLS = [search_kb, query_db, get_user_info, send_notification]

# ── AI side-effect classifier ─────────────────────────────────────────────────

_classifier_cache: frozenset | None = None


def _get_side_effects() -> frozenset:
    """
    Use the AI classifier to determine which tools have irreversible side effects.

    Called once; result is cached for the session. This is the mechanism that
    makes SentinelTrace generic: no hardcoded list, no manual annotation.
    The LLM reads tool descriptions and decides what must be blocked in replay.
    """
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    print("\n[SentinelTrace] AI Tool Safety Classifier running…")
    classifier = SideEffectClassifier()
    report = classifier.classify(TOOLS)

    if report.used_fallback:
        print(f"  [warn] LLM unavailable — fallback set: {sorted(report.side_effect_tools)}")
    else:
        print(f"  Confidence        : {report.confidence:.0%}")
        print(f"  Side-effect tools : {sorted(report.side_effect_tools)}")
        print(f"  Safe tools        : {sorted(report.safe_tools)}")
        for name, reason in report.reasoning.items():
            if name != "_error":
                marker = "⚠" if name in report.side_effect_tools else "✓"
                print(f"    {marker} {name}: {reason}")

    _classifier_cache = report.side_effect_tools
    return _classifier_cache


def build_divergence_tools(queue: ToolResponseQueue) -> list:
    """
    Mock versions of all 4 tools backed by a ToolResponseQueue.

    During live divergence replay, the agent calls these instead of the real
    tools. Recorded responses are returned in order, except at the patched
    position where the injected value is returned. The LLM then reasons with
    this new data and may take a different trajectory.
    """
    @tool
    def search_kb(query: str) -> str:  # noqa: F811
        """Search the knowledge base for articles relevant to a query."""
        return queue.pop("search_kb")

    @tool
    def query_db(sql: str) -> str:  # noqa: F811
        """Run a SELECT query against the tickets database."""
        return queue.pop("query_db")

    @tool
    def get_user_info(user_id: str) -> str:  # noqa: F811
        """Retrieve a user profile from the directory."""
        return queue.pop("get_user_info")

    @tool
    def send_notification(input: str) -> str:  # noqa: F811
        """Send notification — BLOCKED in divergence replay."""
        return queue.pop("send_notification")

    return [search_kb, query_db, get_user_info, send_notification]


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent(temperature: float = 0.2) -> AgentExecutor:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=temperature)
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True, handle_parsing_errors=True
    )


# ── SentinelTrace-instrumented run ────────────────────────────────────────────

def run_live(ticket_id: str) -> str:
    """Run the agent live. Every call is recorded by FlightRecorderCallback."""
    # Classify tools before running — AI determines which ones are safe to mock
    _get_side_effects()
    run_id = recorder.start_run("jira_triage", ticket_id)
    cb = FlightRecorderCallback(recorder, run_id)
    executor = build_agent()
    try:
        executor.invoke(
            {
                "input": (
                    f"Triage ticket {ticket_id}. "
                    f"Search the knowledge base for relevant policies, "
                    f"query the database for ticket details, "
                    f"look up the assignee, and send them a professional notification."
                )
            },
            config={"callbacks": [cb]},
        )
        recorder.end_run(run_id, "completed")
    except Exception as exc:
        recorder.end_run(run_id, "failed")
        print(f"[SentinelTrace] Run failed: {exc}")

    # Automatic silent failure scan after every run
    steps = recorder.get_steps(run_id)
    detector = SilentFailureDetector(expected_tools=JIRA_EXPECTED_TOOLS)
    det_report = detector.detect(run_id, steps)
    if det_report.is_silent_failure:
        print(f"\n[SentinelTrace] ⚠ SILENT FAILURE DETECTED (confidence {det_report.confidence:.0%})")
        for a in det_report.anomalies:
            print(f"  [{a.severity.upper()}] {a.type} — {a.description}")
        print(f"  → Run --analyze {run_id} for AI root cause diagnosis")
    else:
        print(f"[SentinelTrace] Silent failure scan: OK")

    print(f"\n[SentinelTrace] Run recorded: {run_id}")
    print(f"  Replay : python agent/jira_triage.py {ticket_id} --replay {run_id}")
    print(f"  Analyze: python agent/jira_triage.py {ticket_id} --analyze {run_id}")
    return run_id


def run_replay(run_id: str) -> None:
    """Replay a recorded run. Guaranteed: no live calls, no side effects."""
    print(f"\n[SentinelTrace] Replaying run {run_id} …")
    engine = ReplayEngine(recorder, side_effect_tools=_get_side_effects())
    result = engine.replay(run_id)
    print(f"  Steps replayed     : {result.steps_replayed}")
    print(f"  Divergence detected: {result.divergence_detected}")
    if result.divergence_detected:
        print(f"  At step            : {result.divergence_step}")
        print(f"  Details            : {result.divergence_details}")
    blocked = [
        s for s in result.outputs
        if str((s.get("output_data") or {}).get("output", "")).startswith("[MOCK-BLOCKED]")
    ]
    print(f"  Side-effect blocks : {len(blocked)}")
    for b in blocked:
        print(f"    step {b['step_index']}: {b.get('name', '?')} → BLOCKED")
    log_lines = NOTIFICATIONS_LOG.read_text().splitlines() if NOTIFICATIONS_LOG.exists() else []
    print(f"\n  notifications.log  : {len(log_lines)} entries (UNCHANGED by replay)")


def auto_remediate(run_id: str) -> None:
    """
    Full auto-remediation loop for silent failures caused by bad tool responses.

    Flow:
      1. Detect — SilentFailureDetector finds EmptyToolResult / IgnoredToolResult
      2. Diagnose — RootCauseSubAgent identifies the bad step and suggests a SQL fix
      3. Repair — extract corrected SQL from the suggestion, execute it locally
      4. Verify — divergence replay with real result → agent takes new trajectory
    """
    print(f"\n[SentinelTrace] Auto-Remediation — run {run_id}")
    steps = recorder.get_steps(run_id)
    if not steps:
        print("  ERROR: no steps found")
        return

    # ── 1. Detect ─────────────────────────────────────────────────────────────
    from flight_recorder.anomaly_detector import AnomalyType, SilentFailureDetector
    detector = SilentFailureDetector(expected_tools=JIRA_EXPECTED_TOOLS)
    det_report = detector.detect(run_id, steps)

    empty_anomalies = [a for a in det_report.anomalies
                       if a.type == AnomalyType.EMPTY_TOOL_RESULT]
    if not empty_anomalies:
        print("  No empty/error tool results detected — try --analyze for other failure types.")
        return

    print(f"  Found {len(empty_anomalies)} empty tool result(s):")
    for a in empty_anomalies:
        print(f"    [step {a.step_index}] {a.description[:100]}")

    # Priority: query_db errors first (most impactful), then others
    PRIORITY = ["query_db", "get_user_info", "send_notification", "search_kb"]
    def anomaly_priority(a: "Anomaly") -> int:
        desc = a.description.lower()
        for i, tool in enumerate(PRIORITY):
            if f"'{tool}'" in desc:
                return i
        return len(PRIORITY)

    target_anomaly = sorted(empty_anomalies, key=anomaly_priority)[0]
    target_step = target_anomaly.step_index  # tool_result step

    # Find the corresponding tool_call step (step before the result)
    tool_call_step = next(
        (s for s in steps
         if s["step_index"] == target_step - 1 and s["step_type"] == "tool_call"),
        None,
    )
    if not tool_call_step:
        print("  ERROR: could not find the tool_call step preceding the empty result.")
        return

    tool_name = tool_call_step.get("name", "unknown")
    tool_input = str((tool_call_step.get("input_data") or {}).get("input", ""))
    print(f"\n  Targeting: {tool_name} at step {tool_call_step['step_index']}")
    print(f"  Original input: {tool_input[:120]}")

    # ── 2. Diagnose ───────────────────────────────────────────────────────────
    store = PatternStore(recorder)
    similar = store.find_similar(run_id, top_k=3)
    context = store.format_for_context(similar)

    subagent = RootCauseSubAgent()
    report = subagent.analyze(
        run_id, steps,
        error_hint=f"Empty/error result from {tool_name} at step {target_step}. Fix the query.",
        similar_runs_context=context,
    )
    print(f"\n  AI diagnosis:")
    print(f"    Category      : {report.category}")
    print(f"    Root cause    : {report.root_cause}")
    print(f"    Suggested fix : {report.suggested_fix}")

    # ── 3. Repair ─────────────────────────────────────────────────────────────
    corrected_result = None

    if tool_name == "query_db":
        # Extract SELECT query from the suggested fix
        # Match a full SELECT statement — stop at newline, backtick, or end of string
        sql_match = re.search(r"(SELECT\s+[^\n`]+)", report.suggested_fix, re.IGNORECASE)
        corrected_sql = sql_match.group(1).strip().rstrip("'\".,") if sql_match else None

        if not corrected_sql:
            # Fallback: ask the subagent for just the SQL
            from langchain_core.messages import HumanMessage
            raw = subagent.llm.invoke([HumanMessage(content=(
                f"The original SQL query failed: {tool_input}\n"
                f"Write ONLY the corrected SELECT query for the tickets table "
                f"(columns: id, category, summary, status, assignee, priority). "
                f"No explanation, just the SQL."
            ))])
            corrected_sql = raw.content.strip().strip("`").strip()

        print(f"\n  Corrected SQL: {corrected_sql}")

        # Execute the corrected SQL against the real local DB
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.execute(corrected_sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            conn.close()
            corrected_result = json.dumps(
                [dict(zip(cols, r)) for r in rows], ensure_ascii=False
            )
            print(f"  Real result    : {corrected_result[:120]}")
        except Exception as exc:
            print(f"  SQL still invalid: {exc}")
            print("  Manual patch required — use --diverge with --patch-value")
            return

    elif tool_name == "search_kb":
        # Re-run search_kb with a broader/corrected query derived from the fix
        corrected_query = report.suggested_fix[:80]
        articles = json.loads(KB_PATH.read_text())
        hits = [a for a in articles
                if any(w in a["content"].lower() for w in corrected_query.lower().split()[:5])]
        corrected_result = json.dumps(hits[:3], ensure_ascii=False) if hits else "No articles found."
        print(f"  KB re-query result: {corrected_result[:120]}")

    else:
        print(f"  Auto-repair not supported for '{tool_name}' — use --diverge manually.")
        return

    # ── 4. Verify — divergence replay with the real corrected result ──────────
    print(f"\n  Launching divergence replay with corrected result …")
    run_divergence(run_id, patch_step=tool_call_step["step_index"], patch_value=corrected_result, patch_prompt=None)


def list_steps(run_id: str) -> None:
    """Print all steps so the user can identify which step_index to patch."""
    steps = recorder.get_steps(run_id)
    if not steps:
        print(f"No steps found for run {run_id}")
        return
    print(f"\n[SentinelTrace] Steps for run {run_id} ({len(steps)} total)\n")
    print(f"  {'Step':>4}  {'Type':<15}  {'Tool/Model':<22}  Preview")
    print(f"  {'-'*4}  {'-'*15}  {'-'*22}  {'-'*45}")
    for s in steps:
        out = (s.get("output_data") or {}).get("output", "")
        preview = str(out)[:45].replace("\n", " ")
        name = (s.get("name") or "")[:22]
        marker = " ◄ patch here?" if s["step_type"] == "tool_call" else ""
        print(f"  {s['step_index']:>4}  {s['step_type']:<15}  {name:<22}  {preview}{marker}")
    print(f"\n  Patch a tool result:")
    print(f"  python agent/jira_triage.py . --diverge {run_id} --patch-step <N> --patch-value '<json>'")
    print(f"\n  Patch the initial prompt:")
    print(f"  python agent/jira_triage.py . --diverge {run_id} --patch-prompt 'new instruction here'")


def run_divergence(
    run_id: str,
    patch_step: int | None = None,
    patch_value: str = "",
    patch_prompt: str | None = None,
) -> str:
    """
    Live divergence replay — re-runs the agent with real LLM + mock tools.

    Two patching modes (can be combined):
    - Tool result patch : tool calls return recorded responses, except at
      patch_step where patch_value is injected. The LLM re-reasons with
      the new data and may take a different trajectory.
    - Prompt patch      : the initial instruction to the agent is replaced
      with patch_prompt. All tool calls still return recorded responses
      (mocked). Used to observe how a different instruction changes the
      agent's decisions without touching live systems.
    """
    print(f"\n[SentinelTrace] Live Divergence Replay")
    print(f"  Original run   : {run_id}")
    if patch_prompt is not None:
        print(f"  Prompt patch   : {patch_prompt[:100]}")
    if patch_step is not None:
        print(f"  Patching step  : {patch_step}")
        print(f"  Injected value : {patch_value[:80]}")

    original_steps = recorder.get_steps(run_id)
    if not original_steps:
        print("  ERROR: no steps found for this run ID")
        return ""

    # Recover original agent input from the first LLM prompt
    original_input = next(
        (
            s["input_data"].get("prompts", [""])[0]
            for s in original_steps
            if s["step_type"] == "llm_start" and s.get("input_data")
        ),
        "Triage the assigned ticket.",
    )

    # Build mock tools backed by the recorded responses + patch
    queue = ToolResponseQueue(
        original_steps,
        patch_step_index=patch_step,
        patch_value=patch_value if patch_step is not None else None,
        side_effect_tools=_get_side_effects(),
    )
    mock_tools = build_divergence_tools(queue)

    # New run ID for the divergence trajectory
    div_run_id = recorder.start_run(
        "jira_triage_diverge", f"diverge:{run_id}@step{patch_step}"
    )
    cb = FlightRecorderCallback(recorder, div_run_id)

    # Real LLM + mock tools — the agent re-reasons with patched data
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.2)
    agent = create_react_agent(llm, mock_tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=mock_tools, verbose=True, handle_parsing_errors=True
    )

    effective_input = patch_prompt if patch_prompt is not None else original_input
    print(f"\n  Re-running agent with real LLM…")
    print(f"  Input: {effective_input[:120]}\n")
    try:
        executor.invoke({"input": effective_input}, config={"callbacks": [cb]})
        recorder.end_run(div_run_id, "completed")
    except Exception as exc:
        recorder.end_run(div_run_id, "failed")
        print(f"  [warn] {exc}")

    # Compare trajectories
    div_steps = recorder.get_steps(div_run_id)
    orig_calls = [s.get("name", "?") for s in original_steps if s["step_type"] == "tool_call"]
    div_calls  = [s.get("name", "?") for s in div_steps if s["step_type"] == "tool_call"]
    q = queue.summary()

    print(f"\n{'─'*60}")
    print(f"[SentinelTrace] Divergence Result")
    print(f"  Original run       : {run_id} ({len(original_steps)} steps)")
    print(f"  Divergence run     : {div_run_id} ({len(div_steps)} steps)")
    print(f"  Step count delta   : {len(div_steps) - len(original_steps):+d}")

    if patch_prompt is not None:
        print(f"\n  Prompt change:")
        print(f"    Original : {original_input[:100]}")
        print(f"    Patched  : {patch_prompt[:100]}")

    if patch_step is not None:
        print(f"  Tool patch at      : {q['patch_tool']} call #{q['patch_call_position']} (step {patch_step})")

    print(f"\n  Original path  : {' → '.join(orig_calls) or '(none)'}")
    print(f"  Diverged path  : {' → '.join(div_calls) or '(none)'}")
    trajectory_changed = orig_calls != div_calls
    print(f"\n  Trajectory changed : {'YES ✓' if trajectory_changed else 'NO (same decisions)'}")
    print(f"\n  Analyze divergence run:")
    print(f"  python agent/jira_triage.py . --analyze {div_run_id}")
    return div_run_id


def run_analyze(run_id: str, hint: str = "") -> None:
    """Run the AI Root Cause Analyzer with cross-run pattern context."""
    print(f"\n[SentinelTrace] AI Root Cause Analysis — run {run_id} …")
    steps = recorder.get_steps(run_id)
    if not steps:
        print("  ERROR: no steps found for this run ID")
        return

    # Silent failure scan — enriches the hint if anomalies are found
    detector = SilentFailureDetector(expected_tools=JIRA_EXPECTED_TOOLS)
    det_report = detector.detect(run_id, steps)
    if det_report.anomalies:
        anomaly_summary = "; ".join(f"{a.type}@step{a.step_index}" for a in det_report.anomalies)
        hint = f"{hint} [auto-detected: {anomaly_summary}]".strip()
        print(f"  Silent failures    : {len(det_report.anomalies)} detected — {det_report.summary}")
    else:
        print(f"  Silent failures    : none detected")

    # Find similar past runs and build few-shot context
    store = PatternStore(recorder)
    similar = store.find_similar(run_id, top_k=3)
    context = store.format_for_context(similar)
    if similar:
        print(f"  Similar runs found : {len(similar)} "
              f"({', '.join(f'{r.run_id}={r.similarity_score:.0%}' for r in similar)})")
    else:
        print(f"  Similar runs found : 0 (new pattern)")

    subagent = RootCauseSubAgent()
    report = subagent.analyze(run_id, steps, error_hint=hint, similar_runs_context=context)

    print(f"\n  Category       : {report.category}")
    print(f"  Confidence     : {report.confidence:.0%}")
    print(f"  Known pattern  : {'YES — ' + report.pattern_note if report.is_known_pattern else 'NO (first occurrence)'}")
    print(f"  Failed step    : {report.failed_step}")
    print(f"  Failed var     : {report.failed_variable}")
    print(f"  Root cause     : {report.root_cause}")
    print(f"  Suggested fix  : {report.suggested_fix}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jira Triage Agent with SentinelTrace instrumentation"
    )
    parser.add_argument("ticket_id", help="Ticket ID to triage (e.g. PROD-2847), or '.' for replay/diverge commands")
    parser.add_argument("--replay",      metavar="RUN_ID", help="Static replay — no LLM, no live calls")
    parser.add_argument("--analyze",     metavar="RUN_ID", help="AI root cause analysis")
    parser.add_argument("--diverge",     metavar="RUN_ID", help="Live divergence replay with real LLM")
    parser.add_argument("--list-steps",  metavar="RUN_ID", help="List all steps of a run")
    parser.add_argument("--patch-step",   type=int, default=None, help="Step index to patch during divergence")
    parser.add_argument("--patch-value",  default="", help="Value to inject at the patched step")
    parser.add_argument("--patch-prompt", default=None, help="Replace the initial prompt during divergence replay")
    parser.add_argument("--auto-fix",    metavar="RUN_ID", help="Auto-remediate: detect → diagnose → fix → replay")
    parser.add_argument("--hint",        default="", help="Optional hint for the analyzer")
    args = parser.parse_args()

    if args.auto_fix:
        auto_remediate(args.auto_fix)
    elif args.list_steps:
        list_steps(args.list_steps)
    elif args.replay:
        run_replay(args.replay)
    elif args.analyze:
        run_analyze(args.analyze, hint=args.hint)
    elif args.diverge:
        has_tool_patch   = args.patch_step is not None and args.patch_value
        has_prompt_patch = args.patch_prompt is not None
        if not has_tool_patch and not has_prompt_patch:
            print("ERROR: --diverge requires at least one of:")
            print("  --patch-prompt '<new instruction>'")
            print("  --patch-step <N> --patch-value '<json>'  (use --list-steps to find N)")
        else:
            run_divergence(
                args.diverge,
                patch_step=args.patch_step,
                patch_value=args.patch_value,
                patch_prompt=args.patch_prompt,
            )
    else:
        run_live(args.ticket_id)
