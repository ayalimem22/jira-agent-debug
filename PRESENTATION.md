# SentinelTrace — AINS Hackathon 2026
## Jury Presentation · 10 Slides

---

## SLIDE 1 — Title

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   S E N T I N E L T R A C E                                 ║
║                                                              ║
║   Agent Execution Tracer with Deterministic Replay Engine    ║
║                                                              ║
║   "When your AI agent breaks in production,                  ║
║    don't restart it. Replay it."                             ║
║                                                              ║
║   Use Case 2 — Enterprise Automation                         ║
║   AINS Hackathon 2026 · First Submission                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Tagline:** *Record. Replay. Diagnose. Self-Heal.*

---

## SLIDE 2 — The Problem

### The incident that triggered this project

> Your Jira Triage Agent ran at 2:00 AM.
> A customer called at 9:00 AM — furious.
> The agent had sent them an aggressive automated email.
> Your on-call engineer types: `python agent.py --rerun PROD-2847`

**Three things just went wrong:**

| Action | Consequence |
|--------|-------------|
| Reran the agent | LLM produced a different output — the bug may not reproduce |
| Tool calls re-fired | A second email may have been sent |
| Original trace lost | You can never know what the agent actually decided at 2:00 AM |

### The silent failure problem

Even worse: **most agent failures raise no exception at all.**
The agent returns "Done." It called 2 of 4 required tools, hallucinated the rest,
and no alert fired.

**Standard observability (LangSmith / Langfuse / Phoenix) gives you logs. It does not give you:**
- Replay without re-running
- Automatic detection of wrong-but-no-exception failures
- A way to test a fix before applying it

---

## SLIDE 3 — Why Existing Tools Fall Short

```
                    LangSmith    Langfuse    Phoenix    SentinelTrace
                    ─────────    ────────    ───────    ─────────────
Trace logging           ✓           ✓          ✓            ✓
Deterministic replay    ✗           ✗          ✗            ✓
Side-effect blocking    ✗           ✗          ✗            ✓  (enforced)
Silent failure detect   ✗           ✗          ✗            ✓  (7 checks)
AI root cause analysis  ✗           ✗          ✗            ✓  (subagent)
Cross-run learning      ✗           ✗          ✗            ✓  (LCS + few-shot)
Auto-remediation loop   ✗           ✗          ✗            ✓  (detect→fix→replay)
Tamper-proof audit      ✗           ✗          ✗            ✓  (HMAC-SHA256)
```

**The gap:** existing tools are *read-only mirrors*. SentinelTrace is an *active debugging engine*.

---

## SLIDE 4 — The SentinelTrace Approach

### One rule changes everything

```
┌─────────────────────────────────────────────────────────────────┐
│  WHEN A BUG OCCURS: NEVER RESTART THE LIVE AGENT.              │
│  Simulate only — on the recorded trace.                         │
│  The Mock Injector makes this a guarantee, not a convention.    │
└─────────────────────────────────────────────────────────────────┘
```

### Three principles

**1. Record everything, trust nothing.**
Every LLM call, tool call, token count, and context snapshot is persisted with
HMAC-SHA256 signatures. The trace is WORM — it cannot be altered after the fact.

**2. Replay without re-running.**
The Mock Injector intercepts all tool calls during simulation. There is no code
path that reaches a live endpoint. Side-effect tools (`send_notification`) are
hard-blocked *before* any recorded data is consulted.

**3. Diagnose with a dedicated AI.**
A separate LLM subagent receives the full trace as context and returns a
structured diagnosis. Separation of concerns — inspired by Microsoft Research's
Debug2Fix (ACM Feb 2026): +21% resolution rate vs. self-debugging.

---

## SLIDE 5 — Architecture: 5 Layers

```
┌─────────────────────────────────────────────────────────┐
│ L1  AGENT LAYER                                         │
│     LangChain AgentExecutor · 4 tools · LLM · Run ID   │
├─────────────────────────────────────────────────────────┤
│ L2  PROXY & INTERCEPT                                   │
│     FlightRecorderCallback · SilentFailureDetector      │
│     PatternStore · Context Snapshotter                  │
├─────────────────────────────────────────────────────────┤
│ L3  STORAGE                                             │
│     SQLite steps · BLOB snapshots · HMAC-SHA256 vault   │
├─────────────────────────────────────────────────────────┤
│ L4  REPLAY ENGINE                                       │
│     ReplayEngine · MockInjector · DivergenceEngine      │
│     ToolResponseQueue (live re-run with patch)          │
├─────────────────────────────────────────────────────────┤
│ L5  OBSERVABILITY                                       │
│     FastAPI REST · Trace Timeline · Compliance Export   │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** L4 is the innovation. The replay engine is not a log viewer —
it is an active simulation harness that can re-run the LLM with a surgical patch
at any step and observe the new trajectory.

---

## SLIDE 6 — The Two AI Mechanisms

SentinelTrace uses two dedicated LLM subagents — both run at temperature=0.

### AI Mechanism 1 — SideEffectClassifier

Runs before every live agent run. Reads each tool's description and decides
which ones have irreversible side effects. No hardcoded list, no manual annotation.

```
Input  : list of tools with names + descriptions
Output : {"side_effect_tools": ["send_notification"],
          "safe_tools": ["query_db", "search_kb", "get_user_info"],
          "reasoning": {"send_notification": "Sends real messages — cannot be undone"},
          "confidence": 0.98}
```

**Why this matters:** without it, SentinelTrace needs a per-agent hardcoded blocklist.
With it, SentinelTrace is **generic** — it can safely replay any unknown agent.
Falls back to a default set if the LLM is unavailable — safety never silently dropped.

---

### AI Mechanism 2 — RootCauseSubAgent

Runs after a failure is detected. Receives the full serialized trace + few-shot
context from similar past runs (PatternStore). Returns a structured JSON diagnosis.

```json
{
  "root_cause":       "Step 7: query_db returned DB ERROR — SQL used unquoted string ID",
  "failed_step":      7,
  "failed_variable":  "sql",
  "suggested_fix":    "SELECT * FROM tickets WHERE id = 'DB-1193'",
  "category":         "RootCause",
  "confidence":       0.87,
  "is_known_pattern": true,
  "pattern_note":     "Same SQL error seen in run b7c3a1f2 (similarity: 0.85)"
}
```

**Why a dedicated subagent?** Microsoft Research (Debug2Fix, ACM Feb 2026):
dedicated debugging subagent improves resolution rate by **+21%** vs. self-debugging.
Separation of concerns — one agent acts, one agent diagnoses.

---

## SLIDE 7 — Live Divergence Replay

### The problem it solves

Static replay validates that the Mock Injector works.
But it answers the wrong question: *"did the simulation match the original?"*

The right question is: *"if I fix step N, does the agent recover?"*

### How it works

```
Recorded trace
  steps 0 → 5 : returned as-is (mock)
  step 6      : YOUR PATCH (correct SQL result)   ← injected here
  steps 7 → ? : LLM re-runs and reasons freely

AgentExecutor receives the patched data as if it were real.
The LLM does not know it is in a test.
The new trajectory is recorded as a new run.
```

### What you observe

| | Original run | Diverged run |
|-|-------------|-------------|
| Steps | 8 (stopped at DB error) | 14 (continued to completion) |
| query_db result | DB ERROR | 1 row returned |
| get_user_info called | NO | YES |
| send_notification | NO | BLOCKED (safe) |
| Trajectory changed | — | YES |

**The LLM reasoned correctly with the fixed data — without restarting anything.**

---

## SLIDE 8 — Silent Failure Detection

### The hardest failures to catch

An agent that raises an exception is easy to notice.
An agent that calls 2 of 4 required tools and returns "I've processed the ticket"
is invisible to standard monitoring.

### 7 heuristic checks — zero LLM, zero cost, runs automatically

| # | Anomaly | Severity | Example |
|---|---------|----------|---------|
| 1 | `MissingToolCall` | HIGH | `send_notification` never called for a critical ticket |
| 2 | `ToolLoop` | HIGH | `search_kb` called 5 times — stuck in a reasoning loop |
| 3 | `EmptyToolResult` | HIGH | `query_db` returned `DB ERROR` — agent continued with hallucinated data |
| 4 | `IgnoredToolResult` | MEDIUM | Tool returned ticket data, next LLM output references none of it |
| 5 | `PrematureTermination` | MEDIUM | Final Answer after only 1 tool call (expected 4) |
| 6 | `UncertainCompletion` | MEDIUM | Final Answer: "I think the ticket has been processed..." |
| 7 | `HallucinationSignal` | HIGH | Final Answer references `u_carol` — who appears in no tool result |

**Auto-triggers `--auto-fix` pipeline when `EmptyToolResult` on `query_db` detected.**

---

## SLIDE 9 — Live Demo: 3 Scenarios

### Scenario A — Side-Effect Trap (PROD-2847)

1. Agent runs live, sends aggressive email
2. `--replay`: Mock Injector blocks `send_notification` — `notifications.log` line count unchanged
3. `--analyze`: RootCauseSubAgent → `category: SideEffect, suggested_fix: add tone-check`

**Jury sees:** the log file is physically unchanged after replay. Guarantee enforced, not configured.

---

### Scenario B — Malformed SQL + Live Divergence (DB-1193)

1. Agent runs live, `query_db` returns DB ERROR, agent stops at step 7
2. `--list-steps`: trace table with `<< patch here?` marker at step 6
3. `--diverge --patch-step 6 --patch-value '[...]'`: agent continues, 6 new steps appear
4. `--analyze`: `suggested_fix: "SELECT * FROM tickets WHERE id = 'DB-1193'"`

**Jury sees:** two run records in the DB — original (8 steps) and diverged (14 steps).

---

### Scenario D — Auto-Remediation (one command)

```
python agent/jira_triage.py . --auto-fix a3557394

[1/4] Detect  → EmptyToolResult for query_db (HIGH)
[2/4] Diagnose → "Change SQL to SELECT * FROM tickets WHERE id = 'DB-1193'"
[3/4] Execute → 1 row returned
[4/4] Replay  → Trajectory changed: YES
```

**Jury sees:** bug detected, diagnosed, corrected, and validated in a single command with no human intervention.

---

## SLIDE 10 — Value Proposition & What's Next

### What SentinelTrace delivers today

| Problem | SentinelTrace answer |
|---------|---------------------|
| Can't replay without re-running | Deterministic replay — zero live calls, WORM audit trail |
| Side effects fire during debug | Hard-blocked at proxy layer — architectural, not configurable |
| Agent fails silently | 7-check heuristic detector on every run, zero LLM cost |
| Don't know which step failed | RootCauseSubAgent — structured JSON diagnosis, 5 categories |
| Same bugs keep recurring | PatternStore — LCS similarity, few-shot context injection |
| Manual fix-test loop | Auto-remediation — detect → diagnose → fix → validate in 1 command |

### Acceptance criteria — all satisfied

Deterministic replay · State inspection · Divergence editing ·
AI root cause analysis · Cross-run learning · Silent failure detection ·
Auto-remediation · Compliance audit · Side-effect safety

### Next steps

- **Multi-agent support:** record and replay chains of agents, not just single executors
- **Production deployment:** async FastAPI workers + Redis for high-volume trace ingestion
- **LLM-agnostic:** `ChatOllama` already supported — zero-cost local debugging
- **Anomaly alerting:** webhook dispatch when `is_silent_failure = true`
- **Pattern library:** shared repo of known failure signatures across teams

---

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   SentinelTrace does not observe your agents.             ║
║   It lets you rewind them, fix them, and prove it.        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```
