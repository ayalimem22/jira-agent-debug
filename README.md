# SentinelTrace

**Record every decision an AI agent makes. Replay it deterministically. Detect silent failures automatically. Let a dedicated AI subagent tell you exactly what broke and how to fix it — without ever restarting the agent.**

---

## The Problem

When an AI agent misbehaves in production, the standard instinct is to rerun it.
That instinct is wrong — and dangerous.

> *An agent that sent an aggressive email to a customer cannot be "debugged" by running it again.*
> *The LLM will produce a different output. New tool calls will fire. New side effects will happen.*

Existing observability tools — LangSmith, Langfuse, Phoenix — give you logs and traces.
They do not give you:
- A way to **replay a specific past run**, deterministically, without triggering any live system
- A way to **test a fix** before applying it, against the exact context that caused the failure
- **Automatic detection of silent failures** — runs that succeed without error but produce wrong results
- An **AI-powered diagnosis** of which step failed, why, and what to change
- A **self-healing loop** that detects, diagnoses, corrects, and validates a fix without human intervention

SentinelTrace was built to close every one of those gaps.

---

## The Core Rule

```
┌─────────────────────────────────────────────────────────────────────┐
│  WHEN A BUG OCCURS: NEVER RESTART THE LIVE AGENT.                  │
│  Simulate only — on the recorded trace.                             │
│  The Mock Injector makes this a guarantee, not a convention.        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Phase 1 — Record (live run)

Every LLM call, tool call, and context snapshot is intercepted at the proxy layer
and persisted to SQLite with HMAC-SHA256 signatures (tamper-proof audit vault).
Before the run starts, the **AI Side-Effect Classifier** reads tool descriptions
and determines which ones must be blocked during replay — no hardcoded list needed.

```
User request
     │
     ▼
┌────────────────────────────────────────────┐
│  SideEffectClassifier  (LLM, temp=0)       │
│  Reads tool descriptions → decides:        │
│    send_notification → BLOCK               │
│    query_db          → SAFE                │
└──────────────────┬─────────────────────────┘
                   │ classified set passed to MockInjector
                   ▼
┌────────────────────────────────────────────┐
│  Jira Triage Agent  (AgentExecutor)        │
│  ┌──────────┐  ┌──────┐  ┌──────────────┐ │
│  │ search_kb│  │ query│  │send_notif.   │ │
│  │          │  │  _db │  │(side-effect) │ │
│  └──────────┘  └──────┘  └──────────────┘ │
└──────────────────┬─────────────────────────┘
                   │  every call intercepted
                   ▼
┌────────────────────────────────────────────┐
│  FlightRecorderCallback (LangChain hook)   │
│  LLM Proxy · Tool Proxy · Anomaly Detector │
└────────────────────────────────────────────┘
                   │
                   ▼
        SQLite -- Snapshots -- Audit Vault
         (steps)   (BLOB)     (HMAC-SHA256)
```

After every live run, the **Silent Failure Detector** runs automatically
(zero LLM calls, zero cost) and reports any anomalies found.

### Phase 2a — Static Replay (simulation, no LLM)

The Replay Engine reads the recorded trace. The Mock Injector intercepts **all**
tool calls and returns the stored responses. There is no code path in simulation
mode that reaches a live endpoint. Side-effect tools are blocked using the
AI-classified set from Phase 1.

```
Recorded trace
     │
     ▼
┌────────────────────────────────────────────┐
│  ReplayEngine (static -- no LLM invoked)  │
│  ┌──────────────────────────────────────┐ │
│  │  MockInjector (AI-classified set)    │ │
│  │  · send_notification → BLOCKED       │ │
│  │  · All others        → [MOCK] data  │ │
│  └──────────────────────────────────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │  DivergenceEngine                    │ │
│  │  · Compares step-by-step             │ │
│  │  · Reports first deviation           │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

### Phase 2b — Live Divergence Replay (real LLM re-execution)

The engineer inspects the recorded steps, identifies the failure point, and
injects a corrected value. The agent **re-runs with the real LLM** — receiving
the patched data at the injection point and reasoning freely from there.
The new trajectory is recorded as a new run and compared to the original.

```
Recorded trace + patch at step N
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  ToolResponseQueue                                   │
│  steps 0 to N-1 : recorded responses (mock)         │
│  step N         : injected value  <<-- PATCH HERE    │
│  steps N+1 to ? : LLM decides freely                │
└──────────────────────────┬───────────────────────────┘
                           │ mock tools fed to real LLM
                           ▼
              AgentExecutor (real LLM re-run)
                           │
                ┌──────────┴──────────┐
                │  Original path      │  Diverged path
                │  search_kb          │  search_kb
                │  query_db -> ERROR  │  query_db -> OK  << patch
                │  (agent stops)      │  get_user_info
                │                     │  send_notification -> BLOCKED
                └─────────────────────┘
                           │
                           ▼
              New run recorded -> trajectory diff shown
```

**Key property:** the LLM is not told it is in a divergence test. It receives
the patched tool response as if it were real and reasons naturally.

### Phase 3 — Analyze (AI Root Cause Analyzer)

A dedicated LLM subagent receives the full execution trace and similar past runs
as context, then returns a structured diagnosis.

```
Recorded trace (all steps)  +  Similar-run context (PatternStore)
     │                               │
     └───────────────┬───────────────┘
                     ▼
┌────────────────────────────────────────────┐
│  RootCauseSubAgent  (dedicated LLM call)  │
│                                            │
│  Output:                                   │
│    root_cause    : plain language          │
│    suggested_fix : concrete action         │
│    category      : one of 5 types         │
│    confidence    : 0.0 - 1.0              │
│    is_known_pattern : true / false         │
└────────────────────────────────────────────┘
```

### Phase 4 — Silent Failure Detection (heuristic, zero LLM cost)

7 heuristic checks run automatically after every live run — no LLM, no cost.

```
Recorded steps
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  SilentFailureDetector                                 │
│                                                        │
│  1 MissingToolCall     -- expected tool never invoked  │
│  2 ToolLoop            -- same tool called 3+ times    │
│  3 EmptyToolResult     -- DB error / [] / null output  │
│  4 IgnoredToolResult   -- result not used by next LLM  │
│  5 PrematureTermination -- Final Answer too early       │
│  6 UncertainCompletion -- hedging language detected    │
│  7 HallucinationSignal -- Final Answer names unknown   │
│                           entities                     │
│                                                        │
│  Output: confidence score · anomaly list · is_silent   │
└────────────────────────────────────────────────────────┘
```

### Phase 5 — Cross-Run Pattern Learning (PatternStore)

LCS similarity on tool-call sequences finds structurally similar past failures
and injects them as few-shot context into the RootCauseSubAgent prompt.

### Phase 6 — Auto-Remediation (Detect → Diagnose → Fix → Validate)

```
Completed run
     │
     ▼
 SilentFailureDetector: EmptyToolResult for query_db?
     │ YES
     ▼
 RootCauseSubAgent: suggest corrected SQL
     │
     ▼
 Execute corrected SQL against local DB
     │
     ▼
 Live Divergence Replay: inject real result at failure step
     │
     ▼
 Report: original path vs new trajectory
```

---

## Architecture — 5 Layers

| Layer | Name | Components | Role |
|-------|------|-----------|------|
| L1 | Agent Layer | LangChain AgentExecutor, 4 tools, LLM, Run ID | The agent under observation |
| L2 | Proxy & Intercept | FlightRecorderCallback, **SideEffectClassifier**, **SilentFailureDetector**, **PatternStore** | Capture everything + AI-driven safety classification |
| L3 | Storage | SQLite (runs + steps), Snapshot Store (BLOB gzip), Audit Vault WORM (HMAC-SHA256) | Persist traces with tamper-proof integrity |
| L4 | Replay Engine | ReplayEngine, MockInjector, DivergenceEngine, ToolResponseQueue | Deterministic simulation — zero live calls |
| L5 | Observability | FastAPI REST, Trace Timeline, Prompt Inspector, Compliance Export | Inspect, patch, and export |

---

## The AI Mechanisms

> This section describes the AI contributions **of SentinelTrace itself**, not of the Jira agent it monitors.

SentinelTrace uses two dedicated LLM subagents.

### Mechanism 1 — SideEffectClassifier

An LLM (temperature=0) reads each tool's name and description and decides
whether it has irreversible side effects. No hardcoded list, no manual annotation.

This makes SentinelTrace **generic**: it can safely replay any agent without knowing
in advance which tools are dangerous.

```json
{
  "side_effect_tools": ["send_notification"],
  "safe_tools": ["search_kb", "query_db", "get_user_info"],
  "reasoning": {
    "send_notification": "Sends real messages to users — cannot be undone",
    "query_db": "Read-only SELECT — no state is modified"
  },
  "confidence": 0.98
}
```

Falls back to a hardcoded default set if the LLM is unavailable — the safety
guarantee is **never silently dropped**.

### Mechanism 2 — RootCauseSubAgent

A dedicated LLM subagent (temperature=0) receives the full serialized trace
and returns a structured JSON diagnosis. Inspired by Microsoft Research (Debug2Fix,
ACM Feb 2026): dedicated debugging subagents improve resolution rate by **+21%**.

**Five failure categories:**

| Category | What it means |
|----------|--------------|
| `Exception` | A tool or LLM call raised an unhandled error |
| `RootCause` | The agent produced wrong output due to bad reasoning or bad input |
| `VariableInspection` | A variable held an unexpected value that propagated through steps |
| `Divergence` | Replay output differs from original at a specific step |
| `SideEffect` | A side-effect tool was called when it should not have been |

**Output contract:**
```json
{
  "root_cause":        "Step 7: query_db returned DB ERROR — SQL used string ID format",
  "failed_step":       7,
  "failed_variable":   "sql",
  "failed_value":      "SELECT * FROM tickets WHERE id = DB-1193",
  "suggested_fix":     "SELECT * FROM tickets WHERE id = 'DB-1193'",
  "category":          "RootCause",
  "confidence":        0.87,
  "is_known_pattern":  true,
  "pattern_note":      "Similar failure in run b7c3a1f2 (similarity: 0.85)"
}
```

---

## Silent Failure Detection — 7 Anomaly Types

| Anomaly | Severity | Trigger |
|---------|----------|---------|
| `MissingToolCall` | HIGH | An expected tool was never called |
| `ToolLoop` | HIGH | Same tool called 3+ times — stuck in a reasoning loop |
| `EmptyToolResult` | HIGH | Tool returned a DB error, `[]`, or empty string |
| `IgnoredToolResult` | MEDIUM | Tool result tokens have zero overlap with the next LLM output |
| `PrematureTermination` | MEDIUM | Final Answer reached after fewer than 3 tool calls |
| `UncertainCompletion` | MEDIUM | Final Answer contains hedging language |
| `HallucinationSignal` | HIGH/MEDIUM | Final Answer mentions entities absent from all tool results |

---

## Quickstart

```bash
# 1 -- Install dependencies and seed fixtures
pip install -r requirements.txt
python agent/fixtures/seed_db.py

# 2 -- Add your OpenAI API key
# Create .env:  OPENAI_API_KEY=sk-...  and  OPENAI_MODEL=gpt-4o-mini

# 3 -- Full pipeline demo (6 steps, one command)
python demo.py
# or with a specific ticket:
python demo.py DB-1193

# 4 -- Or run step by step:
python agent/jira_triage.py PROD-2847          # live run + auto silent-failure scan
python agent/jira_triage.py . --replay <run>   # static replay, side-effects blocked
python agent/jira_triage.py . --list-steps <run>
python agent/jira_triage.py . --diverge <run> --patch-step 6 --patch-value '[...]'
python agent/jira_triage.py . --analyze <run> --hint "agent stopped at query_db"
python agent/jira_triage.py . --auto-fix <run>
```

**Start the API:**
```bash
uvicorn api.server:app --reload
# -> http://localhost:8000/docs
```

> Requires `OPENAI_API_KEY` in environment or `.env` file.
> For zero-cost local runs, swap to `ChatOllama(model="llama3")` in `ai_debugger.py` and `side_effect_classifier.py`.

---

## CLI Reference

| Flag | Argument | Description |
|------|----------|-------------|
| *(none)* | `TICKET_ID` | Live run — records trace, classifies tools, scans for silent failures |
| `--replay` | `RUN_ID` | Static replay — MockInjector, no LLM, side-effects blocked |
| `--list-steps` | `RUN_ID` | Show all steps with patch-point markers |
| `--diverge` | `RUN_ID` | Live divergence replay (requires `--patch-step` + `--patch-value` or `--patch-prompt`) |
| `--patch-step` | `N` | Step index to inject the patched value at |
| `--patch-value` | `JSON` | JSON string to inject as the tool result at patch step |
| `--patch-prompt` | `TEXT` | Replace the initial prompt for divergence replay |
| `--analyze` | `RUN_ID` | AI Root Cause Analysis via RootCauseSubAgent |
| `--hint` | `TEXT` | Optional error hint passed to RootCauseSubAgent |
| `--auto-fix` | `RUN_ID` | Full auto-remediation: detect → diagnose → fix → replay |

---

## Demo Scenarios

### Scenario A — PROD-2847: Side-Effect Trap

```bash
python agent/jira_triage.py PROD-2847
python agent/jira_triage.py PROD-2847 --replay <run_id>
# -> [MOCK-BLOCKED] send_notification -- notifications.log UNCHANGED
python agent/jira_triage.py PROD-2847 --analyze <run_id> --hint "aggressive notification"
# -> category: SideEffect · suggested_fix: add tone-check prompt
```

### Scenario B — DB-1193: Malformed SQL + Live Divergence Replay

```bash
python agent/jira_triage.py DB-1193
# -> [SilentFailureDetector] EmptyToolResult at step 7 -- HIGH
python agent/jira_triage.py . --list-steps <run_id>
python agent/jira_triage.py . --diverge <run_id> --patch-step 6 \
  --patch-value '[{"id":"DB-1193","summary":"...","assignee":"u_dave"}]'
# -> Trajectory changed: YES (original: 8 steps -- diverged: 14 steps)
```

### Scenario C — SEC-0412: Compliance Audit

```bash
curl http://localhost:8000/runs/<run_id>/integrity
# -> {"clean": true, "tampered_steps": []}
curl http://localhost:8000/runs/<run_id>/steps/4
# -> full prompt, model_name, temperature, tokens, cost -- HMAC-signed
```

### Scenario D — Auto-Remediation: One-Command Self-Healing

```bash
python agent/jira_triage.py . --auto-fix <run_id>
# [1/4] Detect  -> EmptyToolResult for query_db (HIGH)
# [2/4] Diagnose -> "SELECT * FROM tickets WHERE id = 'DB-1193'"
# [3/4] Execute -> 1 row returned
# [4/4] Replay  -> Trajectory changed: YES
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/runs` | List all recorded runs |
| `GET` | `/runs/{id}` | Run metadata (status, step count, duration) |
| `GET` | `/runs/{id}/steps` | All steps decompressed |
| `GET` | `/runs/{id}/steps/{n}` | Single step (prompt, tokens, cost, hmac_sig) |
| `GET` | `/runs/{id}/anomalies` | Silent failure report — 7 heuristics, no LLM |
| `GET` | `/runs/{id}/integrity` | HMAC-SHA256 audit verification |
| `POST` | `/runs/{id}/replay` | Static replay with optional step patch |
| `POST` | `/runs/{id}/analyze` | AI Root Cause Analysis with cross-run patterns |
| `POST` | `/agent/run` | Trigger a live agent run |

---

## Project Structure

```
sentineltrace/
├── README.md
├── PRESENTATION.md               # 10-slide jury presentation
├── demo.py                       # End-to-end pipeline demo (6 steps)
├── requirements.txt
├── sentineltrace.db              (created on first run)
├── notifications.log             (created on first run)
│
├── flight_recorder/
│   ├── __init__.py
│   ├── recorder.py               # FlightRecorder + FlightRecorderCallback
│   ├── replay.py                 # MockInjector + ReplayEngine + ToolResponseQueue
│   ├── ai_debugger.py            # RootCauseSubAgent -- AI mechanism #2
│   ├── side_effect_classifier.py # SideEffectClassifier -- AI mechanism #1
│   ├── anomaly_detector.py       # SilentFailureDetector -- 7 heuristics, zero LLM
│   └── pattern_store.py          # PatternStore -- LCS similarity, few-shot context
│
├── agent/
│   ├── __init__.py
│   ├── jira_triage.py            # Cobaye agent + full CLI
│   └── fixtures/
│       ├── seed_db.py
│       ├── tickets.db
│       ├── kb_articles.json
│       └── users.json
│
└── api/
    ├── __init__.py
    └── server.py                 # FastAPI REST API
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent framework | LangChain AgentExecutor | `BaseCallbackHandler` gives clean interception points |
| LLM interface | langchain-openai / ChatOllama | Swappable — zero-cost local option |
| Recording storage | SQLite + cbor2 + gzip | Embedded, zero-dependency, compact binary blobs |
| Tamper detection | HMAC-SHA256 (stdlib `hmac`) | WORM-style audit vault without external KMS |
| Silent failure detection | Pure Python heuristics | Zero LLM cost, deterministic, runs on every live run |
| Cross-run similarity | LCS on tool-call sequences | No ML required |
| API layer | FastAPI + Pydantic | Auto-generated OpenAPI docs |
| Python | 3.11 | `match` statements, `X | Y` union types |

---

## Hackathon Acceptance Criteria

| Criterion | Implementation | Status |
|-----------|---------------|--------|
| Deterministic replay | `ReplayEngine` + `MockInjector` | OK |
| State inspection | `GET /runs/{id}/steps/{n}` | OK |
| Divergence editing | `ToolResponseQueue` -- real LLM, patched step | OK |
| AI tool safety classification | `SideEffectClassifier` -- LLM reads descriptions | OK |
| AI root cause analysis | `RootCauseSubAgent` -- structured JSON, 5 categories | OK |
| Cross-run learning | `PatternStore` -- LCS similarity, few-shot context | OK |
| Silent failure detection | `SilentFailureDetector` -- 7 heuristics, zero LLM | OK |
| Auto-remediation | `auto_remediate()` -- detect, diagnose, fix, replay | OK |
| Audit / compliance | HMAC-SHA256 WORM vault | OK |
| Side-effect safety | AI-classified + hardcoded fallback | OK |
