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

**Tagline:** *Record. Replay. Inspect. Diverge. Self-Heal.*

---

## SLIDE 2 — The Problem (the incident that drives everything)

> **2:00 AM.** Your Jira Triage Agent runs automatically.
> **9:00 AM.** A customer calls — furious. The agent sent them an aggressive email.
> Your engineer types: `python agent.py --rerun PROD-2847`

Three things went wrong simultaneously:

| Action | Consequence |
|--------|-------------|
| Reran the agent | LLM produced different output — bug may not reproduce |
| Tool calls re-fired | A second email may have been sent |
| Original trace lost | You can never know what the agent *actually* decided at 2 AM |

**The core insight:** an agent that has already caused a side effect cannot be safely debugged by running it again. You need to replay the *original* execution — deterministically, without triggering any live system.

This is what SentinelTrace was built to do.

---

## SLIDE 3 — Why Existing Tools Fall Short

```
                    LangSmith    Langfuse    Phoenix    SentinelTrace
                    ─────────    ────────    ───────    ─────────────
Trace logging           ✓           ✓          ✓            ✓
Deterministic replay    ✗           ✗          ✗            ✓  (MUST)
Side-effect blocking    ✗           ✗          ✗            ✓  (enforced at proxy layer)
State inspection        partial     partial    partial      ✓  (MUST — per step)
Divergence editing      ✗           ✗          ✗            ✓  (SHOULD — prompt or tool)
AI root cause analysis  ✗           ✗          ✗            ✓  (bonus)
Silent failure detect   ✗           ✗          ✗            ✓  (bonus)
Compliance audit trail  ✗           ✗          ✗            ✓  (HMAC-SHA256)
```

Existing tools are *read-only mirrors*. SentinelTrace is an *active replay and debugging engine*.

---

## SLIDE 4 — Architecture: 5 Layers + The Proxy Solution

The spec identifies **proxying** as the primary technical challenge:
> "Building the proxy that seamlessly intercepts tool calls without altering the agent's core architecture."

**SentinelTrace's answer:** `BaseCallbackHandler` — LangChain's built-in extension point.
Zero changes to the agent under observation. Zero monkey-patching.

```
┌─────────────────────────────────────────────────────────┐
│ L1  AGENT LAYER                                         │
│     LangChain AgentExecutor · 4 tools · LLM · Run ID   │
├─────────────────────────────────────────────────────────┤
│ L2  PROXY & INTERCEPT  ← the primary challenge, solved  │
│     FlightRecorderCallback (BaseCallbackHandler)        │
│     → on_llm_start/end  · on_tool_start/end             │
│     SideEffectClassifier · SilentFailureDetector        │
├─────────────────────────────────────────────────────────┤
│ L3  STORAGE                                             │
│     SQLite steps · BLOB snapshots · HMAC-SHA256 vault   │
├─────────────────────────────────────────────────────────┤
│ L4  REPLAY ENGINE                                       │
│     ReplayEngine · MockInjector · DivergenceEngine      │
│     ToolResponseQueue (live divergence)                 │
├─────────────────────────────────────────────────────────┤
│ L5  OBSERVABILITY                                       │
│     FastAPI REST · Dashboard UI · Compliance Export     │
└─────────────────────────────────────────────────────────┘
```

**Key property of the proxy:** no code path in simulation mode reaches a live endpoint.
The side-effect block is enforced *before* any recorded data is consulted.

---

## SLIDE 5 — Acceptance Criteria: MUST (all satisfied)

### Criterion 1 — Record functionality works (MUST)

> *"A live agent run is successfully recorded, capturing at least one LLM call and one tool call."*

```python
# Transparent — no agent code changed
executor.invoke({"input": "Triage ticket PROD-2847"},
                config={"callbacks": [FlightRecorderCallback(recorder, run_id)]})

# Every step captured:
#   llm_start  · prompts, model_name
#   llm_result · generated text, tokens_in, tokens_out, latency_ms, cost_usd
#   tool_call  · tool name, input payload, status
#   tool_result· output payload
# Signed with HMAC-SHA256. Compressed with cbor2+gzip.
```

---

### Criterion 2 — Deterministic replay (MUST)

> *"The recorded run is replayed successfully without triggering the live external tool."*

```
python agent/jira_triage.py . --replay a3f9e2b1

  Steps replayed     : 12
  Live calls         : 0   (Mock Injector — no live endpoint reachable)
  Side-effect blocks : 1
    step 9: send_notification → [MOCK-BLOCKED]
  notifications.log  : 3 entries  (UNCHANGED by replay)
```

The Mock Injector returns recorded responses. `send_notification` is blocked
*by name* before any other logic runs — architectural, not configurable.

---

### Criterion 3 — State inspection (MUST)

> *"The user can inspect the exact context/prompt sent to the LLM at a specific step."*

Every `llm_start` step records the **full prompt array** in `input_data`.
Every step stores: model_name, temperature, tokens_in, tokens_out, cost_usd, hmac_sig.

```bash
# CLI
python agent/jira_triage.py . --list-steps a3f9e2b1

# API
curl http://localhost:8000/runs/a3f9e2b1/steps/2
# → {
#     "step_index": 2,
#     "step_type": "llm_start",
#     "input_data": {"prompts": ["Triage ticket PROD-2847...\nThought:"]},
#     "model_name": "gpt-4o-mini",
#     "tokens_in": 312,
#     "cost_usd": 0.00156,
#     "hmac_sig": "a7f3..."
#   }
```

Dashboard: `http://localhost:8000` — step timeline with prompt inspector.

---

### Criterion 4 — Divergence editing (SHOULD)

> *"A developer modifies a prompt or tool result during replay and the agent takes a new path."*

Two patching modes — both supported:

**Mode A — Modify a tool result** (inject corrected SQL at failure point):
```bash
python agent/jira_triage.py . --diverge a3f9e2b1 \
  --patch-step 6 \
  --patch-value '[{"id":"PROD-2847","status":"open","assignee":"u_alice"}]'

# Original path  : search_kb → query_db → (stopped — DB error)
# Diverged path  : search_kb → query_db → get_user_info → send_notification
# Trajectory changed : YES
```

**Mode B — Modify the prompt** (change agent instruction):
```bash
python agent/jira_triage.py . --diverge a3f9e2b1 \
  --patch-prompt "Investigate ticket PROD-2847 but do NOT send any notifications."

# Diverged path  : search_kb → query_db → get_user_info → (no send_notification)
# Trajectory changed : YES
```

The LLM re-runs with real reasoning on the patched context.
The new trajectory is recorded as a new run and compared step-by-step.

---

## SLIDE 6 — The Two AI Mechanisms

SentinelTrace uses two dedicated LLM subagents — both run at temperature=0.

### AI Mechanism 1 — SideEffectClassifier

Runs before every live agent run. Reads each tool's name and description and
decides which ones have irreversible side effects. No hardcoded list needed.

```
Input  : list of tools with names + descriptions
Output :
  side_effect_tools : ["send_notification"]
  safe_tools        : ["search_kb", "query_db", "get_user_info"]
  reasoning         : {"send_notification": "Sends real messages — cannot be undone",
                       "query_db": "Read-only SELECT — no state modified"}
  confidence        : 0.98
```

**Why this matters:** without it, SentinelTrace requires a per-agent hardcoded blocklist.
With it, SentinelTrace safely replays **any** unknown agent.
Falls back to a default set if the LLM is unavailable — safety never dropped.

---

### AI Mechanism 2 — RootCauseSubAgent

Diagnoses failures after they are detected. Receives the full serialized trace
+ few-shot context from structurally similar past runs (PatternStore/LCS).

```json
{
  "root_cause":       "Step 7: query_db returned DB ERROR — SQL used unquoted string ID",
  "failed_step":      7,
  "failed_variable":  "sql",
  "suggested_fix":    "SELECT * FROM tickets WHERE id = 'DB-1193'",
  "category":         "RootCause",
  "confidence":       0.87,
  "is_known_pattern": true,
  "pattern_note":     "Same SQL error seen in run b7c3a1f2 (LCS similarity: 0.85)"
}
```

**Why a dedicated subagent?** Microsoft Research, Debug2Fix (ACM Feb 2026):
dedicated debugging subagent improves bug resolution by **+21%** vs. self-debugging.

---

## SLIDE 7 — Scenario A: The Side-Effect Trap (spec §3.2-A)

> *"An agent drafts an aggressive email. The engineer needs to replay the exact scenario*
> *without actually sending the email, then tweak the prompt and test deterministically."*

```
Step 1 — Live run records the bad behaviour
  python agent/jira_triage.py PROD-2847
  → notifications.log : 1 entry  ← the aggressive email

Step 2 — Replay: Mock Injector blocks send_notification
  python agent/jira_triage.py . --replay <run_id>
  → [MOCK-BLOCKED] send_notification disabled in simulation mode
  → notifications.log : 1 entry  ← UNCHANGED (verifiable: count the lines)

Step 3 — Patch the prompt, re-run with real LLM
  python agent/jira_triage.py . --diverge <run_id> \
    --patch-prompt "Triage PROD-2847. Use professional tone. Flag for human review."
  → Trajectory changed: YES
  → send_notification called with professional subject

Step 4 — AI confirms root cause
  python agent/jira_triage.py . --analyze <run_id>
  → category: SideEffect
  → suggested_fix: "Add tone-moderation prompt before send_notification step"
```

**Jury verifies:** `notifications.log` line count is identical before and after replay.
The guarantee is architectural — not a config flag.

---

## SLIDE 8 — Scenario B: Divergence Testing (spec §3.2-B)

> *"An agent fails because it queried a database tool with incorrect SQL syntax.*
> *The engineer steps through the trace, injects the correct SQL at the exact moment*
> *of failure, and lets the agent continue to verify subsequent steps."*

```
Step 1 — Live run: agent stops at query_db (DB error)
  python agent/jira_triage.py DB-1193
  → [SilentFailureDetector] EmptyToolResult at step 7 — HIGH

Step 2 — Inspect trace to find the injection point
  python agent/jira_triage.py . --list-steps <run_id>
  #   6  tool_call   query_db   SELECT * FROM tickets...  << patch here?
  #   7  tool_result            DB ERROR: no such column...

Step 3 — Inject correct SQL, watch agent continue
  python agent/jira_triage.py . --diverge <run_id> \
    --patch-step 6 \
    --patch-value '[{"id":"DB-1193","summary":"...","assignee":"u_dave"}]'

  Original  (8 steps):  search_kb → query_db → (stopped)
  Diverged (14 steps):  search_kb → query_db → get_user_info → send_notification
  Trajectory changed   : YES ✓
```

**What the jury sees:** the diverged run has 6 more steps than the original.
The LLM was not restarted — it re-reasoned from the injection point using mock tools.
The new trajectory is persisted as a separate run in the database.

---

## SLIDE 9 — Scenario C: Auditing and Compliance (spec §3.2-C)

> *"A compliance officer needs to understand exactly what context an agent had access to*
> *when it approved a sensitive workflow three weeks ago."*

SentinelTrace stores every step with HMAC-SHA256 signatures at write time.
The audit vault is WORM — signatures cannot be forged after the fact.

```bash
# 1 — Verify the trace has not been tampered with
curl http://localhost:8000/runs/<run_id>/integrity
→ {"clean": true, "tampered_steps": []}

# 2 — Inspect the exact context at the decision step
curl http://localhost:8000/runs/<run_id>/steps/4
→ {
    "step_type"   : "llm_start",
    "input_data"  : {"prompts": ["Review access request for sensitive-workflow..."]},
    "model_name"  : "gpt-4o-mini",
    "temperature" : 0.2,
    "tokens_in"   : 412,
    "cost_usd"    : 0.00206,
    "hmac_sig"    : "a7f3b2...",
    "recorded_at" : 1748000412.3
  }

# 3 — Export the full trace for the compliance record
curl http://localhost:8000/runs/<run_id>/steps > audit_export.json
```

The agent doesn't need to run again. The evidence is immutable and signed.

---

## SLIDE 10 — Value Proposition & What's Next

### What SentinelTrace delivers against the spec

| Spec requirement | Implementation | Priority |
|------------------|---------------|----------|
| Record: LLM call + tool call captured | `FlightRecorderCallback` — cbor2+gzip+HMAC | **MUST** |
| Deterministic replay: no live tool | `MockInjector` — architectural block | **MUST** |
| State inspection: prompt per step | `GET /runs/{id}/steps/{n}` — full input_data | **MUST** |
| Divergence: prompt or tool patch | `--patch-prompt` / `--patch-step` + real LLM | **SHOULD** |
| Proxying challenge solved | `BaseCallbackHandler` — zero agent code change | key consideration |
| Visualization | Dashboard at `http://localhost:8000` + `--list-steps` | key consideration |

### Beyond the spec (bonus features)

| Feature | Value |
|---------|-------|
| AI side-effect classification | Generic replay for any unknown agent |
| AI root cause analysis | Structured JSON diagnosis — category, confidence, fix |
| Silent failure detection | 7 heuristics — catches failures that raise no exception |
| Cross-run pattern learning | LCS similarity — few-shot context for repeat bugs |
| Auto-remediation | Detect → diagnose → fix → validate in one command |
| HMAC-SHA256 audit vault | Tamper-evident compliance evidence |

### Next steps

- Multi-agent chain support (trace chains of agents, not just single executors)
- Async FastAPI workers + Redis for high-volume production ingestion
- LLM-agnostic: `ChatOllama` already supported — zero-cost local debugging
- Anomaly alerting: webhook dispatch when `is_silent_failure = true`

---

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   SentinelTrace doesn't just observe your agents.         ║
║   It lets you rewind them, fix them, and prove it.        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```
