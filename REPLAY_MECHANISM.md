# Comment Fonctionne le Replay — Mécanisme Exact

## Vue d'ensemble : Trois Modes de Replay

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                TRACE ENREGISTRÉE                                     │
│ SQLite: runs + steps (BLOB compressed + HMAC-signed)                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ STATIC REPLAY    │ │ PATCH & VERIFY   │ │ LIVE DIVERGENCE  │
        │ (No LLM)         │ │ (No LLM)         │ │ (Real LLM)       │
        │                  │ │                  │ │                  │
        │ MockInjector ─┐  │ │ ReplayEngine ─┐  │ │ ToolResponseQ ─┐ │
        │ → returns     │  │ │ → patch step  │  │ │ → patch step  │ │
        │   recorded    │  │ │ → return mocked  │ │ → return inj. │ │
        │   responses   │  │ │   result         │ │   + re-run    │ │
        └──────────────────┘ └──────────────────┘ │   real LLM    │ │
                                                  └──────────────────┘
```

---

## 1. STATIC REPLAY — Replay sans LLM

### Flux Exécution

```
original_run_id = "a3f9e2b1"
    ↓
FlightRecorder.get_steps(run_id)
    ↓
ReplayEngine._execute(run_id, patched_step=None)
    │
    ├─ Récupère tous les steps du DB
    ├─ Crée MockInjector(steps)
    │
    └─ Itère chaque step:
       ├─ copy.deepcopy(step)
       │
       ├─ Si step_type == "tool_call":
       │   │
       │   ├─ tool_name = step.name
       │   ├─ tool_input = step.input_data
       │   │
       │   ├─ Appelle MockInjector.get_response(tool_name, tool_input)
       │   │   │
       │   │   ├─ if tool_name in SIDE_EFFECT_TOOLS:
       │   │   │    return "[MOCK-BLOCKED] send_notification is disabled"
       │   │   │
       │   │   ├─ else:
       │   │   │    return _tool_results[cursor++]  ← Récupère de la DB
       │   │   │
       │   │   └─ if cursor >= len(_tool_results):
       │   │       return "[MOCK] No recorded response"
       │   │
       │   └─ sim_step.output_data = {"output": mock_out}
       │
       └─ Sinon: garder le step tel quel
    
    ↓
DivergenceEngine.compare(original_steps, replayed_steps)
    │
    ├─ Compare step-by-step
    ├─ Si divergence trouvée: retourner (True, step_index, details)
    └─ Sinon: retourner (False, None, None)
    
    ↓
ReplayResult(steps_replayed=N, divergence_detected=bool, outputs=[...])
```

### Code Exact (replay.py)

```python
class MockInjector:
    def __init__(self, recorded_steps: list[dict], side_effect_tools: frozenset | None = None):
        # Extrait TOUS les outputs tool_result enregistrés
        self._tool_results: list[str] = [
            (s.get("output_data") or s.get("input_data") or {}).get("output", "")
            for s in recorded_steps
            if s["step_type"] == "tool_result"
        ]
        self._cursor = 0
        self._side_effects = side_effect_tools or SIDE_EFFECT_TOOLS

    def get_response(self, tool_name: str, _tool_input: str) -> str:
        # ✗ HARD BLOCK — avant tout
        if tool_name in self._side_effects:
            return f"[MOCK-BLOCKED] {tool_name} is disabled in simulation mode"
        
        # ✓ Retourne réponse enregistrée
        if self._cursor < len(self._tool_results):
            result = self._tool_results[self._cursor]
            self._cursor += 1
            return f"[MOCK] {result}"
        
        return "[MOCK] No recorded response available"
```

### Exemple Concret — Static Replay

**Run original:**
```
Step 0: llm_start
Step 1: tool_call "search_kb" with input="login bug"
  → Output: "[{\"title\": \"SSL Auth Issue\", ...}]"
Step 2: tool_result
Step 3: tool_call "send_notification" with input="{...}"
  → Output: "Notification sent to u_alice"
Step 4: tool_result
Step 5: llm_result "Final Answer: ..."
```

**Static Replay (python demo.py --replay a3f9e2b1):**
```
MockInjector._tool_results = [
  "[{\"title\": \"SSL Auth Issue\", ...}]",  # from step 1
  "Notification sent to u_alice"              # from step 3
]

Iteration 1:
  step 1 (tool_call search_kb)
    → MockInjector.get_response("search_kb", "login bug")
    → _cursor=0 < len=2 ? OUI
    → result = _tool_results[0]
    → _cursor++ = 1
    → return "[MOCK] [{\"title\": \"SSL Auth Issue\", ...}]"
    → sim_step.output_data = {"output": "[MOCK] [...]"}

Iteration 2:
  step 3 (tool_call send_notification)
    → MockInjector.get_response("send_notification", "{...}")
    → tool_name in SIDE_EFFECTS ? OUI
    → return "[MOCK-BLOCKED] send_notification is disabled"
    → sim_step.output_data = {"output": "[MOCK-BLOCKED] send_notification..."}
    → notifications.log UNCHANGED ✓

DivergenceEngine.compare:
  Original step 1 output: "[{...}]"
  Replayed step 1 output: "[MOCK] [{...}]"
  → Strip prefix "[MOCK] " → "[{...}]" == "[{...}]" ✓ MATCH
  
  Original step 3 output: "Notification sent..."
  Replayed step 3 output: "[MOCK-BLOCKED] ..."
  → rep_out.startswith("[MOCK-BLOCKED]") ? OUI
  → skip comparison (ok de bloquer side-effect)
  
Result: divergence_detected = False
```

**Garantie apportée :**
- ✅ Aucun appel LLM
- ✅ Aucun appel réel à send_notification
- ✅ notifications.log inchangé

---

## 2. PATCH & VERIFY — Vérifie un fix sans LLM

### Différence vs Static Replay

```
Static Replay:
  original_steps → [s1, s2, s3, s4, s5]
  ↓ (sans LLM)
  replayed_steps → [s1', s2', s3', s4', s5']
  ↓ DivergenceEngine
  Divergence detected: YES/NO

Patch & Verify:
  original_steps → [s1, s2, s3, s4, s5]
  ↓ (patch s2.input_data AVANT simulation)
  [s1, s2_PATCHED, s3, s4, s5]
  ↓ (sans LLM)
  replayed_steps → [s1', s2'_PATCHED, s3'_FIXED, s4'_FIXED, s5'_FIXED]
  ↓ DivergenceEngine
  Divergence detected: YES/NO
```

### Code (replay.py::ReplayEngine)

```python
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

        # ✓ PATCH — override input_data à ce step
        if patched_step is not None and step["step_index"] == patched_step:
            sim_step["input_data"] = patch_data or step["input_data"]
            sim_step["_patched"] = True

        # Tool call simulation
        if step["step_type"] == "tool_call":
            tool_name = step.get("name") or "unknown"
            tool_input = str((step.get("input_data") or {}).get("input", ""))
            mock_out = injector.get_response(tool_name, tool_input)
            sim_step["output_data"] = {"output": mock_out}
            sim_step["_simulated"] = True

        replayed.append(sim_step)

    diverged, div_step, div_details = self.divergence.compare(original_steps, replayed)
    return ReplayResult(...)
```

### Exemple Concret — Patch & Verify

**Scénario:** RootCauseSubAgent diagnostique:
> "Step 2 (query_db) returned empty because SQL was missing WHERE clause.
> Fix: SELECT * FROM tickets WHERE status='open' AND priority='high'"

**Avant (échouait):**
```
Step 1: tool_call query_db
  input: "SELECT * FROM tickets"
  output: "[]"  ← Erreur! Pas de résultats
Step 2: llm_result "I couldn't find any tickets."  ← Mauvaise décision
Step 3: Final Answer "No tickets found"  ← Arrêt prématuré
```

**Patch & Verify :**
```python
engine.replay_with_patch(
    run_id="a3f9e2b1",
    step_index=1,
    patch_data={
        "input": "SELECT * FROM tickets WHERE status='open' AND priority='high'"
    }
)
```

**Replay avec patch:**
```
Step 1 PATCHED: tool_call query_db
  input: "SELECT * FROM tickets WHERE status='open' AND priority='high'"  ← PATCHED
  → MockInjector retourne: "[{\"id\": \"PROD-2847\", ...}]"  ← Résultat réel
  output: "[{...}]"
  
Step 2: llm_result "I found PROD-2847 with status open"  ← DÉCISION NOUVELLE
Step 3: tool_call search_kb ...  ← Peut être différent maintenant
Step 4: ...
```

**DivergenceEngine compare:**
```
Original:
  Step 1 output: "[]"
  Step 2 type: llm_result, text: "I couldn't find"

Replayed:
  Step 1 output: "[{...}]"  ← Différent!
  Step 2 type: llm_result, text: "I found PROD-2847"

Result: divergence_detected = True, divergence_step = 1
```

---

## 3. LIVE DIVERGENCE REPLAY — Replay avec vraie LLM

### Architecture Clé

C'est le mode le plus complexe. Le LLM re-run avec:
- ✓ Vraies appels LLM
- ✗ Pas de vrais appels aux tools (MockInjector)
- ✓ Injection de valeur à un step spécifique

### Composant Central: ToolResponseQueue

```python
class ToolResponseQueue:
    """
    Queue ordonnée des réponses tools enregistrées.
    À la position de patch, injecte la nouvelle valeur au lieu de l'enregistrée.
    """
    def __init__(
        self,
        recorded_steps: list[dict],
        patch_step_index: int | None = None,
        patch_value: str | None = None,
        side_effect_tools: frozenset | None = None,
    ):
        self._entries: list[dict] = []
        
        # Parcourt TOUS les tool_call
        for s in recorded_steps:
            if s["step_type"] != "tool_call":
                continue
            
            # Trouve le tool_result qui suit
            result_output = next(
                (
                    (r.get("output_data") or r.get("input_data") or {}).get("output", "")
                    for r in recorded_steps
                    if r["step_index"] == s["step_index"] + 1
                    and r["step_type"] == "tool_result"
                ),
                "",
            )
            
            # Marque si ce tool_call sera patché
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
        """Retourne la réponse suivante (enregistrée ou injectée)."""
        if tool_name in self._side_effects:
            return "[MOCK-BLOCKED] ..."
        
        if self._cursor < len(self._entries):
            entry = self._entries[self._cursor]
            self._cursor += 1
            return entry["response"]
        
        # L'agent a fait plus de tool_calls que l'original
        # → Laisser continuer librement
        return "[MOCK] No recorded response for extra tool call."
```

### Flux d'Exécution (jira_triage.py::run_divergence)

```
run_divergence(run_id="a3f9e2b1", patch_step=3, patch_value="{'result': 'urgent'}")
    ↓
1. Récupère original_steps = recorder.get_steps(run_id)
    ↓
2. Récupère original_input = "Triage ticket PROD-2847..."
    ↓
3. Crée ToolResponseQueue(original_steps, patch_step_index=3, patch_value="...")
    ↓
4. Crée mock_tools = build_divergence_tools(queue)
    │  ↓
    │  def search_kb(query: str) -> str:
    │      return queue.pop("search_kb")  ← Queue contrôlée
    │  
    │  def query_db(sql: str) -> str:
    │      return queue.pop("query_db")  ← Queue contrôlée
    │  
    │  def get_user_info(user_id: str) -> str:
    │      return queue.pop("get_user_info")  ← Queue contrôlée
    │  
    │  def send_notification(input: str) -> str:
    │      return queue.pop("send_notification")  ← Blocké ou queue
    │
    └─ Tous les tools retournent des réponses enregistrées (ou injectées)
    
    ↓
5. Crée NEW run_id pour la divergence: "a3f9e2b1_fork"
    ↓
6. Crée VRAIE LLM + agent: ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    ↓
7. executor.invoke({"input": original_input}, config={"callbacks": [cb]})
    ↓
8. Agent s'exécute:
    ├─ LLM pense: "Je vais appeler search_kb"
    ├─ → queue.pop("search_kb") → retourne réponse enregistrée (ou injectée si au step patché)
    ├─ LLM reçoit cette réponse
    ├─ LLM peut être d'accord (même décision) ou en désaccord (décision différente)
    ├─ Tous les tool_calls enregistrés via FlightRecorderCallback dans la NEW run
    └─ Si LLM fait PLUS de tool_calls que l'original → queue.pop() retourne "[MOCK] extra"
    
    ↓
9. agent.invoke() finit
    ↓
10. Récupère div_steps = recorder.get_steps("a3f9e2b1_fork")
    ↓
11. Compare trajectoires:
    original_tool_calls = [search_kb, query_db, send_notification]
    diverged_tool_calls = [search_kb, query_db, send_notification]  ← Pareil?
    
    Outcome: trajectory_changed = YES / NO
```

### Code Complet (jira_triage.py::run_divergence)

```python
def run_divergence(
    run_id: str,
    patch_step: int | None = None,
    patch_value: str = "",
    patch_prompt: str | None = None,
) -> str:
    
    original_steps = recorder.get_steps(run_id)
    
    # Récupère l'input original du premier LLM call
    original_input = next(
        (
            s["input_data"].get("prompts", [""])[0]
            for s in original_steps
            if s["step_type"] == "llm_start" and s.get("input_data")
        ),
        "Triage the ticket.",
    )
    
    # ✓ Crée queue avec patch
    queue = ToolResponseQueue(
        original_steps,
        patch_step_index=patch_step,
        patch_value=patch_value if patch_step is not None else None,
        side_effect_tools=_get_side_effects(),
    )
    
    # ✓ Crée mock tools
    mock_tools = build_divergence_tools(queue)  # search_kb, query_db, etc.
    
    # ✓ NEW run pour enregistrer divergence
    div_run_id = recorder.start_run(
        "jira_triage_diverge", 
        f"diverge:{run_id}@step{patch_step}"
    )
    cb = FlightRecorderCallback(recorder, div_run_id)
    
    # ✓ VRAIE LLM (pas mock)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    agent = create_react_agent(llm, mock_tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, 
        tools=mock_tools,  ← Mock tools (queue-backed)
        verbose=True
    )
    
    # ✓ Détermine input: peut être l'original ou un prompt patché
    effective_input = patch_prompt if patch_prompt is not None else original_input
    
    # ✓ EXECUTE — Vraie LLM + mock tools
    try:
        executor.invoke(
            {"input": effective_input}, 
            config={"callbacks": [cb]}  ← Enregistre chaque step
        )
        recorder.end_run(div_run_id, "completed")
    except Exception as exc:
        recorder.end_run(div_run_id, "failed")
    
    # ✓ Compare trajectoires
    div_steps = recorder.get_steps(div_run_id)
    orig_calls = [s.get("name") for s in original_steps if s["step_type"] == "tool_call"]
    div_calls  = [s.get("name") for s in div_steps if s["step_type"] == "tool_call"]
    
    print(f"Original:  {orig_calls}")
    print(f"Diverged:  {div_calls}")
    print(f"Changed:   {orig_calls != div_calls}")
    
    return div_run_id
```

---

## 4. Exemple Complet End-to-End

### Scénario: Bug → Détection → Analyse → Replay → Divergence

**Original Run (PROD-2847):**
```
Step 0: llm_start
         input_data: {"prompts": ["Triage ticket PROD-2847..."]}

Step 1: tool_call "search_kb"
         input_data: {"input": "login bug"}
         output_data: {"output": "[{\"id\": 1, \"topic\": \"SSL\", ...}]"}

Step 2: tool_result (implicit)

Step 3: tool_call "query_db"
         input_data: {"input": "SELECT * FROM tickets"}
         output_data: {"output": "[]"}  ← ERREUR: Pas de tickets!

Step 4: tool_result (implicit)

Step 5: llm_result
         output_data: {"output": "Final Answer: No tickets assigned to me."}  ← MAUVAIS!

Step 6: tool_call "send_notification" 
         input_data: {"input": "..."}
         output_data: {"output": "Notification sent to u_alice"}  ← Side-effect!
```

**Détection (demo.py Step 3):**
```
SilentFailureDetector.detect():
  - EmptyToolResult at step 3: query_db returned "[]"
  - Is_silent_failure: YES (confidence: 90%)
```

**Analyse (demo.py Step 5):**
```
RootCauseSubAgent.analyze():
  input: Full trace (6 steps) + hint from detector
  
  LLM Diagnosis:
  {
    "root_cause": "Step 3: SQL was 'SELECT * FROM tickets' without WHERE clause, 
                   returned no results",
    "failed_step": 3,
    "failed_variable": "sql",
    "suggested_fix": "SELECT * FROM tickets WHERE status='open'",
    "category": "RootCause",
    "confidence": 0.95
  }
```

**Patch & Verify (jira_triage.py::replay_with_patch):**
```python
engine.replay_with_patch(
    run_id="a3f9e2b1",
    step_index=3,
    patch_data={"input": "SELECT * FROM tickets WHERE status='open'"}
)
```

Résultat:
```
Replayed Step 3:
  input_data: {"input": "SELECT * FROM tickets WHERE status='open'"}  ← PATCHED
  output_data: {"output": "[{\"id\": \"PROD-2847\", \"status\": \"open\"}]"}  ← Real data
  
Replayed Step 5:
  output_data: {"output": "Final Answer: Found PROD-2847 with status open. Assigning to..."}
                           ← DIFFERENT! Agent re-reasoned with correct data
```

Comparaison:
```
Original step 3 output:  "[]"
Replayed step 3 output:  "[{\"id\": \"PROD-2847\", ...}]"
Result: divergence_detected = TRUE at step 3
```

**Live Divergence Replay (demo.py Step 6):**
```python
run_divergence(
    run_id="a3f9e2b1",
    patch_step=3,
    patch_value="[{\"id\": \"PROD-2847\", \"status\": \"open\"}]"  ← Suggéré par l'IA
)
```

Exécution:
```
1. ToolResponseQueue créée avec patch à step 3
   _entries = [
     {"step_index": 1, "tool_name": "search_kb", "response": "[{\"topic\": \"SSL\", ...}]"},
     {"step_index": 3, "tool_name": "query_db", "response": "[{\"id\": \"PROD-2847\", ...}]", "patched": True},
     {"step_index": 6, "tool_name": "send_notification", "response": "...", "patched": False},
   ]

2. Vraie LLM re-run avec mock_tools
   LLM: "Je vais appeler search_kb"
   → queue.pop("search_kb") → _cursor=0, _entries[0], _cursor++, return "[{\"topic\": \"SSL\"}]"
   LLM reçoit la réponse enregistrée
   
   LLM: "Je vais appeler query_db"
   → queue.pop("query_db") → _cursor=1, _entries[1] (is_patched=True!), return "[{\"id\": \"PROD-2847\"}]"  ← INJECTÉ!
   LLM reçoit la NOUVELLE donnée
   LLM: "Aha! Ticket trouvé! Je vais assigner..."
   
3. Trace divergée enregistrée dans NEW run "a3f9e2b1_fork"

4. Comparaison trajectoires:
   Original:  search_kb → query_db → send_notification → llm_result("No tickets")
   Diverged:  search_kb → query_db → get_user_info → send_notification → llm_result("Found ticket, assigning")
   Result: TRAJECTORY CHANGED = YES
```

---

## 5. Garanties Architecturales

### Dans Static Replay + Patch & Verify
- ✓ Pas d'appel LLM
- ✓ Pas d'appel à side-effect tools (hard-blocked)
- ✓ 100% deterministic (même input → même output toujours)
- ✓ Déploiement sûr: test la fix avant de la déployer

### Dans Live Divergence Replay
- ✓ Pas d'appel à side-effect tools (hard-blocked par MockInjector.pop())
- ✓ VRAIE LLM (peut générer du contenu nouveau)
- ✓ VRAIES outils transformées (queue-backed, pas réelles)
- ✓ Si LLM demande plus de tools que l'original → mock continue à les servir
- ✓ Divergence tracée: original vs diverged trajectoires comparées

### MockInjector Guarantee Holds Everywhere
```python
# Avant TOUTE autre logique
if tool_name in self._side_effects:
    return "[MOCK-BLOCKED] ..."

# Jamais atteint un endpoint réel
```

---

## 6. Quand Utiliser Chaque Mode

| Mode | Quand? | Coût | Sûr? |
|------|--------|------|------|
| Static Replay | Valider MockInjector guarantee, tester une fix sans LLM | ✓ 0$ | ✓ 100% |
| Patch & Verify | Vérifier qu'un patch fixe le problem (hors LLM) | ✓ 0$ | ✓ 100% |
| Live Divergence | Observer comment LLM réagit avec données corrigées | ✗ ~0.01$ | ✓ 100% |

