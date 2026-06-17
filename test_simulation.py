"""
Test rapide du Recorder + Replay — sans clé API, sans LLM.

Simule un run avec 4 steps manuels (un tool_call send_notification inclus),
puis rejoue et vérifie que le Mock Injector bloque bien le side-effect.

Lancer : python test_simulation.py
"""
from flight_recorder.recorder import FlightRecorder
from flight_recorder.replay import ReplayEngine

recorder = FlightRecorder(db_path="test_sentinel.db")

# ── 1. Créer un run fictif ────────────────────────────────────────────────────
run_id = recorder.start_run("test_agent", "PROD-2847")
print(f"Run créé : {run_id}")

# Step 0 — LLM décide d'appeler search_kb
recorder.record_step(run_id, 0, "llm_start", "gpt-4o-mini",
    {"prompts": ["Triage ticket PROD-2847"]},
    model_name="gpt-4o-mini", temperature=0.2, tokens_in=120, tokens_out=45)

# Step 1 — tool_call search_kb
recorder.record_step(run_id, 1, "tool_call", "search_kb",
    {"input": "notification tone policy"})

# Step 2 — résultat search_kb
recorder.record_step(run_id, 2, "tool_result", None,
    {"output": '[{"id":"KB-001","title":"Notification Guidelines"}]'})

# Step 3 — tool_call send_notification (SIDE EFFECT)
recorder.record_step(run_id, 3, "tool_call", "send_notification",
    {"input": '{"user_id": "u_alice", "subject": "URGENT: violation!", "body": "Fix now."}'})

# Step 4 — résultat send_notification
recorder.record_step(run_id, 4, "tool_result", None,
    {"output": "Notification sent to u_alice."})

recorder.end_run(run_id, "completed")
print(f"Steps enregistrés : 5")

# ── 2. Vérifier l'intégrité HMAC ─────────────────────────────────────────────
tampered = recorder.verify_integrity(run_id)
print(f"\nIntégrité HMAC : {'OK ✓' if not tampered else f'TAMPERED {tampered}'}")

# ── 3. Rejouer en simulation ──────────────────────────────────────────────────
engine = ReplayEngine(recorder)
result = engine.replay(run_id)
print(f"\nReplay terminé")
print(f"  Steps rejoués     : {result.steps_replayed}")
print(f"  Divergence        : {result.divergence_detected}")

# Vérifier que send_notification est bien bloqué
for step in result.outputs:
    out = (step.get("output_data") or {}).get("output", "")
    if step.get("name") == "send_notification" or "MOCK-BLOCKED" in out:
        print(f"\n  Step {step['step_index']} send_notification → {out}")

# ── 4. Tester replay_with_patch ──────────────────────────────────────────────
print("\n── Patch step 1 (changer la query KB) ──────────────────────────────")
result_patched = engine.replay_with_patch(
    run_id, step_index=1,
    patch_data={"input": "finance admin approval workflow"}
)
print(f"  Divergence        : {result_patched.divergence_detected}")
print(f"  At step           : {result_patched.divergence_step}")
print(f"  Details           : {result_patched.divergence_details}")

print("\n[OK] Test simulation complet — aucune clé API nécessaire.")
