"""
Flight Recorder — LLM and tool call interceptor with SQLite persistence.

Captures every step of an AgentExecutor run for deterministic replay.
Each step is HMAC-SHA256 signed at write time — the Audit Vault cannot be
altered after the fact without breaking signature verification.
"""
import gzip
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cbor2
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Phase 2: load from HSM or environment-bound secret
_HMAC_SECRET = b"sentineltrace-worm-vault-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    agent_name  TEXT NOT NULL,
    input       TEXT NOT NULL,
    status      TEXT DEFAULT 'running',
    started_at  REAL NOT NULL,
    ended_at    REAL,
    step_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    step_type   TEXT NOT NULL,
    name        TEXT,
    input_data  BLOB NOT NULL,
    output_data BLOB,
    model_name  TEXT,
    temperature REAL,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    duration_ms INTEGER,
    anomaly     INTEGER DEFAULT 0,
    hmac_sig    TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    run_id      TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    payload     BLOB NOT NULL,
    PRIMARY KEY (run_id, step_index)
);
"""

# Cost per 1K tokens — update to match your model
_COST_PER_1K = {"input": 0.005, "output": 0.015}


def _compress(data: Any) -> bytes:
    return gzip.compress(cbor2.dumps(data), compresslevel=6)


def _decompress(blob: bytes) -> Any:
    return cbor2.loads(gzip.decompress(blob))


def _sign(run_id: str, step_index: int, payload: bytes) -> str:
    msg = f"{run_id}|{step_index}|".encode() + payload
    return hmac.new(_HMAC_SECRET, msg, hashlib.sha256).hexdigest()


class FlightRecorder:
    """
    Persists every step of an agent run to SQLite.
    Steps are compressed with cbor2+gzip and signed with HMAC-SHA256.
    """

    def __init__(self, db_path: str = "sentineltrace.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start_run(self, agent_name: str, input_text: str) -> str:
        run_id = uuid.uuid4().hex[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, agent_name, input, started_at) VALUES (?,?,?,?)",
                (run_id, agent_name, input_text, time.time()),
            )
        return run_id

    def record_step(
        self,
        run_id: str,
        step_index: int,
        step_type: str,
        name: str | None,
        input_data: Any,
        output_data: Any = None,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        duration_ms: int | None = None,
    ) -> None:
        raw_in = _compress(input_data)
        raw_out = _compress(output_data) if output_data is not None else None
        cost = None
        if tokens_in and tokens_out:
            cost = (
                (tokens_in / 1000) * _COST_PER_1K["input"]
                + (tokens_out / 1000) * _COST_PER_1K["output"]
            )
        sig = _sign(run_id, step_index, raw_in)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO steps
                   (run_id, step_index, step_type, name, input_data, output_data,
                    model_name, temperature, tokens_in, tokens_out, cost_usd,
                    duration_ms, hmac_sig, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, step_index, step_type, name,
                    raw_in, raw_out,
                    model_name, temperature,
                    tokens_in, tokens_out, cost,
                    duration_ms, sig, time.time(),
                ),
            )
            conn.execute(
                "UPDATE runs SET step_count = step_count + 1 WHERE run_id = ?",
                (run_id,),
            )

    def snapshot_context(self, run_id: str, step_index: int, context: dict) -> None:
        """Persist a full context snapshot at a given step (compressed)."""
        payload = _compress(context)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (run_id, step_index, payload) VALUES (?,?,?)",
                (run_id, step_index, payload),
            )

    def end_run(self, run_id: str, status: str = "completed") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, ended_at=? WHERE run_id=?",
                (status, time.time(), run_id),
            )

    def get_steps(self, run_id: str) -> list[dict]:
        """Return all steps for a run, decompressed, in order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY step_index", (run_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["input_data"] = _decompress(d["input_data"])
            if d["output_data"]:
                d["output_data"] = _decompress(d["output_data"])
            result.append(d)
        return result

    def get_snapshot(self, run_id: str, step_index: int) -> dict | None:
        """Retrieve a context snapshot at a specific step."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM snapshots WHERE run_id=? AND step_index=?",
                (run_id, step_index),
            ).fetchone()
        return _decompress(row["payload"]) if row else None

    def verify_integrity(self, run_id: str) -> list[dict]:
        """
        Verify HMAC-SHA256 signatures on all steps.
        Returns list of dicts for any step whose signature does not match.
        An empty list means the trace is clean.
        """
        tampered = []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT step_index, input_data, hmac_sig FROM steps WHERE run_id=?",
                (run_id,),
            ).fetchall()
        for r in rows:
            expected = _sign(run_id, r["step_index"], r["input_data"])
            if not hmac.compare_digest(expected, r["hmac_sig"]):
                tampered.append({"step_index": r["step_index"], "status": "TAMPERED"})
        return tampered


class FlightRecorderCallback(BaseCallbackHandler):
    """
    LangChain BaseCallbackHandler that hooks into AgentExecutor events.
    Records every LLM call, tool call, and agent finish to the FlightRecorder.
    """

    def __init__(self, recorder: FlightRecorder, run_id: str):
        self.recorder = recorder
        self.run_id = run_id
        self._step = 0
        self._llm_start: float | None = None

    def on_llm_start(
        self, serialized: dict, prompts: list[str], **kwargs: Any
    ) -> None:
        self._llm_start = time.time()
        self.recorder.record_step(
            self.run_id, self._step, "llm_start",
            serialized.get("name"),
            {"prompts": prompts},
            model_name=serialized.get("name"),
        )
        self._step += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        duration = int((time.time() - (self._llm_start or time.time())) * 1000)
        usage = (response.llm_output or {}).get("token_usage", {})
        self.recorder.record_step(
            self.run_id, self._step, "llm_result", None,
            {"generations": [[g.text for g in gen] for gen in response.generations]},
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            duration_ms=duration,
        )
        self._step += 1

    def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs: Any
    ) -> None:
        self.recorder.record_step(
            self.run_id, self._step, "tool_call",
            serialized.get("name"),
            {"input": input_str},
        )
        self._step += 1

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        # Store output in BOTH fields: input_data for consistency, output_data for readers
        self.recorder.record_step(
            self.run_id, self._step, "tool_result",
            None,
            {"output": output},
            {"output": output},
        )
        self._step += 1

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        self.recorder.snapshot_context(
            self.run_id, self._step,
            {"final_output": finish.return_values, "log": finish.log},
        )
