"""
Pattern Store — cross-run similarity search for the RootCauseSubAgent.

Finds past runs whose tool call trajectories are similar to the current
failed run, then formats them as few-shot context. The RootCauseSubAgent
can say "this pattern has been seen before — in run b7c3a1f2 it was fixed
by adding tenant_id to the SQL query."

Similarity metric: Longest Common Subsequence (LCS) ratio on tool name
sequences. Simple, deterministic, zero ML dependencies.

  seq_a = [search_kb, query_db, send_notification]
  seq_b = [search_kb, query_db, get_user_info, send_notification]
  LCS   = [search_kb, query_db, send_notification] → score = 0.86
"""
import sqlite3
from dataclasses import dataclass, field

from flight_recorder.recorder import FlightRecorder


@dataclass
class SimilarRun:
    run_id: str
    tool_sequence: list[str]
    status: str
    similarity_score: float
    step_count: int
    input_text: str
    failed_at_tool: str | None = None   # first tool_call with anomaly=1, if any


def _tool_sequence(steps: list[dict]) -> list[str]:
    """Ordered list of tool names called in a run."""
    return [
        s["name"]
        for s in steps
        if s["step_type"] == "tool_call" and s.get("name")
    ]


def _first_anomaly_tool(steps: list[dict]) -> str | None:
    """Return the tool name at the first anomaly step, if any."""
    for s in steps:
        if s.get("anomaly") and s["step_type"] == "tool_call":
            return s.get("name")
    return None


def _lcs_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """
    LCS-based similarity ratio in [0, 1].
    Uses F1-style normalisation: 2*|LCS| / (|A| + |B|).
    """
    if not seq_a or not seq_b:
        return 0.0
    m, n = len(seq_a), len(seq_b)
    # Standard DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return (2 * dp[m][n]) / (m + n)


class PatternStore:
    """
    Finds runs in the database whose tool trajectories are similar to the
    run under analysis, and formats them as few-shot context for the
    RootCauseSubAgent.

    With only a handful of runs (as expected in a hackathon), this already
    adds value: if the same tool sequence failed twice, the subagent sees
    both failure modes and can distinguish them or confirm a pattern.
    """

    def __init__(self, recorder: FlightRecorder):
        self.recorder = recorder

    def find_similar(
        self,
        run_id: str,
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> list[SimilarRun]:
        """
        Return the top_k most similar past runs by tool sequence.
        Excludes the run itself, divergence runs, and test runs.
        """
        target_steps = self.recorder.get_steps(run_id)
        target_seq = _tool_sequence(target_steps)
        if not target_seq:
            return []

        conn = sqlite3.connect(self.recorder.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT run_id, agent_name, input, status, step_count
               FROM runs
               WHERE run_id != ?
                 AND agent_name NOT LIKE '%diverge%'
                 AND agent_name NOT LIKE '%test%'
               ORDER BY started_at DESC
               LIMIT 100""",
            (run_id,),
        ).fetchall()
        conn.close()

        scored: list[SimilarRun] = []
        for row in rows:
            steps = self.recorder.get_steps(row["run_id"])
            seq = _tool_sequence(steps)
            score = _lcs_similarity(target_seq, seq)
            if score >= min_similarity:
                scored.append(SimilarRun(
                    run_id=row["run_id"],
                    tool_sequence=seq,
                    status=row["status"],
                    similarity_score=score,
                    step_count=row["step_count"] or 0,
                    input_text=(row["input"] or "")[:120],
                    failed_at_tool=_first_anomaly_tool(steps),
                ))

        scored.sort(key=lambda r: r.similarity_score, reverse=True)
        return scored[:top_k]

    def format_for_context(self, similar_runs: list[SimilarRun]) -> str:
        """
        Serialize similar runs into a compact string for injection into the
        RootCauseSubAgent prompt. Designed to be informative but concise —
        we are paying per token.
        """
        if not similar_runs:
            return "(No similar past runs found — this may be a new failure pattern.)"

        lines = ["=== Similar past runs (few-shot pattern context) ==="]
        for i, run in enumerate(similar_runs, 1):
            status_label = "FAILED" if run.status == "failed" else "OK"
            lines.append(
                f"\n[Pattern {i}] run_id={run.run_id} "
                f"similarity={run.similarity_score:.0%} status={status_label}"
            )
            lines.append(f"  Input   : {run.input_text}")
            lines.append(
                f"  Path    : {' → '.join(run.tool_sequence) if run.tool_sequence else '(no tools)'}"
            )
            if run.failed_at_tool:
                lines.append(f"  Crashed : at {run.failed_at_tool}")
            elif run.status == "failed":
                lines.append(f"  Crashed : unknown step")
            else:
                lines.append(f"  Outcome : completed successfully")
        lines.append("\n=== End of pattern context ===")
        return "\n".join(lines)
