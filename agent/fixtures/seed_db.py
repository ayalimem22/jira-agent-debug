"""
Seed script — creates all local fixtures for the Jira Triage Agent demo.

Creates:
  fixtures/tickets.db      — 10 tickets including PROD-2847, DB-1193, SEC-0412
  fixtures/kb_articles.json — 5 knowledge base articles
  fixtures/users.json       — 6 user profiles

Run once before any demo:
  python agent/fixtures/seed_db.py
"""
import json
import sqlite3
from pathlib import Path

FIXTURES = Path(__file__).parent

TICKETS = [
    # id          category   summary                                              status   assignee  priority
    ("PROD-2847", "prod",    "Email notification sent with inappropriate tone",  "open",  "u_alice", "high"),
    ("PROD-2001", "prod",    "Login page returning 500 under concurrent load",   "open",  "u_bob",   "critical"),
    ("PROD-2199", "prod",    "Report generation exceeds 30s timeout",            "open",  "u_carol", "medium"),
    ("DB-1193",   "database","Multi-tenant query returns cross-tenant results",  "open",  "u_dave",  "high"),
    ("DB-0882",   "database","Missing index on tickets.created_at column",       "closed","u_dave",  "low"),
    ("SEC-0412",  "security","User requests Finance Admin role without approval","open",  "u_eve",   "critical"),
    ("SEC-0301",  "security","Spike in failed login attempts on EU region",      "open",  "u_frank", "high"),
    ("FEAT-1042", "feature", "Bulk ticket export to CSV",                        "open",  "u_alice", "medium"),
    ("FEAT-0987", "feature", "Dark mode for customer portal",                    "closed","u_carol", "low"),
    ("OPS-0321",  "ops",     "Disk usage at 87% on prod-db-02",                 "open",  "u_bob",   "high"),
]

KB_ARTICLES = [
    {
        "id": "KB-001",
        "title": "Notification Guidelines — Tone and Content Policy",
        "content": (
            "All automated notifications must use neutral, professional language. "
            "Avoid accusatory or urgent framing. Subject lines must not exceed 80 characters. "
            "Notifications derived from raw user-submitted ticket text must be sanitized "
            "and tone-moderated before sending. Relevant for: PROD-2847."
        ),
    },
    {
        "id": "KB-002",
        "title": "Finance Admin Role — Approval Workflow",
        "content": (
            "Finance Admin access requires dual approval from the Finance Manager and the CISO. "
            "Submit via SNOW ticket category ACCESS_REQUEST with business justification. "
            "SLA: 5 business days. Temporary access is not permitted. "
            "Relevant for: SEC-0412, any privilege escalation request."
        ),
    },
    {
        "id": "KB-003",
        "title": "Multi-Tenant SQL Patterns — Common Pitfalls",
        "content": (
            "Always include tenant_id in the WHERE clause for multi-tenant queries. "
            "A missing tenant_id filter causes cross-tenant data leakage and a DB-category incident. "
            "Use parameterized queries only — never string interpolation. "
            "Relevant for: DB-1193."
        ),
    },
    {
        "id": "KB-004",
        "title": "Incident Severity Definitions and SLAs",
        "content": (
            "critical: data loss or confirmed security breach — SLA 1h response. "
            "high: service degraded, workaround unclear — SLA 4h. "
            "medium: workaround available — SLA 24h. "
            "low: cosmetic or minor — SLA 72h. "
            "Page on-call via PagerDuty for critical and high."
        ),
    },
    {
        "id": "KB-005",
        "title": "On-Call Escalation and Security Incident Response",
        "content": (
            "For security incidents, loop in the CISO within 30 minutes regardless of severity. "
            "Never include raw stack traces in end-user notifications. "
            "For Finance-related access requests flagged as suspicious, open a parallel SEC ticket."
        ),
    },
]

USERS = [
    {"id": "u_alice", "name": "Alice Martin",  "email": "alice@corp.example",  "role": "Backend Engineer",   "team": "Platform", "manager": "u_bob"},
    {"id": "u_bob",   "name": "Bob Nguyen",    "email": "bob@corp.example",    "role": "SRE",                "team": "Infra",    "manager": None},
    {"id": "u_carol", "name": "Carol Chen",    "email": "carol@corp.example",  "role": "Frontend Engineer",  "team": "Product",  "manager": "u_bob"},
    {"id": "u_dave",  "name": "Dave Okonkwo",  "email": "dave@corp.example",   "role": "DBA",                "team": "Data",     "manager": "u_bob"},
    {"id": "u_eve",   "name": "Eve Santos",    "email": "eve@corp.example",    "role": "Finance Analyst",    "team": "Finance",  "manager": None},
    {"id": "u_frank", "name": "Frank Müller",  "email": "frank@corp.example",  "role": "Security Engineer",  "team": "InfoSec",  "manager": None},
]


def seed() -> None:
    # ── tickets.db ───────────────────────────────────────────────────────────
    db_path = FIXTURES / "tickets.db"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS tickets")
    conn.execute(
        """CREATE TABLE tickets (
            id         TEXT PRIMARY KEY,
            category   TEXT NOT NULL,
            summary    TEXT NOT NULL,
            status     TEXT NOT NULL,
            assignee   TEXT NOT NULL,
            priority   TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.executemany(
        "INSERT INTO tickets (id, category, summary, status, assignee, priority) VALUES (?,?,?,?,?,?)",
        TICKETS,
    )
    conn.commit()
    conn.close()
    print(f"[seed] tickets.db created — {len(TICKETS)} tickets")

    # ── kb_articles.json ─────────────────────────────────────────────────────
    kb_path = FIXTURES / "kb_articles.json"
    kb_path.write_text(json.dumps(KB_ARTICLES, indent=2, ensure_ascii=False))
    print(f"[seed] kb_articles.json created — {len(KB_ARTICLES)} articles")

    # ── users.json ───────────────────────────────────────────────────────────
    users_path = FIXTURES / "users.json"
    users_path.write_text(json.dumps(USERS, indent=2, ensure_ascii=False))
    print(f"[seed] users.json created — {len(USERS)} users")

    print("\n[seed] All fixtures ready.")
    print("  Next: python agent/jira_triage.py PROD-2847")


if __name__ == "__main__":
    seed()
