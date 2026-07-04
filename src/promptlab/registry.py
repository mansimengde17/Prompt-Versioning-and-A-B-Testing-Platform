"""Prompt registry: versions, diffs, rollback, templates, audit log."""

from __future__ import annotations

import difflib
import re
import sqlite3
import time


class Registry:
    def __init__(self, path: str = "promptlab.db"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            active_version INTEGER);
        CREATE TABLE IF NOT EXISTS versions (
            prompt_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            template TEXT NOT NULL,
            params TEXT NOT NULL DEFAULT '{}',
            commit_message TEXT NOT NULL,
            created REAL NOT NULL,
            PRIMARY KEY (prompt_id, version));
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL);
        """)
        self.conn.commit()

    def log(self, actor: str, action: str, detail: str) -> None:
        self.conn.execute(
            "INSERT INTO audit (timestamp, actor, action, detail)"
            " VALUES (?, ?, ?, ?)", (time.time(), actor, action, detail))
        self.conn.commit()

    def create_prompt(self, name: str, template: str, commit_message: str,
                      actor: str = "system") -> int:
        cur = self.conn.execute(
            "INSERT INTO prompts (name, active_version) VALUES (?, 1)",
            (name,))
        prompt_id = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO versions VALUES (?, 1, ?, '{}', ?, ?)",
            (prompt_id, template, commit_message, time.time()))
        self.conn.commit()
        self.log(actor, "prompt_created", f"{name} v1")
        return prompt_id

    def add_version(self, prompt_id: int, template: str,
                    commit_message: str, actor: str = "system") -> int:
        (latest,) = self.conn.execute(
            "SELECT MAX(version) FROM versions WHERE prompt_id = ?",
            (prompt_id,)).fetchone()
        version = (latest or 0) + 1
        self.conn.execute(
            "INSERT INTO versions VALUES (?, ?, ?, '{}', ?, ?)",
            (prompt_id, version, template, commit_message, time.time()))
        self.conn.commit()
        self.log(actor, "version_added", f"prompt {prompt_id} v{version}:"
                                         f" {commit_message}")
        return version

    def get_version(self, prompt_id: int, version: int) -> dict | None:
        row = self.conn.execute(
            "SELECT template, commit_message, created FROM versions"
            " WHERE prompt_id = ? AND version = ?",
            (prompt_id, version)).fetchone()
        if row is None:
            return None
        return {"version": version, "template": row[0],
                "commit_message": row[1], "created": row[2]}

    def list_versions(self, prompt_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT version, commit_message, created FROM versions"
            " WHERE prompt_id = ? ORDER BY version", (prompt_id,)).fetchall()
        return [{"version": v, "commit_message": m, "created": c}
                for v, m, c in rows]

    def diff(self, prompt_id: int, a: int, b: int) -> str:
        va, vb = self.get_version(prompt_id, a), self.get_version(prompt_id, b)
        if va is None or vb is None:
            raise KeyError("version not found")
        return "\n".join(difflib.unified_diff(
            va["template"].splitlines(), vb["template"].splitlines(),
            fromfile=f"v{a}", tofile=f"v{b}", lineterm=""))

    def activate(self, prompt_id: int, version: int, actor: str = "system",
                 reason: str = "") -> None:
        if self.get_version(prompt_id, version) is None:
            raise KeyError("version not found")
        self.conn.execute(
            "UPDATE prompts SET active_version = ? WHERE id = ?",
            (version, prompt_id))
        self.conn.commit()
        self.log(actor, "version_activated",
                 f"prompt {prompt_id} -> v{version}"
                 + (f" ({reason})" if reason else ""))

    def active_version(self, prompt_id: int) -> int:
        (version,) = self.conn.execute(
            "SELECT active_version FROM prompts WHERE id = ?",
            (prompt_id,)).fetchone()
        return version

    @staticmethod
    def fill(template: str, variables: dict) -> str:
        required = set(re.findall(r"{{\s*(\w+)\s*}}", template))
        missing = required - set(variables)
        if missing:
            raise ValueError(f"missing template variables: {sorted(missing)}")
        filled = template
        for key in required:
            filled = re.sub(r"{{\s*" + key + r"\s*}}", str(variables[key]),
                            filled)
        return filled

    def audit_log(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, actor, action, detail FROM audit"
            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"timestamp": t, "actor": a, "action": ac, "detail": d}
                for t, a, ac, d in rows]
