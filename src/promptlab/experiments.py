"""Experiment engine: variants, deterministic traffic split, guardrails."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from .stats import min_detectable_effect, welch_t_test


class Experiments:
    def __init__(self, conn: sqlite3.Connection, registry):
        self.conn = conn
        self.registry = registry
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            prompt_id INTEGER NOT NULL,
            variants TEXT NOT NULL,       -- JSON [{version, split}, ...]
            primary_metric TEXT NOT NULL,
            target_samples INTEGER NOT NULL,
            higher_is_better INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'running',
            winner INTEGER);
        CREATE TABLE IF NOT EXISTS observations (
            experiment_id INTEGER NOT NULL,
            unit_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            metrics TEXT NOT NULL,        -- JSON {metric: value}
            error INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL);
        """)
        self.conn.commit()

    def create(self, name: str, prompt_id: int, variants: list[dict],
               primary_metric: str, target_samples: int = 200,
               higher_is_better: bool = True) -> int:
        total = sum(v["split"] for v in variants)
        if abs(total - 100) > 1e-6:
            raise ValueError("traffic splits must sum to 100")
        cur = self.conn.execute(
            "INSERT INTO experiments (name, prompt_id, variants,"
            " primary_metric, target_samples, higher_is_better)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (name, prompt_id, json.dumps(variants), primary_metric,
             target_samples, int(higher_is_better)))
        self.conn.commit()
        self.registry.log("system", "experiment_created",
                          f"{name} on prompt {prompt_id}")
        return int(cur.lastrowid)

    def get(self, experiment_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, prompt_id, variants, primary_metric,"
            " target_samples, higher_is_better, status, winner"
            " FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            return None
        keys = ["id", "name", "prompt_id", "variants", "primary_metric",
                "target_samples", "higher_is_better", "status", "winner"]
        record = dict(zip(keys, row))
        record["variants"] = json.loads(record["variants"])
        return record

    def running_for_prompt(self, prompt_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id FROM experiments WHERE prompt_id = ? AND"
            " status = 'running' ORDER BY id DESC LIMIT 1",
            (prompt_id,)).fetchone()
        return self.get(row[0]) if row else None

    def assign(self, experiment: dict, unit_id: str) -> int:
        """Consistent hashing: the same unit always gets the same variant."""
        digest = hashlib.sha256(
            f"{experiment['id']}:{unit_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        cumulative = 0
        for variant in experiment["variants"]:
            cumulative += variant["split"]
            if bucket < cumulative:
                return variant["version"]
        return experiment["variants"][-1]["version"]

    def record(self, experiment_id: int, unit_id: str, version: int,
               metrics: dict, error: bool = False) -> None:
        self.conn.execute(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, unit_id, version, json.dumps(metrics),
             int(error), time.time()))
        self.conn.commit()

    def _metric_values(self, experiment_id: int, version: int,
                       metric: str) -> list[float]:
        rows = self.conn.execute(
            "SELECT metrics FROM observations WHERE experiment_id = ?"
            " AND version = ? AND error = 0", (experiment_id, version))
        values = []
        for (payload,) in rows:
            data = json.loads(payload)
            if metric in data:
                values.append(float(data[metric]))
        return values

    def error_rate(self, experiment_id: int, version: int) -> float:
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(error) FROM observations"
            " WHERE experiment_id = ? AND version = ?",
            (experiment_id, version)).fetchone()
        total, errors = row[0], row[1] or 0
        return errors / total if total else 0.0

    def check_guardrails(self, experiment_id: int,
                         error_threshold: float = 0.10) -> str | None:
        experiment = self.get(experiment_id)
        for variant in experiment["variants"]:
            rate = self.error_rate(experiment_id, variant["version"])
            count = self.conn.execute(
                "SELECT COUNT(*) FROM observations WHERE experiment_id = ?"
                " AND version = ?",
                (experiment_id, variant["version"])).fetchone()[0]
            if count >= 30 and rate > error_threshold:
                self.conn.execute(
                    "UPDATE experiments SET status = 'stopped' WHERE id = ?",
                    (experiment_id,))
                self.conn.commit()
                self.registry.log("system", "experiment_auto_stopped",
                                  f"experiment {experiment_id}: variant"
                                  f" v{variant['version']} error rate"
                                  f" {rate:.1%}")
                return (f"auto-stopped: variant v{variant['version']}"
                        f" error rate {rate:.1%} exceeds"
                        f" {error_threshold:.0%}")
        return None

    def results(self, experiment_id: int, alpha: float = 0.05) -> dict:
        experiment = self.get(experiment_id)
        metric = experiment["primary_metric"]
        versions = [v["version"] for v in experiment["variants"]]
        samples = {v: self._metric_values(experiment_id, v, metric)
                   for v in versions}
        baseline = versions[0]
        comparisons = {}
        for version in versions[1:]:
            test = welch_t_test(samples[baseline], samples[version], alpha)
            comparisons[f"v{baseline}_vs_v{version}"] = {
                **vars(test),
                "mde": min_detectable_effect(samples[baseline],
                                             samples[version])}

        status = "inconclusive"
        winner = None
        target = experiment["target_samples"]
        reached_target = all(len(s) >= target for s in samples.values())
        significant = [key for key, c in comparisons.items()
                       if c["significant"]]
        if significant and reached_target:
            better = max if experiment["higher_is_better"] else min
            means = {v: (sum(s) / len(s)) if s else 0.0
                     for v, s in samples.items()}
            winner = better(means, key=means.get)
            status = "winner"
        elif reached_target:
            status = "no_winner"

        return {"experiment": experiment["name"],
                "primary_metric": metric,
                "samples": {f"v{v}": len(s) for v, s in samples.items()},
                "means": {f"v{v}": round(sum(s) / len(s), 4) if s else None
                          for v, s in samples.items()},
                "comparisons": comparisons,
                "status": status,
                "winner": winner,
                "target_samples": target}

    def declare_winner(self, experiment_id: int, winner: int,
                       promote: bool = True) -> None:
        experiment = self.get(experiment_id)
        self.conn.execute(
            "UPDATE experiments SET status = 'completed', winner = ?"
            " WHERE id = ?", (winner, experiment_id))
        self.conn.commit()
        self.registry.log("system", "winner_declared",
                          f"experiment {experiment_id}: v{winner}")
        if promote:
            self.registry.activate(experiment["prompt_id"], winner,
                                   actor="system",
                                   reason=f"experiment {experiment_id} winner")
