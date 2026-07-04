#!/usr/bin/env python3
"""Offline demo: three prompt variants, 600 requests, statistical winner."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, "src")

from promptlab.experiments import Experiments
from promptlab.registry import Registry

ZERO_SHOT = """Classify the customer email into billing, technical, account,
or general and return JSON. Email: {{email}}"""

FEW_SHOT = """Classify the customer email into billing, technical, account,
or general and return JSON.
Examples:
"I was charged twice" -> billing
"The app crashes on load" -> technical
Email: {{email}}"""

CHAIN_OF_THOUGHT = """Classify the customer email into billing, technical,
account, or general. Think step by step about the customer's primary
blocker before answering, then return JSON. Email: {{email}}"""


def simulated_llm(filled: str, rng: random.Random) -> dict:
    quality = 0.60 \
        + 0.10 * ("step by step" in filled.lower()) \
        + 0.08 * ("examples:" in filled.lower()) \
        + rng.uniform(-0.15, 0.15)
    latency = 0.5 + 0.4 * ("step by step" in filled.lower()) \
        + rng.uniform(0, 0.2)
    return {"quality": min(1.0, quality), "latency_s": latency,
            "tokens": len(filled.split()) + rng.randint(30, 90)}


def main() -> None:
    if os.path.exists("promptlab.db"):
        os.remove("promptlab.db")
    rng = random.Random(3)

    registry = Registry()
    experiments = Experiments(registry.conn, registry)

    prompt_id = registry.create_prompt(
        "support-email-classifier", ZERO_SHOT, "zero-shot baseline")
    registry.add_version(prompt_id, FEW_SHOT, "add few-shot examples")
    registry.add_version(prompt_id, CHAIN_OF_THOUGHT,
                         "chain-of-thought variant")

    print("Diff v1 -> v3:")
    print(registry.diff(prompt_id, 1, 3)[:400], "...\n")

    experiment_id = experiments.create(
        "classifier-prompt-bakeoff", prompt_id,
        [{"version": 1, "split": 34}, {"version": 2, "split": 33},
         {"version": 3, "split": 33}],
        primary_metric="quality", target_samples=150)

    for i in range(600):
        user = f"user-{rng.randint(1, 400)}"
        experiment = experiments.get(experiment_id)
        version = experiments.assign(experiment, user)
        template = registry.get_version(prompt_id, version)["template"]
        filled = registry.fill(template, {"email": f"sample email {i}"})
        metrics = simulated_llm(filled, rng)
        experiments.record(experiment_id, user, version, metrics)

    # Consistency check: the same user is always assigned the same variant.
    experiment = experiments.get(experiment_id)
    assert all(experiments.assign(experiment, f"user-{u}")
               == experiments.assign(experiment, f"user-{u}")
               for u in range(50))

    results = experiments.results(experiment_id)
    print(f"Samples: {results['samples']}")
    print(f"Means ({results['primary_metric']}): {results['means']}")
    for pair, comparison in results["comparisons"].items():
        print(f"{pair}: diff={comparison['diff']:+.4f}"
              f" p={comparison['p_value']:.5f}"
              f" ci=[{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]"
              f" significant={comparison['significant']}")
    print(f"Status: {results['status']}, winner: v{results['winner']}")

    if results["status"] == "winner":
        experiments.declare_winner(experiment_id, results["winner"])
        print(f"Promoted v{results['winner']} to active"
              f" (active_version={registry.active_version(prompt_id)})")

    print("\nAudit log:")
    for entry in reversed(registry.audit_log(10)):
        print(f"  [{entry['action']}] {entry['detail']}")


if __name__ == "__main__":
    main()
