# Prompt Versioning and A/B Testing Platform

A platform that treats prompts as versioned artifacts, deploys multiple prompt variants simultaneously, splits traffic deterministically between them, measures performance across pluggable metrics, and declares statistically significant winners. Feature-flag rigor for LLM development.

Live demo: https://mansimengde17.github.io/Prompt-Versioning-and-A-B-Testing-Platform/

## The problem

Most AI teams change prompts by editing a string in production and hoping. Prompt changes are experiments and deserve experiment infrastructure: versioning, controlled traffic splits, metrics, significance testing, audit trails, and one-click rollback.

## Architecture

```
 +---------------------+       +------------------------+
 |   Prompt Registry    |       |   Experiment Engine     |
 |  versions, diffs,    |<----->|  variants, traffic      |
 |  rollback, template  |       |  split percentages,     |
 |  variables, audit    |       |  auto-stop guardrails   |
 +---------------------+       +------------------------+
            |                              |
            v                              v
 +--------------------------------------------------+
 |            POST /v1/completions                    |
 |  resolve active version or experiment variant     |
 |  (consistent hash on user_id: same user, same     |
 |  variant), fill template, call LLM, log            |
 +--------------------------------------------------+
            |
            v
 +--------------------------------------------------+
 |   Async metric collectors: latency, tokens,       |
 |   cost, task accuracy, LLM-as-judge score         |
 +--------------------------------------------------+
            |
            v
 +--------------------------------------------------+
 |   Statistics: Welch's t-test per metric,          |
 |   p-value, confidence interval, sample-size       |
 |   progress -> winner / no winner / inconclusive   |
 +--------------------------------------------------+
```

## Quick start (offline)

```bash
pip install -r requirements.txt
python demo.py
```

The demo seeds a customer-support classifier prompt with three variants (zero-shot, few-shot, chain-of-thought), routes 600 simulated requests through the splitter, collects metrics, runs the significance analysis, declares the winner, and promotes it to the active version with a full audit trail.

To serve the API:

```bash
uvicorn src.promptlab.api:app --reload
# POST /prompts                        create a prompt
# POST /prompts/{id}/versions          add a version (auto-increment, commit message)
# GET  /prompts/{id}/versions          list versions
# GET  /prompts/{id}/diff?a=1&b=2      unified diff between versions
# POST /prompts/{id}/activate/{v}      rollback or promote without redeploy
# POST /experiments                    create an experiment with traffic split
# POST /v1/completions                 serve a request (splitter decides variant)
# GET  /experiments/{id}/results       live stats and significance
# GET  /audit                          full audit log
```

## Statistical details

Per metric and per variant pair, the platform computes mean, variance, Welch's t statistic, a two-sided p-value, and the 95 percent confidence interval of the difference. An experiment declares a winner when the primary metric reaches significance at the configured level (default 95 percent) and the target sample size is met. Guardrails auto-stop an experiment when a variant's error rate crosses a threshold, and winner auto-promotion has a configurable hold period.

Deterministic assignment uses a consistent hash of the user or session identifier, so the same user always sees the same variant and results are unbiased by re-assignment.

## Storage

SQLite by default for zero-infrastructure runs; the store layer is a thin interface, and swapping in PostgreSQL for team deployments changes one connection string in `docker-compose.yml`.

## Repository layout

```
src/promptlab/registry.py     prompt versions, diffs, rollback, templates, audit
src/promptlab/experiments.py  experiments, consistent-hash splitter, guardrails
src/promptlab/stats.py        Welch's t-test, p-values, confidence intervals
src/promptlab/api.py          FastAPI service
demo.py                       seeded three-variant experiment to convergence
tests/
```
