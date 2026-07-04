"""FastAPI service for the prompt experimentation platform."""

from __future__ import annotations

import hashlib
import random
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .experiments import Experiments
from .registry import Registry

app = FastAPI(title="Prompt Versioning and A/B Testing", version="1.0.0")
registry = Registry()
experiments = Experiments(registry.conn, registry)


class CreatePrompt(BaseModel):
    name: str
    template: str
    commit_message: str = "initial version"


class NewVersion(BaseModel):
    template: str
    commit_message: str


class CreateExperiment(BaseModel):
    name: str
    prompt_id: int
    variants: list[dict]        # [{"version": 1, "split": 50}, ...]
    primary_metric: str = "quality"
    target_samples: int = 200


class Completion(BaseModel):
    prompt_id: int
    variables: dict = {}
    user_id: str


def simulated_llm(filled_prompt: str) -> dict:
    """Offline stand-in for the provider call; returns metrics that depend
    deterministically on the prompt content so experiments have signal."""
    seed = int(hashlib.sha256(filled_prompt.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    quality_bias = 0.6 + 0.1 * ("step by step" in filled_prompt.lower()) \
        + 0.08 * ("examples:" in filled_prompt.lower())
    return {"text": f"response-{seed % 10**6}",
            "quality": min(1.0, quality_bias + rng.uniform(-0.15, 0.15)),
            "latency_s": 0.5 + 0.4 * ("step by step" in filled_prompt.lower())
            + rng.uniform(0, 0.2),
            "tokens": len(filled_prompt.split()) + rng.randint(30, 90)}


@app.post("/prompts")
def create_prompt(request: CreatePrompt):
    prompt_id = registry.create_prompt(request.name, request.template,
                                       request.commit_message)
    return {"prompt_id": prompt_id, "version": 1}


@app.post("/prompts/{prompt_id}/versions")
def add_version(prompt_id: int, request: NewVersion):
    version = registry.add_version(prompt_id, request.template,
                                   request.commit_message)
    return {"version": version}


@app.get("/prompts/{prompt_id}/versions")
def list_versions(prompt_id: int):
    return registry.list_versions(prompt_id)


@app.get("/prompts/{prompt_id}/diff")
def diff(prompt_id: int, a: int, b: int):
    try:
        return {"diff": registry.diff(prompt_id, a, b)}
    except KeyError:
        raise HTTPException(404, "version not found")


@app.post("/prompts/{prompt_id}/activate/{version}")
def activate(prompt_id: int, version: int):
    try:
        registry.activate(prompt_id, version, actor="api", reason="manual")
    except KeyError:
        raise HTTPException(404, "version not found")
    return {"active_version": version}


@app.post("/experiments")
def create_experiment(request: CreateExperiment):
    experiment_id = experiments.create(
        request.name, request.prompt_id, request.variants,
        request.primary_metric, request.target_samples)
    return {"experiment_id": experiment_id}


@app.post("/v1/completions")
def completions(request: Completion):
    experiment = experiments.running_for_prompt(request.prompt_id)
    if experiment:
        version = experiments.assign(experiment, request.user_id)
    else:
        version = registry.active_version(request.prompt_id)
    record = registry.get_version(request.prompt_id, version)
    filled = registry.fill(record["template"], request.variables)

    start = time.perf_counter()
    result = simulated_llm(filled)
    result["latency_s"] += time.perf_counter() - start

    if experiment:
        experiments.record(experiment["id"], request.user_id, version,
                           {"quality": result["quality"],
                            "latency_s": result["latency_s"],
                            "tokens": result["tokens"]})
        experiments.check_guardrails(experiment["id"])
    return {"text": result["text"], "prompt_version": version,
            "experiment": experiment["name"] if experiment else None}


@app.get("/experiments/{experiment_id}/results")
def results(experiment_id: int):
    if experiments.get(experiment_id) is None:
        raise HTTPException(404, "experiment not found")
    return experiments.results(experiment_id)


@app.get("/audit")
def audit():
    return registry.audit_log()
