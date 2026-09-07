"""Run the benchmark task ladder and persist transcripts.

Each task gets a fresh session and a fresh LLM (factories), so tasks cannot
contaminate each other through loaded classes or conversation state.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import time
from typing import Callable, Optional

from .loop import AgentLoop, LLM, WarningGate
from .session import OMSession
from .tasks import get_tasks


def run_ladder(
    session_factory: Callable[[], OMSession],
    llm_factory: Callable[[], LLM],
    task_ids: Optional[list[str]] = None,
    max_tier: Optional[int] = None,
    out_dir: str = "transcripts",
    max_attempts: int = 4,
    warning_gate: Optional[WarningGate] = None,
    verbose: bool = False,
) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for task in get_tasks(task_ids, max_tier):
        if verbose:
            print(f"[tier {task.tier}] {task.id} ...", flush=True)
        llm = llm_factory()
        session = session_factory()
        t0 = time.time()

        env_failure = None
        if task.requires_msl:
            msl = session.load_msl()
            if not msl.success:
                env_failure = [d.brief() for d in msl.diagnostics] or \
                    ["loadModel(Modelica) returned false"]

        if env_failure is not None:
            res = None
            elapsed = time.time() - t0
        else:
            loop = AgentLoop(
                session, llm, max_attempts=max_attempts,
                simulate_options=dict(task.simulate_options),
                verifier=task.verifier,
                warning_gate=warning_gate)
            res = loop.run(task.prompt)
            elapsed = time.time() - t0

        record = {
            "task_id": task.id,
            "tier": task.tier,
            "prompt": task.prompt,
            "notes": task.notes,
            "success": bool(res and res.success),
            "model_name": res.model_name if res else None,
            "elapsed_s": round(elapsed, 2),
            "environment_failure": env_failure,
            "attempts": [
                {"n": a.n, "stage": a.stage, "complaint": a.complaint,
                 "diagnostics": [d.brief() for d in a.diagnostics],
                 "code": a.code}
                for a in res.attempts] if res else [],
            "llm_transcript": getattr(llm, "transcript", None),
        }
        (out / f"{task.id}.json").write_text(json.dumps(record, indent=2))

        summary_row = {
            "task": task.id,
            "tier": task.tier,
            "success": record["success"],
            "attempts": len(res.attempts) if res else 0,
            "final_stage": ("environment" if env_failure is not None else
                            (res.attempts[-1].stage if res and res.attempts
                             else None)),
            "elapsed_s": record["elapsed_s"],
        }
        results.append(summary_row)
        if verbose:
            if env_failure is not None:
                print("    FAIL (environment) — task not attempted:")
                for line in env_failure:
                    print(f"      {line}")
            else:
                mark = "PASS" if record["success"] else "FAIL"
                print(f"    {mark} in {len(res.attempts)} attempt(s), "
                      f"{elapsed:.1f}s, final stage: {summary_row['final_stage']}")

    report = {"results": results,
              "passed": sum(r["success"] for r in results),
              "total": len(results)}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def run_comparison(
    session_factory: Callable[[], OMSession],
    llm_factories: dict[str, Callable[[], LLM]],
    task_ids: Optional[list[str]] = None,
    max_tier: Optional[int] = None,
    repeats: int = 1,
    max_attempts: int = 4,
    out_dir: str = "transcripts",
    warning_gate: Optional[WarningGate] = None,
    verbose: bool = False,
) -> dict:
    """Run the ladder several times per LLM and compare models.

    Each (model, task, run) triple gets a fresh session and LLM, so runs are
    fully independent; variance across repeats captures LLM / omc
    nondeterminism rather than state contamination.

    Transcripts land in ``{out_dir}/{model}/rep{k}/{task}.json``; the
    aggregated ``comparison.json`` includes per-model pass rates, per-task
    success counts, and mean/variance of attempts and wall time.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not llm_factories:
        raise ValueError("llm_factories must not be empty")

    tasks = get_tasks(task_ids, max_tier)
    root = pathlib.Path(out_dir)
    models: dict[str, dict] = {}

    for name, llm_factory in llm_factories.items():
        per_task: dict[str, list[dict]] = {t.id: [] for t in tasks}
        for k in range(1, repeats + 1):
            run_dir = str(root / name / f"rep{k}")
            report = run_ladder(
                session_factory, llm_factory,
                task_ids=[t.id for t in tasks],
                out_dir=run_dir, max_attempts=max_attempts,
                warning_gate=warning_gate,
                verbose=False)
            for row in report["results"]:
                per_task[row["task"]].append(row)
            if verbose:
                passed = sum(r["success"] for r in report["results"])
                print(f"[{name} rep {k}/{repeats}] {passed}/"
                      f"{len(report['results'])} passed", flush=True)

        task_stats = {}
        for t in tasks:
            runs = per_task[t.id]
            attempts = [r["attempts"] for r in runs]
            elapsed = [r["elapsed_s"] for r in runs]
            n = len(runs)
            task_stats[t.id] = {
                "tier": t.tier,
                "runs": n,
                "passed": sum(r["success"] for r in runs),
                "pass_rate": sum(r["success"] for r in runs) / n if n else 0.0,
                "attempts_mean": statistics.fmean(attempts) if n else 0.0,
                "attempts_stdev": statistics.pstdev(attempts) if n > 1 else 0.0,
                "elapsed_mean_s": statistics.fmean(elapsed) if n else 0.0,
                "final_stages": sorted({r["final_stage"] for r in runs}),
            }

        total = sum(len(runs) for runs in per_task.values())
        passed = sum(s["passed"] for s in task_stats.values())
        models[name] = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "per_task": task_stats,
        }

    comparison = {"repeats": repeats, "models": models}
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2))

    if verbose:
        for name, m in models.items():
            print(f"{name}: {m['passed']}/{m['total']} "
                  f"({m['pass_rate']:.0%})")
            for tid, s in m["per_task"].items():
                print(f"    {tid:<24} {s['passed']}/{s['runs']}"
                      f"  attempts mean {s['attempts_mean']:.2f}"
                      f"  {s['elapsed_mean_s']:.1f}s")
    return comparison
