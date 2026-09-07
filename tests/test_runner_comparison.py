"""Tests for run_comparison: multi-run variance + cross-model aggregation."""

import pytest

from omagent.session import OMSession
from omagent.runner import run_comparison

from tests.fakes import FakeOMC
from tests.test_loop import GOOD, BAD, CHECK_OK, ScriptedLLM

SIM_OK = {"resultFile": "RESULT", "messages": "LOG_SUCCESS | info | The simulation finished successfully.\n"}
SIM_FAIL = {"resultFile": "", "messages": "Simulation execution failed for model: T\n"}


@pytest.fixture()
def comparison_session(tmp_path):
    csv = tmp_path / "res.csv"
    csv.write_text(
        "time,x,v\n" + "".join(
            f"{t/10:.1f},{0.1 if t <= 5 else 0.0:.3f},0\n"
            for t in range(0, 101)))
    ok = dict(SIM_OK, resultFile=str(csv))
    # One shared fake: strong consumes sim-ok replies, weak consumes sim-fail.
    fake = FakeOMC({
        "loadString(": [True] * 20,
        "checkModel(": [CHECK_OK] * 20,
        "simulate(": [ok] * 2 + [dict(SIM_FAIL)] * 18,
        "getErrorString()": ['""'] * 100,
    })
    return lambda: OMSession(fake)


def test_comparison_arguments_validated():
    with pytest.raises(ValueError):
        run_comparison(lambda: None, {})
    with pytest.raises(ValueError):
        run_comparison(lambda: None, {"m": lambda: None}, repeats=0)


def test_comparison_two_models_two_repeats(comparison_session, tmp_path):
    comparison = run_comparison(
        session_factory=comparison_session,
        llm_factories={
            "strong": lambda: ScriptedLLM([GOOD]),
            "weak": lambda: ScriptedLLM([BAD, BAD, BAD]),
        },
        task_ids=["msd_equations"],
        repeats=2,
        out_dir=str(tmp_path / "tr"),
        max_attempts=3,
    )

    assert comparison["repeats"] == 2
    strong = comparison["models"]["strong"]
    weak = comparison["models"]["weak"]

    assert strong["total"] == weak["total"] == 2
    assert strong["passed"] == 2 and strong["pass_rate"] == 1.0
    assert weak["passed"] == 0 and weak["pass_rate"] == 0.0

    ts = strong["per_task"]["msd_equations"]
    assert ts["runs"] == 2 and ts["passed"] == 2 and ts["pass_rate"] == 1.0
    assert ts["attempts_mean"] == 1.0 and ts["attempts_stdev"] == 0.0
    assert ts["final_stages"] == ["ok"]
    assert ts["tier"] == 1
    assert ts["elapsed_mean_s"] >= 0.0

    ws = weak["per_task"]["msd_equations"]
    assert ws["passed"] == 0 and ws["final_stages"] == ["simulate"]
    assert ws["attempts_mean"] == 3.0 and ws["attempts_stdev"] == 0.0


def test_comparison_transcripts_per_model_rep(comparison_session, tmp_path):
    run_comparison(
        session_factory=comparison_session,
        llm_factories={"m1": lambda: ScriptedLLM([GOOD])},
        task_ids=["msd_equations"],
        repeats=2,
        out_dir=str(tmp_path / "tr"),
        max_attempts=2,
    )
    root = tmp_path / "tr"
    assert (root / "comparison.json").exists()
    for rep in ("rep1", "rep2"):
        assert (root / "m1" / rep / "msd_equations.json").exists()
        assert (root / "m1" / rep / "summary.json").exists()


def test_comparison_verbose_output(comparison_session, tmp_path, capsys):
    run_comparison(
        session_factory=comparison_session,
        llm_factories={"m1": lambda: ScriptedLLM([GOOD])},
        task_ids=["msd_equations"],
        repeats=1,
        out_dir=str(tmp_path / "tr"),
        max_attempts=2,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "m1 rep 1/1" in out
    assert "1/1 passed" in out
    assert "m1: 1/1 (100%)" in out


def test_comparison_attempt_variance_tracked(tmp_path):
    csv = tmp_path / "res.csv"
    csv.write_text(
        "time,x,v\n" + "".join(
            f"{t/10:.1f},{0.1 if t <= 5 else 0.0:.3f},0\n"
            for t in range(0, 101)))
    ok = dict(SIM_OK, resultFile=str(csv))
    # BAD attempt 1 fails at load; GOOD attempt 2 passes. The shared fake
    # serves both repeats: error / ok-ok-ok / error / ok-ok-ok drains.
    fake = FakeOMC({
        "loadString(": [True] * 10,
        "checkModel(": [CHECK_OK] * 10,
        "simulate(": [ok] * 10,
        "getErrorString()": (['"Error: Class FooX not found in scope M.\n"'] + ['""'] * 3) * 6,
    })
    comparison = run_comparison(
        session_factory=lambda: OMSession(fake),
        llm_factories={"var": lambda: ScriptedLLM([BAD, GOOD])},
        task_ids=["msd_equations"],
        repeats=2,
        out_dir=str(tmp_path / "tr"),
        max_attempts=2,
    )
    ts = comparison["models"]["var"]["per_task"]["msd_equations"]
    assert ts["passed"] == 2
    assert ts["attempts_mean"] == 2.0
    assert ts["attempts_stdev"] == 0.0
