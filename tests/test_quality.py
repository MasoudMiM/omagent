"""Tests for warning-level quality gates (opt-in, loop + gate function)."""

import inspect

from omagent.errors import (Diagnostic, Severity, warning_gate_complaints)
from omagent.session import OMSession
from omagent.loop import AgentLoop
from omagent.runner import run_ladder

from tests.fakes import FakeOMC
from tests.test_loop import GOOD, CHECK_OK, SIM_OK, ScriptedLLM

INIT_WARN = ('"[<interactive>:1:1-1:1:writable] Warning: The initial '
             'conditions are not fully specified. For model T: the following '
             'variables have no particular value: v\n"')


def _warning_diag(msg: str) -> Diagnostic:
    return Diagnostic(Severity.WARNING, msg)


def test_gate_matches_initialization_warning():
    complaint = warning_gate_complaints(
        [_warning_diag("The initial conditions are not fully specified.")])
    assert complaint is not None
    assert "initial conditions not fully specified" in complaint


def test_gate_ignores_clean_and_benign_warnings():
    assert warning_gate_complaints([]) is None
    assert warning_gate_complaints(
        [_warning_diag("some unrelated warning")]) is None


def test_gate_ignores_error_severity():
    # Errors are already fatal for ops; the gate is warning-level only.
    d = Diagnostic(Severity.ERROR, "The initial conditions are not fully specified.")
    assert warning_gate_complaints([d]) is None


def test_gate_labels_each_category():
    complaint = warning_gate_complaints([
        _warning_diag("The initial conditions are over-specified. "
                      "2 conflicting start values."),
        _warning_diag("The units of the expressions have to be equivalent."),
    ]) or ""
    assert "over- or inconsistently specified initial conditions" in complaint
    assert "inconsistent units" in complaint


def test_loop_without_gate_ignores_warnings():
    fake = FakeOMC({
        "loadString(": [True],
        "checkModel(": [CHECK_OK],
        "simulate(": [dict(SIM_OK)],
        "getErrorString()": [INIT_WARN, '""', '""'],
    })
    res = AgentLoop(OMSession(fake), ScriptedLLM([GOOD]), max_attempts=1).run("task")
    assert res.success
    assert res.attempts[0].stage == "ok"


def test_loop_with_gate_sends_warning_back(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("time,x,v\n0,0,0\n1,0,0\n")
    sim = dict(SIM_OK, resultFile=str(csv))
    fake = FakeOMC({
        "loadString(": [True] * 2,
        "checkModel(": [CHECK_OK] * 2,
        "simulate(": [dict(SIM_OK), dict(sim)],
        # drain order per attempt: attempt 1 -> warn, '', '' ; attempt 2 clea n
        "getErrorString()": [INIT_WARN, '""', '""', '""', '""', '""', '""'],
    })
    loop = AgentLoop(
        OMSession(fake), ScriptedLLM([GOOD, GOOD]), max_attempts=2,
        warning_gate=warning_gate_complaints)
    res = loop.run("task")

    assert len(res.attempts) == 2
    first = res.attempts[0]
    assert first.stage == "quality" and first.failed
    assert "initial conditions not fully specified" in (first.complaint or "")

    # The gated warning must reach the fix prompt as structured feedback.
    calls = loop.llm.calls
    assert calls[0][0] == "task"
    assert calls[1][1] == GOOD            # previous code attached
    assert "initial conditions" in calls[1][2]

    assert res.attempts[-1].stage == "ok"
    assert res.success


def test_ladder_with_gate_reports_quality_final_stage(tmp_path):
    fake = FakeOMC({
        "loadString(": [True] * 2,
        "checkModel(": [CHECK_OK] * 2,
        "simulate(": [dict(SIM_OK)] * 2,
        "getErrorString()": [INIT_WARN, '""', '""', '""', '""'] * 2,
    })
    report = run_ladder(
        session_factory=lambda: OMSession(fake),
        llm_factory=lambda: ScriptedLLM([GOOD, GOOD]),
        task_ids=["msd_equations"],
        out_dir=str(tmp_path / "t"),
        max_attempts=2,
        warning_gate=warning_gate_complaints,
    )
    r = report["results"][0]
    assert r["success"] is False
    assert r["final_stage"] == "quality"
    assert r["attempts"] == 2


def test_ladder_jit_accepts_gate_parameter():
    sig = inspect.signature(run_ladder)
    assert "warning_gate" in sig.parameters
    assert "verbose" in sig.parameters
