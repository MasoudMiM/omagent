"""Agentic generate -> load -> check -> simulate -> verify loop.

The LLM is behind a narrow protocol so the loop is unit-testable with a fake
and backend-agnostic in production (Anthropic, OpenAI, local models — anything
that can implement `propose`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import difflib

from .errors import Diagnostic, Kind, summarize_for_llm
from .session import OMSession, OpResult


class LLM(Protocol):
    def propose(
        self,
        task: str,
        previous_code: Optional[str],
        error_summary: Optional[str],
    ) -> str:
        """Return complete Modelica code for `task`.

        First call: previous_code/error_summary are None (fresh generation).
        Later calls: both are set (repair a failed attempt).
        """
        ...


# A verifier inspects the simulate() OpResult; returns None if satisfied,
# else a human/LLM-readable complaint that is fed into the next fix round.
Verifier = Callable[[OpResult], Optional[str]]

# A quality gate inspects a whole attempt's diagnostics (warnings included);
# returns None when the model passes, else a complaint that is fed back.
WarningGate = Callable[[list[Diagnostic]], Optional[str]]

_MODEL_NAME = re.compile(r"^\s*(?:model|block|package)\s+([A-Za-z_][A-Za-z0-9_]*)",
                         re.MULTILINE)

_CODE_FENCE = re.compile(r"```(?:modelica|mo)?\s*\n(.*?)```", re.DOTALL)


def extract_code(llm_output: str) -> str:
    """Accept raw Modelica or a fenced markdown block; return bare code."""
    m = _CODE_FENCE.search(llm_output)
    return (m.group(1) if m else llm_output).strip()


def extract_model_name(code: str) -> Optional[str]:
    m = _MODEL_NAME.search(code)
    return m.group(1) if m else None


@dataclass
class Attempt:
    n: int
    code: str
    stage: str   # "load" | "check" | "simulate" | "quality" | "verify" | "ok"
    diagnostics: list[Diagnostic] = field(default_factory=list)
    complaint: Optional[str] = None  # verifier feedback, if any

    @property
    def failed(self) -> bool:
        return self.stage != "ok"


@dataclass
class LoopResult:
    success: bool
    attempts: list[Attempt]
    model_name: Optional[str] = None
    sim_result: Optional[OpResult] = None

    @property
    def final_code(self) -> Optional[str]:
        return self.attempts[-1].code if self.attempts else None


class AgentLoop:
    def __init__(
        self,
        session: OMSession,
        llm: LLM,
        max_attempts: int = 4,
        simulate_options: Optional[dict] = None,
        verifier: Optional[Verifier] = None,
        warning_gate: Optional[WarningGate] = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.session = session
        self.llm = llm
        self.max_attempts = max_attempts
        self.simulate_options = simulate_options or {}
        self.verifier = verifier
        self.warning_gate = warning_gate

    def run(self, task: str, model_name: Optional[str] = None) -> LoopResult:
        attempts: list[Attempt] = []
        prev_code: Optional[str] = None
        feedback: Optional[str] = None

        for n in range(1, self.max_attempts + 1):
            code = extract_code(self.llm.propose(task, prev_code, feedback))
            name = model_name or extract_model_name(code)
            if not name:
                att = Attempt(n, code, "load")
                att.complaint = ("Could not determine the model name: the code "
                                 "must contain a top-level `model <Name>`.")
                attempts.append(att)
                prev_code, feedback = code, att.complaint
                continue

            att, sim = self._try_once(n, code, name)
            attempts.append(att)
            if not att.failed:
                return LoopResult(True, attempts, name, sim)
            prev_code = code
            feedback = self._feedback(att)

        return LoopResult(False, attempts, model_name or
                          extract_model_name(attempts[-1].code))

    # -- internals --------------------------------------------------------
    def _try_once(self, n: int, code: str, name: str):
        all_diags: list[Diagnostic] = []
        res = self.session.load_string(code)
        all_diags += res.diagnostics
        if not res.success:
            return Attempt(n, code, "load", all_diags), None

        res = self.session.check_model(name)
        all_diags += res.diagnostics
        if not res.success:
            return Attempt(n, code, "check", all_diags), None

        sim = self.session.simulate(name, **self.simulate_options)
        all_diags += sim.diagnostics
        if not sim.success:
            return Attempt(n, code, "simulate", all_diags), None

        # Quality gate first: warnings that indicate sloppy models must be
        # fixed even when the sim was fine; complaints feed the fix prompt.
        if self.warning_gate is not None:
            complaint = self.warning_gate(all_diags)
            if complaint:
                att = Attempt(n, code, "quality", all_diags, complaint)
                return att, sim

        if self.verifier is not None:
            complaint = self.verifier(sim)
            if complaint:
                return Attempt(n, code, "verify", all_diags, complaint), sim

        return Attempt(n, code, "ok", all_diags), sim

    def _feedback(self, att: Attempt) -> str:
        parts = [f"Attempt failed at stage '{att.stage}'."]
        if att.diagnostics:
            parts.append(summarize_for_llm(att.diagnostics))
        parts.extend(self._lookup_suggestions(att.diagnostics))
        if att.complaint:
            parts.append(f"Verification feedback: {att.complaint}")
        return "\n".join(parts)

    _MISSING_CLASS = re.compile(r"(?:Class|Import)\s+([A-Za-z_][\w.]*\.[\w]+)"
                                r"\s+not found")

    def _lookup_suggestions(self, diags: list[Diagnostic],
                            max_packages: int = 3) -> list[str]:
        """For lookup failures, ask omc what the parent package really
        contains and surface close matches — turns 'X not found' into
        'did you mean RotationalEMF' (e.g. MSL 3.2 -> 4.x renames)."""
        out: list[str] = []
        seen: set[str] = set()
        for d in diags:
            if d.kind != Kind.LOOKUP:
                continue
            m = self._MISSING_CLASS.search(d.message)
            if not m:
                continue
            fqn = m.group(1)
            parent, _, leaf = fqn.rpartition(".")
            if not parent or parent in seen:
                continue
            seen.add(parent)
            names = self.session.class_names(parent)
            if not names:
                continue
            close = difflib.get_close_matches(leaf, names, n=5, cutoff=0.4)
            sub = [n for n in names
                   if leaf.lower() in n.lower() and n not in close]
            picks = (close + sub)[:6] or sorted(names)[:12]
            out.append(
                f"Hint: {fqn} does not exist in the loaded libraries. "
                f"{parent} actually contains: {', '.join(picks)}. "
                f"Use one of these exact names.")
            if len(seen) >= max_packages:
                break
        return out
