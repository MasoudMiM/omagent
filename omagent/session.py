"""Thin, testable wrapper around an OpenModelica compiler session.

Design notes
------------
- The wrapper never talks to omc directly; it calls a `backend` — any object
  with a ``sendExpression(str) -> object`` method. In production that's
  OMPython's ``OMCSessionZMQ``; in tests it's a fake replaying recorded output.
- Every operation returns a structured ``OpResult`` carrying success, the raw
  omc value, and parsed ``Diagnostic`` records, so the future agent loop can
  reason uniformly over outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .errors import (Diagnostic, Severity, parse_error_string,
                     parse_ompython_exception, parse_simulation_messages)


class Backend(Protocol):
    def sendExpression(self, expr: str) -> Any: ...


@dataclass
class OpResult:
    op: str
    success: bool
    value: Any = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _option_value(v: Any) -> str:
    """Render a simulate() option as a Modelica literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return _quote(v)
    return str(v)


class OMSession:
    """High-level operations over an omc backend."""

    def __init__(self, backend: Optional[Backend] = None):
        if backend is None:
            # Lazy import so the package is usable (and testable) without OMPython.
            from OMPython import OMCSessionZMQ  # pragma: no cover
            backend = OMCSessionZMQ()           # pragma: no cover
        self._omc = backend

    # -- plumbing ---------------------------------------------------------
    def _send(self, expr: str) -> Any:
        return self._omc.sendExpression(expr)

    def _safe_send(self, expr: str) -> tuple[Any, list[Diagnostic]]:
        """Send an expression; convert backend exceptions into diagnostics.

        Newer OMPython raises OMCSessionException when omc logs any
        error-level message, instead of returning and letting the caller read
        getErrorString(). Both contracts must yield a failed OpResult, never
        an uncaught exception.
        """
        try:
            return self._send(expr), []
        except Exception as exc:  # OMCSessionException, zmq errors, ...
            return None, parse_ompython_exception(str(exc))

    def _drain_diagnostics(self) -> list[Diagnostic]:
        try:
            # OMPython warns if getErrorString() is requested parsed; ask for
            # raw output when the backend supports it (fakes may not).
            try:
                raw = self._omc.sendExpression("getErrorString()", parsed=False)
            except TypeError:
                raw = self._send("getErrorString()")
        except Exception as exc:
            return parse_ompython_exception(str(exc))
        return parse_error_string(raw if isinstance(raw, str) else str(raw))

    # -- operations -------------------------------------------------------
    def get_version(self) -> str:
        v = self._send("getVersion()")
        return str(v).strip('"')

    def load_msl(self, version: str = "") -> OpResult:
        expr = "loadModel(Modelica)" if not version else \
            f"loadModel(Modelica, {{{ _quote(version) }}})"
        val, exc_diags = self._safe_send(expr)
        diags = exc_diags + self._drain_diagnostics()
        ok = bool(val) and not any(d.severity == Severity.ERROR for d in diags)
        return OpResult("load_msl", ok, val, diags)

    def load_file(self, path: str) -> OpResult:
        val, exc_diags = self._safe_send(f"loadFile({_quote(path)})")
        diags = exc_diags + self._drain_diagnostics()
        # loadFile can return true while still emitting errors for bad models.
        has_err = any(d.severity == Severity.ERROR for d in diags)
        return OpResult("load_file", bool(val) and not has_err, val, diags)

    def load_string(self, modelica_code: str) -> OpResult:
        val, exc_diags = self._safe_send(f"loadString({_quote(modelica_code)})")
        diags = exc_diags + self._drain_diagnostics()
        has_err = any(d.severity == Severity.ERROR for d in diags)
        return OpResult("load_string", bool(val) and not has_err, val, diags)

    def check_model(self, model: str) -> OpResult:
        val, exc_diags = self._safe_send(f"checkModel({model})")
        text = str(val).strip('"') if val is not None else ""
        diags = exc_diags + self._drain_diagnostics()
        # checkModel reports success inside its return string.
        ok = "completed successfully" in text and not any(
            d.severity == Severity.ERROR for d in diags)
        if not ok and text and "completed successfully" not in text:
            extra = parse_error_string(text)
            known = {d.message for d in diags}
            diags = diags + [d for d in extra if d.message not in known]
        return OpResult("check_model", ok, text, diags)

    def simulate(self, model: str, **options: Any) -> OpResult:
        opts = "".join(f", {k}={_option_value(v)}" for k, v in options.items())
        val, exc_diags = self._safe_send(f"simulate({model}{opts})")
        diags = exc_diags + self._drain_diagnostics()
        messages, result_file = "", ""
        if isinstance(val, dict):
            messages = str(val.get("messages", ""))
            result_file = str(val.get("resultFile", ""))
        diags += parse_simulation_messages(messages)
        run_failed = ("Simulation execution failed" in messages
                      or "simulation terminated" in messages.lower())
        ok = bool(result_file) and not run_failed and not any(
            d.severity == Severity.ERROR for d in diags)
        return OpResult("simulate", ok, val, diags)

    def class_names(self, class_: str) -> list[str]:
        """Names of classes contained in `class_` (empty on any failure).

        Used to turn 'Class X not found' into actionable suggestions by
        asking omc what actually exists in the parent package.
        """
        val, exc_diags = self._safe_send(f"getClassNames({class_})")
        if exc_diags or val is None:
            return []
        self._drain_diagnostics()   # discard lookup noise
        if isinstance(val, (list, tuple)):
            return [str(v) for v in val]
        return []
