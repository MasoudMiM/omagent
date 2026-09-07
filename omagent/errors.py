"""Parser for OpenModelica compiler (omc) diagnostics.

Turns raw getErrorString()/simulate() output into structured records that an
LLM agent (or a human) can act on. This module is deliberately free of any
OMPython or network dependency so it can be unit-tested anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTIFICATION = "notification"


class Kind(str, Enum):
    """Coarse classification of what went wrong — used to steer fix strategies."""

    SYNTAX = "syntax"                # parse errors, missing tokens
    LOOKUP = "lookup"                # class/variable not found
    TYPE = "type"                    # type mismatches, unit issues
    BALANCE = "balance"              # equation/variable count mismatch, singular systems
    CONNECT = "connect"              # connector/connection problems
    INITIALIZATION = "initialization"
    RUNTIME = "runtime"              # simulation-time failures (solver, events, asserts)
    OTHER = "other"


# [/path/File.mo:12:3-14:20:writable] Error: message...
# omc also emits single positions without an end range: [...:12:3:writable]
_LOCATED = re.compile(
    r"^\[(?P<file>[^\]]*?):(?P<l1>\d+):(?P<c1>\d+)"
    r"(?:-(?P<l2>\d+):(?P<c2>\d+))?:[^\]]*\]\s*"
    r"(?P<sev>Error|Warning|Notification):\s*(?P<msg>.*)$",
    re.DOTALL,
)

# Error: message...  (no source location)
_BARE = re.compile(
    r"^(?P<sev>Error|Warning|Notification):\s*(?P<msg>.*)$", re.DOTALL
)

_SEV_MAP = {
    "Error": Severity.ERROR,
    "Warning": Severity.WARNING,
    "Notification": Severity.NOTIFICATION,
}

# Ordered: first match wins.
_KIND_RULES: list[tuple[Kind, re.Pattern]] = [
    (Kind.SYNTAX, re.compile(
        r"Parse error|Parser error|Missing token|syntax error|unexpected token|"
        r"Expected token", re.IGNORECASE)),
    (Kind.LOOKUP, re.compile(
        r"not found in scope|Class \S+ not found|Variable \S+ not found|"
        r"Base class \S+ not found|component .* not found", re.IGNORECASE)),
    (Kind.TYPE, re.compile(
        r"Type mismatch|expected subtype|incompatible types|"
        r"unit .* not compatible|Illegal type", re.IGNORECASE)),
    (Kind.CONNECT, re.compile(
        r"connector|connect\(|not a valid connect|flow variable|stream variable",
        re.IGNORECASE)),
    (Kind.BALANCE, re.compile(
        r"imbalanced|structurally singular|too (?:many|few) equations|"
        r"under-?determined|over-?determined|"
        r"equations? \(\d+\).*variables? \(\d+\)|"
        r"\d+ equations? and \d+ variables?", re.IGNORECASE)),
    (Kind.INITIALIZATION, re.compile(
        r"initial(?:ization| equation| conditions?)|start value|fixed=", re.IGNORECASE)),
    (Kind.RUNTIME, re.compile(
        r"Simulation execution failed|solver|integrator|assert\b|division by zero|"
        r"nonlinear system|chattering|stopped at time|LOG_", re.IGNORECASE)),
]


@dataclass
class Diagnostic:
    severity: Severity
    message: str
    kind: Kind = Kind.OTHER
    file: Optional[str] = None
    line_start: Optional[int] = None
    col_start: Optional[int] = None
    line_end: Optional[int] = None
    col_end: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["kind"] = self.kind.value
        return d

    def brief(self) -> str:
        loc = ""
        if self.file and self.line_start is not None:
            loc = f"{self.file}:{self.line_start}: "
        return f"[{self.severity.value}/{self.kind.value}] {loc}{self.message}"


def classify(message: str) -> Kind:
    for kind, pat in _KIND_RULES:
        if pat.search(message):
            return kind
    return Kind.OTHER


def _split_records(raw: str) -> list[str]:
    """Split a getErrorString() blob into individual diagnostic records.

    omc concatenates records as lines; a record starts with '[' (located) or a
    severity keyword, and continuation lines belong to the previous record.
    """
    records: list[str] = []
    current: list[str] = []
    start = re.compile(r"^(\[|Error:|Warning:|Notification:)")
    for line in raw.splitlines():
        if start.match(line.strip()) and current:
            records.append("\n".join(current).strip())
            current = [line]
        elif start.match(line.strip()):
            current = [line]
        elif current:
            current.append(line)
        elif line.strip():
            # Leading unstructured text (e.g. simulation stdout) — keep as record.
            current = [line]
    if current:
        records.append("\n".join(current).strip())
    return [r for r in records if r.strip().strip('"')]


def parse_error_string(raw: str) -> list[Diagnostic]:
    """Parse the output of omc's getErrorString() into Diagnostics."""
    if not raw:
        return []
    raw = raw.strip().strip('"').strip()
    if not raw:
        return []

    out: list[Diagnostic] = []
    for rec in _split_records(raw):
        m = _LOCATED.match(rec)
        if m:
            msg = m.group("msg").strip()
            out.append(Diagnostic(
                severity=_SEV_MAP[m.group("sev")],
                message=msg,
                kind=classify(msg),
                file=m.group("file") or None,
                line_start=int(m.group("l1")),
                col_start=int(m.group("c1")),
                line_end=int(m.group("l2")) if m.group("l2") else None,
                col_end=int(m.group("c2")) if m.group("c2") else None,
            ))
            continue
        m = _BARE.match(rec)
        if m:
            msg = m.group("msg").strip()
            out.append(Diagnostic(
                severity=_SEV_MAP[m.group("sev")],
                message=msg,
                kind=classify(msg),
            ))
            continue
        # Unstructured content (runtime stdout etc.)
        out.append(Diagnostic(
            severity=Severity.ERROR if "fail" in rec.lower() else Severity.NOTIFICATION,
            message=rec.strip(),
            kind=classify(rec),
        ))
    return out


def parse_simulation_messages(messages: str) -> list[Diagnostic]:
    """Parse the 'messages' field of an omc simulate() result record.

    Runtime logs look like:  LOG_STDOUT | error | msg   or plain text lines.
    """
    if not messages:
        return []
    out: list[Diagnostic] = []
    log_line = re.compile(
        r"^\s*(?P<stream>LOG_\w+)\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<msg>.*)$")
    for line in messages.splitlines():
        if not line.strip():
            continue
        m = log_line.match(line)
        if m:
            level = m.group("level").lower()
            sev = (Severity.ERROR if level in ("error", "assert")
                   else Severity.WARNING if level == "warning"
                   else Severity.NOTIFICATION)
            msg = m.group("msg").strip()
            out.append(Diagnostic(severity=sev, message=msg,
                                  kind=classify(msg) if classify(msg) != Kind.OTHER
                                  else Kind.RUNTIME))
        else:
            sev = (Severity.ERROR if re.search(r"fail|error", line, re.IGNORECASE)
                   else Severity.NOTIFICATION)
            out.append(Diagnostic(severity=sev, message=line.strip(),
                                  kind=classify(line)))
    return out


def summarize_for_llm(diags: list[Diagnostic], limit: int = 20) -> str:
    """Compact, deduplicated plain-text summary suitable for an LLM fix prompt."""
    errors = [d for d in diags if d.severity == Severity.ERROR]
    warnings = [d for d in diags if d.severity == Severity.WARNING]
    seen: set[str] = set()
    lines: list[str] = []
    for d in errors + warnings:
        b = d.brief()
        if b in seen:
            continue
        seen.add(b)
        if len(lines) >= limit:
            break
        lines.append(b)
    suppressed = len(errors) + len(warnings) - len(lines)
    if suppressed > 0:
        lines.append(f"... ({suppressed} more suppressed)")
    return "\n".join(lines) if lines else "No errors or warnings."


# --------------------------------------------------------- quality gates --
# Warning-level quality gates: diagnostics omc emits as mere warnings but
# which indicate real model-quality problems. Matched warnings are surfaced
# as verifier-style complaints and fed back into the fix loop instead of
# being ignored for success. Opt-in at the AgentLoop level so benchmark
# scores stay comparable across versions.

WARNING_GATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("initial conditions not fully specified", re.compile(
        r"initial conditions .*not (?:fully )?specified", re.IGNORECASE)),
    ("over- or inconsistently specified initial conditions", re.compile(
        r"initial conditions .*(?:over-?specified|conflicting|inconsistent)"
        r"|conflicting initial conditions", re.IGNORECASE)),
    ("over/under-determined system", re.compile(
        r"model (?:is )?(?:over-|under-)determined"
        r"|(?:over-|under-)determined system", re.IGNORECASE)),
    ("inconsistent units", re.compile(
        r"units? (?:mismatch|inconsisten|are not equivalent)"
        r"|unit (?:mismatch|inconsisten[ct])"
        r"|units? .*(?:mismatch|inconsisten|equivalen)", re.IGNORECASE)),
]

DEFAULT_WARNING_GATE = WARNING_GATE_PATTERNS


def warning_gate_complaints(
        diags: list[Diagnostic],
        rules: list[tuple[str, re.Pattern]] = DEFAULT_WARNING_GATE,
) -> Optional[str]:
    """Return a label+message complaint for every gated warning, or None.

    Callables with this shape satisfy the ``omagent.loop.WarningGate``
    protocol: given the diagnostics of an attempt, return None when the
    model passes the gate, else a human/LLM-readable complaint.
    """
    out: list[str] = []
    for d in diags:
        if d.severity != Severity.WARNING:
            continue
        for label, pat in rules:
            if pat.search(d.message):
                out.append(f"quality: {label}: {d.message.strip()}")
                break
    return "; ".join(out) if out else None


# Newer OMPython raises OMCSessionException whose message embeds omc's log as
# "[OMC log for 'sendExpression(...)']: [kind:level:id] message"
_OMPY_EXC = re.compile(
    r"\[(?P<kind>\w+):(?P<level>error|warning|notification):(?P<id>-?\d+)\]\s*"
    r"(?P<msg>.*)", re.DOTALL)

_OMPY_KIND_MAP = {
    "syntax": Kind.SYNTAX,
    "grammar": Kind.SYNTAX,
    "simulation": Kind.RUNTIME,
}


def parse_ompython_exception(text: str) -> list[Diagnostic]:
    """Parse an OMPython OMCSessionException message into Diagnostics.

    Falls back to a single generic error diagnostic for unstructured messages
    (e.g. ZeroMQ connection failures), so backend exceptions never vanish.
    """
    m = _OMPY_EXC.search(text or "")
    if not m:
        return [Diagnostic(Severity.ERROR, (text or "unknown backend failure").strip(),
                           Kind.OTHER)]
    level = m.group("level")
    sev = (Severity.ERROR if level == "error"
           else Severity.WARNING if level == "warning"
           else Severity.NOTIFICATION)
    msg = m.group("msg").strip()
    # message-based classification wins when specific; else fall back to
    # omc's own kind tag (syntax/grammar/simulation/...)
    kind = classify(msg)
    if kind == Kind.OTHER:
        kind = _OMPY_KIND_MAP.get(m.group("kind").lower(), Kind.OTHER)
    return [Diagnostic(sev, msg, kind)]
