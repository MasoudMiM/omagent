"""omagent — testable tooling layer for LLM-assisted OpenModelica workflows."""

from .errors import (
    Diagnostic, Kind, Severity,
    classify, parse_error_string, parse_ompython_exception, parse_simulation_messages,
    summarize_for_llm, warning_gate_complaints, WARNING_GATE_PATTERNS,
)
from .session import Backend, OMSession, OpResult
from .loop import (AgentLoop, Attempt, LLM, LoopResult, Verifier, WarningGate,
                   extract_code, extract_model_name)
from .runner import run_ladder, run_comparison
from .results import (SimulationResult, all_of, expect_bounds, expect_final,
                      expect_value_at, load_result)

__version__ = "0.3.0"
__all__ = [
    "Diagnostic", "Kind", "Severity", "classify", "parse_error_string",
    "parse_ompython_exception", "parse_simulation_messages", "summarize_for_llm",
    "warning_gate_complaints", "WARNING_GATE_PATTERNS",
    "Backend", "OMSession", "OpResult",
    "AgentLoop", "Attempt", "LLM", "LoopResult", "Verifier", "WarningGate",
    "extract_code", "extract_model_name", "run_ladder", "run_comparison",
    "SimulationResult", "load_result", "expect_final", "expect_value_at",
    "expect_bounds", "all_of",
]
