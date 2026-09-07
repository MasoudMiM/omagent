# omagent

**LLM-assisted modeling for OpenModelica** — an open-source, headless Python
agent that turns a natural-language task into a *verified* Modelica model:

```
natural language ──> generate ──> compile (omc) ──> simulate ──> verify physics
                        ▲                                            │
                        └──── structured error / verifier feedback ──┘
```

Unlike a plain code assistant, omagent closes the loop on **physics, not just
compilation**: quantitative verifiers check trajectories against expected
behavior (final values, settling windows, bounds, overshoot), and every
failure — compiler diagnostics or physics complaints — is parsed into
structured feedback for the next fix attempt.

Key capabilities:

- **Structured omc diagnostics** — parses `getErrorString()`, simulation
  logs, and OMPython's exception format into records with severity, source
  location, and failure kind (syntax / lookup / type / balance / connect /
  initialization / runtime)
- **Environment-grounded fix hints** — on `Class X not found`, omagent asks
  omc what the parent package *actually* contains (`getClassNames`) and puts
  near-miss suggestions into the fix prompt. This resolves the dominant
  observed failure mode: stale library knowledge (e.g. MSL 3.2 names such as
  `Basic.EMF` vs. MSL 4.x `Basic.RotationalEMF`)
- **Quantitative verification** — reads CSV or Dymola-format `.mat` results;
  verifier complaints ("final value of x is 1.93, expected 2.0") drive tuning
- **Benchmark task ladder** — 5 escalating, auto-gradable tasks with full
  transcript capture, in the format the OpenModelica benchmark discussion
  ([OpenModelica#15385](https://github.com/OpenModelica/OpenModelica/issues/15385))
  calls for
- **Multi-run benchmarking** — repeat runs for variance measurement and
  side-by-side cross-model comparison, aggregated into a single report
- **LLM-backend-agnostic** — the loop depends on a one-method protocol;
  adapters ship for Anthropic and for any OpenAI-compatible endpoint
  (Ollama, LM Studio, llama.cpp, vLLM — local open-weight models included)
- **Tested** — 101 unit tests run without OpenModelica installed; 4
  integration tests validate against a live omc

## Installation

Requires Python >= 3.10. The core package has zero hard dependencies;
features are opt-in extras:

```bash
pip install -e .                    # parsers + loop only (no omc needed)
pip install -e ".[omc]"             # + OMPython (talk to a real omc)
pip install -e ".[results]"         # + scipy (.mat result files; CSV needs nothing)
pip install -e ".[llm]"             # + anthropic adapter
pip install -e ".[all]"             # everything, including pytest
```

To use it against a real compiler you need
[OpenModelica](https://openmodelica.org) (tested with 1.26–1.27) with the
Modelica Standard Library installed for `omc`:

```bash
echo 'installPackage(Modelica); getErrorString();' > /tmp/i.mos && omc /tmp/i.mos
```

> Note: OMEdit installs the MSL for itself automatically; headless `omc`
> sessions do not. If models using `Modelica.*` fail with "Class ... not
> found", this is why.

Verify your setup:

```bash
pytest -m "not integration"   # unit tests, no omc required
pytest -m integration         # against your live omc (+MSL, scipy for .mat)
```

## Quick start

```python
from omagent import AgentLoop, OMSession, all_of, expect_bounds, expect_final
from omagent.llm import ClaudeLLM   # or any object with .propose(...)

# physics acceptance criteria — complaints feed back into the fix loop
verifier = all_of(
    expect_bounds("x", lo=-0.105, hi=0.105),
    expect_final("x", 0.0, atol=0.06, rtol=0.0),
)

loop = AgentLoop(
    OMSession(),                      # real omc via OMPython
    ClaudeLLM(),                      # needs ANTHROPIC_API_KEY
    max_attempts=4,
    simulate_options={"stopTime": 10.0, "outputFormat": "csv"},
    verifier=verifier,
)
result = loop.run(
    "A mass-spring-damper: m = 1 kg, c = 100 N/m, d = 1 N.s/m, released "
    "from x = 0.1 m at rest. Name position x and velocity v.")

print(result.success, result.model_name)
print(result.final_code)
for a in result.attempts:
    print(a.n, a.stage, a.complaint)
```

**Local / open-weight models** work through any OpenAI-compatible server
(Ollama, LM Studio, llama.cpp server, vLLM) with no extra dependencies:

```python
from omagent.llm import OpenAICompatLLM
llm = OpenAICompatLLM(model="qwen2.5-coder:14b")            # Ollama default URL
# llm = OpenAICompatLLM(model="...", base_url="http://localhost:1234/v1")  # LM Studio
```

Or bring your own LLM by implementing one method:

```python
class MyLLM:
    def propose(self, task, previous_code, error_summary):
        # previous_code/error_summary are None on the first (fresh) call;
        # on retries they contain the failed model and structured feedback.
        return "... complete Modelica model ..."
```

### Run the benchmark ladder

```bash
export ANTHROPIC_API_KEY=...
python examples/run_ladder.py                 # all 5 tiers
python examples/run_ladder.py --max-tier 3    # subset by difficulty
python examples/run_ladder.py --tasks dc_motor --model claude-opus-4-8
```

Tiers: (1) pure-equation dynamics, (2) MSL component composition, (3) hybrid
events, (4) verifier-driven design — the requirement is given, the parameter
is not, (5) multi-domain electro-mechanical. Per-task JSON transcripts
(attempt history, diagnostics, code, LLM rounds) land in `transcripts/`,
with `summary.json` aggregating results.

### Compare models with repeated runs

```python
from omagent import OMSession, run_comparison
from omagent.llm import ClaudeLLM, OpenAICompatLLM

comparison = run_comparison(
    session_factory=OMSession,          # fresh omc session per run
    llm_factories={
        "claude-sonnet": lambda: ClaudeLLM(model="claude-sonnet-4-6"),
        "local-qwen":    lambda: OpenAICompatLLM(model="qwen2.5-coder:14b"),
    },
    repeats=3,                          # runs per model for variance
)
```

Each (model, task, run) triple runs in isolation, so variance across repeats
reflects LLM/omc nondeterminism rather than state contamination. Transcripts
land in `transcripts/<model>/rep<k>/`; `comparison.json` aggregates pass
rates per model and per task, plus mean/spread of attempts and wall time.
Use `--tasks`/`--max-tier` equivalents via `task_ids`/`max_tier`, and
`verbose=True` for progress and a final table.

### Warning-level quality gates

Some omc diagnostics come as warnings yet mean the model is sloppy —
under/over-specified initial conditions, inconsistent units, over-determined
systems. Quality gates turn those into verifier-style complaints that feed
the fix loop, without outright failing the operation:

```python
from omagent import AgentLoop, OMSession, warning_gate_complaints

loop = AgentLoop(
    OMSession(), ClaudeLLM(), max_attempts=4,
    verifier=my_verifier,
    warning_gate=warning_gate_complaints,   # opt-in; None by default
)
```

Gated attempts report stage `"quality"` and the gate complaint is appended
to the fix prompt's structured feedback. `run_ladder(..., warning_gate=...)`
threads the gate through the benchmark so scores can be produced under
either strictness. Custom gates are just callables over
`list[Diagnostic] -> Optional[str]`; `WARNING_GATE_PATTERNS` is the default
rule table you can extend.

### Use pieces standalone

```python
from omagent import OMSession, parse_error_string, summarize_for_llm, load_result

s = OMSession()
r = s.load_string(my_modelica_code)     # honest success verdict + diagnostics
print(summarize_for_llm(r.diagnostics)) # deduplicated digest for any prompt

sim = s.simulate("MyModel", stopTime=5.0, outputFormat="csv")
res = load_result(sim.value["resultFile"])
times, x = res.series("x")
```

## Project layout

```
omagent/
  errors.py    # omc diagnostic parsing + classification
  session.py   # OMSession: testable wrapper over OMPython/omc
  loop.py      # AgentLoop + lookup-suggestion feedback
  results.py   # CSV/.mat readers + quantitative verifiers
  llm.py       # Anthropic adapter (protocol: bring your own)
  tasks.py     # benchmark task ladder definitions
  runner.py    # ladder execution + transcript persistence
examples/      # first_run.py, run_ladder.py
tests/         # 101 unit + 4 integration tests
```

## Design notes

- **Testable by construction.** `OMSession` talks to any object with
  `sendExpression()`; tests replay recorded omc output, so the full agent
  loop is unit-tested without a compiler or an API key.
- **Both OMPython contracts.** Older OMPython returns and lets you read
  `getErrorString()`; newer OMPython raises `OMCSessionException` on
  error-level messages. Both yield identical structured failures.
- **Environment failures are not model failures.** The ladder runner loads
  the MSL when a task requires it and reports load problems as
  `environment` outcomes with zero attempts charged to the LLM.

## Roadmap

- Optional MCP tool surface, composing with OMEdit's built-in MCP server
- More ladder tiers targeting thermal/fluid domains and third-party libraries

## License

BSD-3-Clause — see [LICENSE](LICENSE).
