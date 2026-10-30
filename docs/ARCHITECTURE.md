
# CHIA Trusted Verification Architecture

The framework separates agent-based search from trusted verification. A common interface supports different search methods and verification backends.

```text
                  Search Methods
                 /              \
       OpenEvolve / AlphaEvolve   CHIA LLM Agent
                 \              /
                  \            /
             Trusted Verification Layer
       prepare → verify → evaluate → feedback
                          |
                   Backend Registry
                    /           \
             TeAAL-Evolve     RTL Formal
```

## 1. Search methods

The framework supports three agentic search methods:

- **OpenEvolve and AlphaEvolve:** Use `EvolutionSearchStrategy` through the evolve-flows/SkyDiscover adapter.
- **CHIA LLM Agent:** Uses `LLMAgentSearchStrategy` and a trusted verification tool to evaluate generated candidates.

All three methods use the same trusted verification interface.

## 2. Trusted verification interface

Each candidate passes through four stages:

```text
prepare_artifact → verify_artifact → evaluate_artifact → feedback_for_search
```

Each stage runs as a CHIA `@ChiaFunction`.

| Stage | Purpose |
| --- | --- |
| `prepare` | Converts and validates the generated artifact |
| `verify` | Runs the selected trusted verification backend |
| `evaluate` | Calculates results and performance metrics |
| `feedback` | Returns verification evidence and diagnostics to the search method |

For OpenEvolve and AlphaEvolve, the stages run in sequence through `TrustedVerificationEvaluator`. CHIA passes each stage's output to the next and retrieves the final result.

For the CHIA LLM Agent, `TrustedVerificationTool` exposes a single `trusted_verification_evaluate_candidate` operation. The tool runs all four stages and returns the results to the agent.

## 3. Search problem and guidance

Each backend defines a `SearchProblem` containing:

```text
initial_source
file_name
metadata
SearchGuidance
```

`SearchGuidance` provides instructions for generating and modifying artifacts.

For example, TeAAL-Evolve provides instructions for accelerator design changes, while the RTL formal backend provides instructions for generating SystemVerilog verification properties.

Search methods use this guidance without requiring changes to their implementations.

## 4. Verification intent

A `VerificationIntent` specifies what should be checked for each generated artifact.

- **TeAAL-Evolve:** Checks structured accelerator design legality and certification requirements.
- **RTL Formal:** Checks generated assertions and cover properties using trusted formal verification tools.

The selected backend determines how to perform the checks and what evidence to return.

## 5. Verification backends

Each backend implements `TrustedVerificationBackend`:

```python
build_search_problem(settings) -> SearchProblem
prepare(candidate_source, context, intent, settings) -> dict
verify(prepared, intent, settings) -> dict
evaluate(verified, intent, settings) -> dict
feedback(evaluated, intent, settings) -> dict
```

The common result includes:

```text
status
metrics
feedback
evidence
trace
diagnostics
```

The backend registry selects the implementation based on the experiment configuration. New backends can be added without changing the shared search or verification workflow.

See [Adding a Trusted Verification Backend](ADDING_BACKEND.md) for implementation details.

## 6. CHIA workers

CHIA uses logical workers to assign tasks to available resources.

The provided cluster configurations define resources for trusted verification, agent execution, and Google Cloud authentication.

- `cluster/local_gemini.yaml` supports TeAAL-Evolve experiments.
- `cluster/local_gemini_formal.yaml` also provides a `formal` worker for RTL verification.

Worker resources can be changed through the cluster and backend configurations without modifying the verification interface.
