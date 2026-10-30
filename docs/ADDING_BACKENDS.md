# Adding a Trusted Verification Backend

The trusted verification layer supports different backends through a common interface. Each backend defines the artifacts it accepts, how they are verified, and what feedback is returned to the agent.

The existing [Gemmini RTL formal verification backend](../src/chia_trusted_verification/backends/rtl_formal/) provides an example.

## 1. Implement the backend

Create a new package under:

```text
src/chia_trusted_verification/backends/
  custom_backend/
    __init__.py
    backend.py
    worker.py
```

Implement `TrustedVerificationBackend` with the following methods:

| Method | Purpose |
| --- | --- |
| `build_search_problem()` | Defines the artifact format and search guidance |
| `prepare()` | Converts and validates the generated artifact |
| `verify()` | Runs trusted verification tools |
| `evaluate()` | Calculates results and metrics |
| `feedback()` | Returns verification results and diagnostics to the agent |

For example:

```python
from chia_trusted_verification import (
    SearchGuidance,
    SearchProblem,
    TrustedVerificationBackend,
)


class CustomBackend(TrustedVerificationBackend):
    name = "custom_backend"

    def build_search_problem(self, settings):
        return SearchProblem(
            initial_source=...,
            file_name="candidate.sv",
            guidance=SearchGuidance(
                system_message="Generate or repair verification properties.",
                problem_description="Verify the configured RTL requirements.",
            ),
        )

    def prepare(self, candidate_source, context, intent, settings):
        ...

    def verify(self, prepared, intent, settings):
        ...

    def evaluate(self, verified, intent, settings):
        ...

    def feedback(self, evaluated, intent, settings):
        ...
```

This is an example interface. Each method must be implemented for the new backend.

The built-in Gemmini backend uses `prepare()` to check generated formal properties, `verify()` to run Yosys and Z3, `evaluate()` to process formal outcomes, and `feedback()` to return diagnostics for repair.

## 2. Define search guidance

Each backend defines its own artifact format and generation instructions through `SearchProblem` and `SearchGuidance`.

Search strategies use this guidance without needing backend-specific changes. Supported strategies include OpenEvolve, AlphaEvolve, and the CHIA LLM Agent.

Keep backend-specific instructions in the backend rather than in the shared search configurations.

## 3. Register the backend

For a backend included in this repository, add it to `_BUILTINS` in:

```text
src/chia_trusted_verification/core/registry.py
```

For a backend provided by an external Python package, register an entry point in its `pyproject.toml`:

```toml
[project.entry-points."chia_trusted_verification.backends"]
custom_backend = "my_package.backend:CustomBackend"
```

The registry loads the selected backend at runtime.

## 4. Configure verification

Create a YAML configuration specifying the backend, its resources, and the verification intent.

For example:

```yaml
backend:
  name: custom_backend
  settings:
    rtl_root: /path/to/rtl
    solver: symbiyosys
  resources:
    prepare: {}
    verify:
      formal: 1
    evaluate: {}
    feedback: {}

verification:
  intent:
    intent_id: queue_control
    artifact_type: systemverilog_formal_collateral
    requirements:
      - assertions_compile
      - prove_selected_intents
      - reach_selected_covers
    parameters:
      depth: 20
```

These settings are illustrative and must match the new backend's supported options.

Resource requirements must also be configured in the CHIA cluster YAML. The verification intent specifies what the backend should check.

## 5. Run the backend

A registered backend can use any compatible search strategy:

```bash
chia-trusted-verify run --config path/to/backend_config.yaml --engine openevolve
chia-trusted-verify run --config path/to/backend_config.yaml --engine alphaevolve
chia-trusted-verify run --config path/to/backend_config.yaml --engine llm_agent
```

The search method changes, but the backend and verification requirements remain the same.

## 6. Test the backend

Check that the backend:

- Builds the search problem and guidance correctly.
- Processes valid and invalid artifacts.
- Runs all four verification stages.
- Returns useful feedback when verification fails.
- Validates verification intents.
- Registers and runs through the common CHIA interface.

See `tests/test_backend_plugin.py` for an example of integrating another backend without changing the shared verification workflow.
