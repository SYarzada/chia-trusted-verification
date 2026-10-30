
# Gemmini RTL Formal Verification Intents

This folder contains the 24 verification intents used in the Gemmini RTL formal verification evaluation (Backend II).

Each intent defines a requirement to check against fixed Gemmini RTL. The agent generates SystemVerilog assertions and cover properties, which are checked using trusted formal verification tools.

## Verification Intents

The evaluation includes 24 intents across four categories:

| Category | Intents | Target modules |
| --- | --- | --- |
| Queue and dependency | 6 | `ReservationStation` |
| Control | 6 | `LoadController`, `ExecuteController`, `ReservationStation` |
| Memory and DMA | 6 | `ScratchpadBank`, `LoadController`, `ExecuteController`, `StoreController` |
| Interface and protocol | 6 | `LoadController`, `StoreController`, `ReservationStation` |

The complete [intent catalog](intents.yaml) defines each requirement, target module, relevant signals, source references, and verification depth.

Each intent is evaluated with three seeds and up to four proposals per run, totaling **72 runs and 288 proposals**.

## Trusted Verification

The backend verifies agent-generated SystemVerilog assertions and cover properties against fixed Gemmini RTL using Yosys-based formal verification.

The current implementation supports:

- Validation of generated properties before formal checking.
- Rejection of agent-generated assumptions by default.
- Trusted environment assumptions provided through separately reviewed files.
- Formal proofs, counterexamples, cover reachability checks, and tool diagnostics.
- Iterative property repair using trusted verification feedback.

### Limitations and Future Work

A successful proof confirms that the RTL satisfies the generated property under the trusted assumptions. However, it does not guarantee that the property correctly or completely captures the intended requirement.

Future extensions can strengthen property validation through:

- **Specification alignment:** Compare generated properties against trusted reference specifications using equivalence, implication, or refinement checks.
- **Vacuity analysis:** Detect properties that pass without meaningfully checking the intended behavior.
- **Mutation-based testing:** Introduce controlled RTL faults to evaluate whether generated properties detect meaningful violations.

These checks can be integrated into the existing trusted verification backend without changing the shared CHIA interface.

## Gemmini Source

The verification intents reference the original [Gemmini RTL and Chisel implementation](https://github.com/ucb-bar/gemmini), including its reservation station, load and store controllers, execute controller, and scratchpad memory.

See the [Gemmini RTL Formal Verification Quickstart](../../docs/RTL_FORMAL_QUICKSTART.md) for setup and experiment instructions.
