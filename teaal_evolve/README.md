# TeAAL-Evolve Backend

This folder contains the TeAAL modeling tools, accelerator design configurations, verification checks, and optimization objective used by TeAAL-Evolve.

The included inputs are:

- `example/arch/trainium1.yaml` defines the target Trainium-1 architecture.
- `example/kernel/nki/` contains the eight accelerator kernels and selected starting designs used in the paper.

The backend implementation is located in `src/chia_trusted_verification/backends/teaal_evolve/`.

TeAAL-Evolve performs accelerator design space exploration through structured changes to design configurations. Each candidate is checked for legality and correctness. Valid candidates are evaluated using the optimization objective, and verification certificates record the checking results.

For setup and implementation details, see the [TeAAL-Evolve Quickstart](../docs/TEAAL_EVOLVE_QUICKSTART.md), [Backend Reference](../docs/TEAAL_EVOLVE_BACKEND.md), and [Search Methods](../docs/SEARCH_STRATEGIES.md).
