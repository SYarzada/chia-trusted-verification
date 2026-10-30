# Paper Results

This folder contains the results reported in the paper. The files can be inspected without rerunning the experiments.

- **`backend1/` — TeAAL-Evolve:** CSV tables containing search results, certification outcomes, verification reuse, runtime, and token and cost measurements. The evaluation includes **560 runs**, **16,779 proposals**, and **16,629 candidates** used for certification replay.

- **`backend2/` — Gemmini RTL Formal Verification:** Results from **24 verification intents**, each evaluated with three seeds and four proposals per run (**72 runs and 288 proposals**). Compilation success improves from **43.1% to 91.7%**, while decisive formal outcomes (proofs or counterexamples) increase from **34.7% to 81.9%**.

- **`certificate_validation/` — Certificate Validation:** Results from independently checking the **18 saved TeAAL verification certificates** in `artifacts/`.

## Validate saved certificates

Run the following commands from the repository root:

```bash
python -m pip install -e ".[test]"
python scripts/validate_teaal_certificates.py artifacts
```

The original experiment logs are archived separately and are not included in this repository.

See [Reproducing Results](../docs/REPRODUCE_RESULTS.md) for instructions on running the analyses and identifying which steps require the original logs.
