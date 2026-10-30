
# Gemmini RTL Formal Verification Quickstart

This backend uses an agent to generate SystemVerilog assertions and cover properties for fixed Gemmini RTL. The generated properties are checked using trusted formal verification tools.

The agent cannot modify the RTL or introduce new assumptions by default. A successful proof confirms that the RTL satisfies the generated property, but does not guarantee that the property fully captures the intended requirement.

Run all commands from the repository root.

## 1. Set up formal verification tools

Install the project and check the available tools:

```bash
python -m pip install -e ".[test]"
python scripts/check_formal_environment.py
```

Formal verification requires `yosys`, `yosys-smtbmc`, and `z3`. The optional SymbiYosys flow also requires `sby`.

Build the CHIA formal verification worker:

```bash
docker build -t chia-ctv-formal:latest -f docker/formal/Dockerfile .
```

## 2. Run the local verification example

The included example runs without Gemmini RTL or a language model:

```bash
chia-trusted-verify evaluate rtl_formal/examples/known_good_collateral.sv \
  --config configs/rtl_formal_example.yaml
```

A successful run reports `status: pass` with successful compilation, bounded checking, induction, and cover verification.

The backend also rejects unauthorized `assume` statements.

## 3. Configure Gemmini RTL

The 24 verification intents are defined in `rtl_formal/gemmini/intents.yaml`.

Obtain the Gemmini/Chipyard source and generate the RTL for the selected design. Record the source revisions used for the experiment.

Configure the generated RTL:

```bash
python scripts/configure_gemmini_generated_rtl.py \
  /path/to/chipyard/sims/verilator/generated-src/YOUR_GENERATED_DESIGN
```

This creates `configs/rtl_formal_gemmini_local.yaml` and five additional configuration files containing the RTL sources for each target module.

These files are excluded from Git. The script also checks module names and reports missing sources. Update the intent definitions if your Gemmini revision uses different module names.

## 4. Run a single verification intent

Preview the generated formal verification program:

```bash
chia-trusted-verify render-program \
  --config configs/rtl_formal_gemmini_local.yaml
```

Check the experiment plan for one intent:

```bash
python scripts/run_rtl_formal_experiments.py \
  --config configs/rtl_formal_gemmini_local.yaml \
  --intents Q1_no_issue_with_dependencies \
  --seeds 200 \
  --dry-run
```

To run agent-based verification, complete the CHIA and Google Cloud setup in the [TeAAL-Evolve Quickstart](TEAAL_EVOLVE_QUICKSTART.md) and build the formal worker.

Start the cluster and run the experiment:

```bash
chia up cluster/local_gemini_formal.yaml

export RAY_ADDRESS="$CHIA_HEAD_IP:6379"

python examples/run_chia_loop.py \
  --config configs/rtl_formal_gemmini_local.yaml \
  --engine llm_agent

chia down cluster/local_gemini_formal.yaml
```

## 5. Run the full evaluation

The evaluation includes 24 verification intents covering queue and dependency logic, control logic, memory and DMA, and interface protocols.

See the [Gemmini Verification Intents](../rtl_formal/gemmini/README.md) for details.

After validating the setup with one intent, run the full evaluation using three seeds and up to four proposals per run:

```bash
python scripts/run_rtl_formal_experiments.py \
  --config configs/rtl_formal_gemmini_local.yaml \
  --intents all \
  --seeds 200,201,202 \
  --max-attempts 4
```

The paper evaluates **72 runs and 288 proposals**. With iterative repair, compilation success increases from **43.1% to 91.7%**, and decisive formal outcomes (proofs or counterexamples) increase from **34.7% to 81.9%**.

The included [Gemmini Results](../report_data/backend2/backend2_final_analysis.txt) can be inspected without rerunning the experiments.

To regenerate the analysis from the original logs in `runs/rtl_formal_gemmini/`:

```bash
python scripts/analyze_backend2_final.py
```
