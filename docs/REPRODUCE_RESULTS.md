# Inspecting and Reproducing Results

All commands should be run from the repository root. The [Paper Results](../report_data/README.md) folder contains processed results from 560 TeAAL-Evolve runs (16,779 proposals) and 72 Gemmini RTL formal verification runs (288 proposals). The repository also includes 18 independently checkable TeAAL certificates. Full experiment logs are archived separately.

## 1. Validate included results

Install the required packages and run the tests:

```bash
python -m pip install -e ".[test,analysis]"
python -m pytest -q
python scripts/validate_teaal_certificates.py artifacts
```

The certificate validator checks all 18 saved TeAAL verification records.

The included results are organized as follows:

- `report_data/backend1/` contains TeAAL search and certification results.
- `report_data/backend2/backend2_final_analysis.txt` contains Gemmini formal verification results.
- `report_data/certificate_validation/validation_summary.txt` contains certificate validation results.

## 2. Run new TeAAL experiments

The four `configs/teaal_selected_seeds_*.yaml` files define the models, search methods, starting designs, and proposal budgets used in the paper.

Preview the Gemini 3.5 Flash experiment plan without running any models:

```bash
python experiments/run_teaal_experiments.py \
  --config configs/teaal_selected_seeds_gemini35flash.yaml \
  --dry-run > /tmp/teaal-plan.txt

head -n 1 /tmp/teaal-plan.txt
```

Follow the [TeAAL-Evolve Quickstart](TEAAL_EVOLVE_QUICKSTART.md) to set up CHIA and run experiments.

Each completed search run saves evaluation results and a certification replay. Initial baseline evaluations are not counted as search proposals.

### Distributed experiments (optional)

To run experiments across multiple Google Cloud machines, configure `PROJECT`, `ZONE`, and `NUM_VMS` for your account. See `scripts/experiment.env.example` for additional settings, including managed AlphaEvolve access.

After setting up Google Cloud authentication:

```bash
bash scripts/provision_gcp_experiments.sh

export CONFIG=configs/teaal_selected_seeds_gemini35flash.yaml
bash scripts/launch_gcp_search_all.sh
PHASE=search bash scripts/monitor_gcp_experiments.sh

export REMOTE_RESULTS=runs/teaal_chia_challenge_gemini35flash
bash scripts/launch_gcp_replay_all.sh
PHASE=replay bash scripts/monitor_gcp_experiments.sh

bash scripts/collect_gcp_results.sh
```

These experiments require paid cloud resources and access to the selected models.

## 3. Reproduce the paper's results

The final TeAAL analysis scripts require the original experiment logs, which are archived separately.

Run the search analysis first, followed by the certification analysis:

```bash
python analysis/final_teaal_search_analysis.py \
  --input-root /path/to/raw/runs \
  --output results/final_teaal_analysis

python analysis/final_teaal_certification_analysis.py \
  --input-root /path/to/raw/runs \
  --output results/final_teaal_analysis
```

The input directory must contain all four experiment folders covering the 560 runs. If the folders are stored separately, use `--flash-root`, `--pro-alpha-root`, `--pro-openevolve-root`, and `--pro-llm-root` to specify their locations.

Use `--help` to view additional options.

Compare the generated CSV files with the included results in `report_data/backend1/`.

To analyze Gemmini RTL formal verification results, use the original logs under `runs/rtl_formal_gemmini/`:

```bash
python scripts/analyze_backend2_final.py
```

## 4. Summarize new TeAAL runs

To generate summaries from your own completed experiments:

```bash
python analysis/summarize_teaal_results.py \
  --runs runs \
  --output results/teaal_summary
```

See [Results](RESULTS.md) for descriptions of the output files and fields.
