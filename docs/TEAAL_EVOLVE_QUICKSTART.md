# TeAAL-Evolve Quickstart

Run the following commands from the repository root on a Linux machine. The setup requires Docker, SSH access to the CHIA head machine, Conda, Git, Google Cloud CLI, and access to a Google Cloud project with the selected model enabled. CHIA workers run in Docker, while the experiment driver runs in a Conda environment.

## 1. Set up the two repositories

Keep CHIA and this project side by side. Start inside the extracted `chia-trusted-verification` folder:

~~~bash
export CTV_REPO="$PWD"
export CHIA_REPO="$(dirname "$CTV_REPO")/chia"
conda create -n chia_env -c conda-forge python=3.10.19 git -y
conda activate chia_env
git clone https://github.com/ucb-bar/chia.git "$CHIA_REPO"
python -m pip install -e "$CHIA_REPO"
python -m pip install -e ".[search,test,analysis]"
python -m pytest -q
~~~

If CHIA is already installed, use the existing installation. Run the project's `pip` and `pytest` commands from `$CTV_REPO`. If `pandas` or `z3-solver` fails to build, install `pandas`, `matplotlib`, and `z3-solver` using conda-forge before installing the project.

## 2. Authenticate and choose the head address

The included cluster stages your Google Application Default Credentials into its Docker workers. Set your project and perform a headless ADC login if you have not already authenticated:

~~~bash
export GOOGLE_CLOUD_PROJECT="YOUR_GOOGLE_CLOUD_PROJECT"
export GOOGLE_CLOUD_LOCATION=global
export VERTEX_LOCATION=global
gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/cloud-platform
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
gcloud services enable aiplatform.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
gcloud auth application-default print-access-token >/dev/null
~~~

Use the address of the machine running CHIA, reachable by SSH from your shell. Set it to a hostname or address for *your* machine:

~~~bash
export CHIA_HEAD_IP="YOUR_REACHABLE_HEAD_HOST"
export CHIA_PROJECT_ROOT="$CTV_REPO"
ssh -o BatchMode=yes "$USER@$CHIA_HEAD_IP" true
~~~

The SSH check should finish without a password prompt. Keep your local ADC file out of the repository. When opening a new terminal, reactivate the Conda environment and set the environment variables again.

For managed AlphaEvolve, your project must also have its service set up. Export the actual project and engine identifiers:

~~~bash
export ALPHAEVOLVE_PROJECT_ID="$GOOGLE_CLOUD_PROJECT"
export ALPHAEVOLVE_ENGINE_ID="YOUR_ALPHAEVOLVE_ENGINE_ID"
export ALPHAEVOLVE_LOCATION=global
export ALPHAEVOLVE_COLLECTION=default_collection
export ALPHAEVOLVE_ASSISTANT=default_assistant
gcloud services enable discoveryengine.googleapis.com --project "$ALPHAEVOLVE_PROJECT_ID"
~~~

[Google authentication and the three model paths](GOOGLE_GEMINI.md) has more detail.

## 3. Start CHIA and check its workers

~~~bash
cd "$CTV_REPO"
chia up cluster/local_gemini.yaml
export RAY_ADDRESS="$CHIA_HEAD_IP:6379"
python examples/check_chia_workers.py
~~~

The TeAAL worker runs the four trusted stages. The CHIA LLM agent calls them through a `ChiaTool`. OpenEvolve and AlphaEvolve send their proposals through the same trusted evaluator. For the eight kernel examples and their design options, see [TeAAL-Evolve kernels and design options](TEAAL_EVOLVE_BACKEND.md).

## 4. Run one design search

The one-run config `configs/experiment.yaml` uses attention on the included Trainium 1 architecture and writes to `runs/attention/`. Choose a search method:

~~~bash
python examples/run_chia_loop.py --config configs/experiment.yaml --engine llm_agent
~~~

The other supported `--engine` values are `openevolve` and `alphaevolve`. They use the credentials configured above. The initial program and a bounded list of allowed design changes are generated from the kernel and architecture files. The search method proposes changes, and the trusted backend checks each candidate.

A completed run writes `run_manifest.json`, `trusted_results.jsonl`, and `search_result.json` to its run directory. For TeAAL's objective, a smaller `objective_value` is better. Only legal candidates are compared. Results can be summarized after the run using the command in Section 6.


## 5. Inspect or run the final experiments

The TeAAL-Evolve evaluation uses eight starting designs, two Gemini models, and six agentic search methods. Gemini 3.5 Flash also includes two non-LLM baselines. The four experiment configurations are listed in the [main README](../README.md).

To preview the experiment plan without running any models:

```bash
python experiments/run_teaal_experiments.py \
  --config configs/teaal_selected_seeds_gemini35flash.yaml \
  --dry-run > /tmp/teaal-plan.txt

head -n 1 /tmp/teaal-plan.txt
```

The Gemini 3.5 Flash configuration contains 320 runs. Each of the three Gemini 3.1 Pro configurations contains 80 runs.

To regenerate the starting designs and configuration:

```bash
python analysis/select_seed_designs.py --fast
```

Use the provided versions to reproduce the paper's results.

To run a subset of the experiments on your CHIA setup:

```bash
python experiments/run_teaal_experiments.py \
  --config configs/teaal_selected_seeds_gemini35flash.yaml \
  --engine openevolve_structured \
  --kernel attention \
  --resume
```

After each search run, the driver replays the candidates using all four certification methods: Full, Hierarchical, Delta-Syntactic, and Delta-Rewrite. Use `--skip-certification-replay` to skip this additional analysis without affecting the search.

For experiments across multiple Google Cloud machines, see [Reproducing Results](REPRODUCE_RESULTS.md). These experiments require your own cloud resources and access credentials.


## 6. Review results

To summarize completed experiments and shut down the CHIA cluster:

```bash
python analysis/summarize_teaal_results.py --runs runs --output results/teaal_summary
chia down cluster/local_gemini.yaml
```

The summary command requires completed experiment runs. Results from the paper are available in `report_data/`.

See [Results](RESULTS.md) for an explanation of the output fields and [Paper Results](../report_data/README.md) for the published data.