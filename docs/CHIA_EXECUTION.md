# CHIA Execution Model

CHIA manages search execution and trusted verification through two search approaches.

## 1. Search and verification

**OpenEvolve and AlphaEvolve** use `EvolutionSearchStrategy` with the evolve-flows/SkyDiscover adapter. Both share the same `TrustedVerificationEvaluator`, which runs four CHIA `@ChiaFunction` stages:

```text
prepare_artifact → verify_artifact → evaluate_artifact → feedback_for_search
```

Each stage passes its result to the next through a CHIA object reference. Only the final result is retrieved using `get()`. CHIA manages task dependencies and assigns stages to logical workers.

**CHIA LLM Agent** uses `LLMAgentSearchStrategy` with a `TrustedVerificationTool`. The agent calls `trusted_verification_evaluate_candidate`, which runs all four verification stages and returns the trusted results. The default Vertex adapter requires a verification call on the first model turn and allows normal tool selection afterward.

The four stages are not exposed as separate agent tools. Final results are also checked against the trusted evaluation log.

## 2. Logical workers

The `cluster/local_gemini.yaml` configuration defines two worker types:

| Worker | Resources | Purpose |
| --- | --- | --- |
| `trusted_project` | `teaal_evolve`, `trusted_verification_tool`, `vertex_creds` | Runs verification tasks and the optional direct Vertex agent |
| `opencode_vertex` | `opencode_creds` | Runs CHIA OpenCode model requests |

Both workers mount the repository at `/opt/chia-trusted-verification` and install the Python package inside their containers. The OpenCode worker also receives Google Cloud credentials and Vertex AI settings.

The cluster configuration includes several settings for reliable Docker execution:

- Workers mount the repository as read-only and install from a writable copy under `/tmp`.
- Temporary files use `/tmp` to avoid host filesystem permission issues.
- The Ray head uses `/tmp/ctv-${USER}` for temporary files and port `52365` for its dashboard agent.
- The OpenCode container uses `chia-ctv-opencode-${USER}` to avoid conflicts with other CHIA clusters.

For RTL formal verification, `cluster/local_gemini_formal.yaml` adds a `formal` worker. The same verification interface supports both backends without changing the search implementation.

## 3. Verify the workers

Set the repository path and start the CHIA cluster:

```bash
export CHIA_PROJECT_ROOT="$PWD"

chia up cluster/local_gemini.yaml --dry-run
chia up cluster/local_gemini.yaml

export RAY_ADDRESS="${CHIA_HEAD_IP}:6379"
ray status --address="$RAY_ADDRESS"
```

The Ray resource table should show `teaal_evolve`, `trusted_verification_tool`, and `opencode_creds`. The optional direct Vertex agent also uses `vertex_creds`.

Run the worker verification check:

```bash
python examples/check_chia_workers.py
```

Repository commands connect directly to Ray through `RAY_ADDRESS`. CHIA uses this connection to dispatch tasks to the configured workers.

## 4. Backend resource selection

The verification stages are shared across backends. Each backend specifies its resource requirements through `backend.resources`.

At runtime, `TrustedVerificationEvaluator` or `TrustedVerificationTool` applies these settings using `.options(resources=...)`.

For example, TeAAL-Evolve stages use the `teaal_evolve` resource, while RTL formal verification can run on the `formal` worker. This allows different backends to use the same CHIA verification stages with different execution environments.
