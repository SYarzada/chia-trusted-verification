# Google Authentication and Model Setup

Google-based search methods use Application Default Credentials (ADC) to access Vertex AI. Use a Google Cloud project with access to the required models and keep your credentials outside the repository.

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_GOOGLE_CLOUD_PROJECT"
export GOOGLE_CLOUD_LOCATION=global
export VERTEX_LOCATION=global

gcloud auth application-default login --no-launch-browser \
  --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
gcloud services enable aiplatform.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
gcloud auth application-default print-access-token >/dev/null
```

The three agentic search methods use Google Cloud differently:

| Search method | Model access | Requirements |
| --- | --- | --- |
| OpenEvolve | Gemini through Vertex AI using SkyDiscover/evolve-flows | Google Cloud project and ADC |
| AlphaEvolve | Google's managed AlphaEvolve service | ADC, service access, and AlphaEvolve engine settings |
| CHIA LLM Agent | Gemini through CHIA with a trusted verification tool | Google Cloud project and ADC |

OpenEvolve uses an ADC access token for each run. The CHIA LLM Agent calls trusted verification on its first tool turn and uses normal tool selection afterward.

AlphaEvolve requires additional configuration for a managed engine:

```bash
export ALPHAEVOLVE_PROJECT_ID="$GOOGLE_CLOUD_PROJECT"
export ALPHAEVOLVE_ENGINE_ID="YOUR_ALPHAEVOLVE_ENGINE_ID"
export ALPHAEVOLVE_LOCATION=global
export ALPHAEVOLVE_COLLECTION=default_collection
export ALPHAEVOLVE_ASSISTANT=default_assistant

gcloud services enable discoveryengine.googleapis.com \
  --project "$ALPHAEVOLVE_PROJECT_ID"
```

The CHIA cluster makes local Google credentials available to its Docker workers. Before starting the cluster, set `CHIA_PROJECT_ROOT` to the repository path and `CHIA_HEAD_IP` to the SSH-accessible host address.

See the [TeAAL-Evolve Quickstart](TEAAL_EVOLVE_QUICKSTART.md) for setup instructions.
