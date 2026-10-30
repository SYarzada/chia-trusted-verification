
# TeAAL-Evolve Search Methods

TeAAL-Evolve supports different search methods for accelerator design space exploration. Each method generates candidates that are checked through the same trusted verification interface.

## 1. Search methods

| Method | Description |
| --- | --- |
| `openevolve` | Uses OpenEvolve through evolve-flows/SkyDiscover |
| `alphaevolve` | Uses Google's managed AlphaEvolve service through the same adapter |
| `llm_agent` | Uses a CHIA LLM Agent with a trusted verification tool |
| `structured_random` | Randomly samples valid choices from the structured design space |
| `structured_walk` | Explores the design space through local mutations |

OpenEvolve, AlphaEvolve, and the CHIA LLM Agent support two modes:

- **Structured (`*_structured`):** Uses verification scores without detailed repair feedback.
- **Guided (`*_guided`):** Uses verification scores and detailed feedback to guide subsequent proposals.

Both modes use the same design space, verification rules, and optimization objective within each comparison.

## 2. Scores and feedback

The TeAAL-Evolve backend assigns scores based on design validity and optimization results.

| Result | Score |
| --- | --- |
| Legal candidate | Greater than 1, with higher scores for better objectives |
| Illegal candidate | Negative score based on failed verification checks |
| Objective evaluation failure | `-50` |
| Invalid candidate representation or verification execution error | `-100` |

AlphaEvolve uses `combined_score` as its optimization metric and receives verification details as additional feedback.

The search methods access Gemini differently:

- **OpenEvolve:** Gemini through Vertex AI's OpenAI-compatible endpoint.
- **AlphaEvolve:** Google's managed AlphaEvolve service.
- **CHIA LLM Agent:** Gemini through direct Vertex AI calls.

See [Google Authentication and Model Setup](GOOGLE_GEMINI.md) for configuration instructions.

## 3. Backend search guidance

TeAAL-Evolve represents design changes using a mutable `CANDIDATE` field and a fixed `MUTATION_CATALOG` of allowed choices.

The backend provides these instructions through `SearchGuidance`, while the search methods handle candidate generation and exploration.

Other backends can provide different artifact formats and instructions. For example, the Gemmini RTL formal verification backend generates SystemVerilog assertions and cover properties using the same search interface.

See [Adding a Trusted Verification Backend](ADDING_BACKENDS.md) for details.
