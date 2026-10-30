# Result Files and Analysis

A completed TeAAL-Evolve search run writes these files under `runs/`:

| File | What it records |
| --- | --- |
| `run_manifest.json` | Kernel or intent, search method, seed, model, settings, timing, and run status |
| `trusted_results.jsonl` | Each evaluated proposal's legality or proof result, score, feedback, and stage times |
| `search_result.json` | Final best candidate and any model usage or cost reported by the search service |
| `certification_replay.jsonl` | Matched TeAAL checks on saved proposals, when replay is enabled |
| `candidate_programs/` | Evaluated source saved once per source digest |

`run_id` identifies a planned run, and `run_instance_id` identifies an execution or retry. `evaluation_sequence` orders trusted evaluations. `candidate_sha256` identifies a TeAAL candidate, while `catalog_sha256` identifies its allowed change list. These field names are part of the saved result format.

The run starts with a separately labeled `seed_baseline`. OpenEvolve or AlphaEvolve may also evaluate their initial program with the role `search_seed`. Neither is counted as a generated proposal. Before recording `best_score` and `best_solution`, the pipeline checks the engine's selected candidate against its trusted log.

## TeAAL certification replay

Live search uses the configured trusted checker. After a run, `experiments/run_teaal_experiments.py` checks its saved proposals with Full, Hierarchical, Delta-Syntactic, and Delta-Rewrite strategies. This replay makes no new model calls. Each row records legal status, agreement with Full, timing, and reuse of earlier checks.

To summarize new runs:

~~~bash
python analysis/summarize_teaal_results.py --runs runs --output results/teaal_summary
~~~

The summarizer writes run, model, candidate, cost, failure, and certification CSVs. It includes proposal index and time to the best trusted candidate, legal and improving candidate rates, checker agreement, and per-stage runtime. Model token counts and reported cost appear when the provider supplies them or the config supplies rates. The cost file keeps API or compute costs separate from credits.

The final processed tables are in [report_data](../report_data/README.md). Recalculating them requires the separately archived raw runs. See [reproducing results](REPRODUCE_RESULTS.md).

