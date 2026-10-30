import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the final TeAAL-Evolve runs.")
    parser.add_argument(
        "--input-root", type=Path, default=Path("runs"),
        help="Directory containing the four final experiment folders (default: runs)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/final_teaal_analysis"),
        help="Directory for the output CSV files",
    )
    for name in ("flash", "pro-alpha", "pro-openevolve", "pro-llm"):
        parser.add_argument(
            f"--{name}-root", type=Path,
            help=f"Override the {name} experiment directory",
        )
    return parser.parse_args()


args = parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)

ROOTS = {
    "flash": args.flash_root or args.input_root / "teaal_chia_challenge_gemini35flash",
    "pro-alpha": args.pro_alpha_root or args.input_root
    / "_pro31_alphaevolve_final_stage/teaal_chia_challenge_gemini31pro",
    "pro-openevolve": args.pro_openevolve_root or args.input_root
    / "_pro31_openevolve_final_stage/teaal_chia_challenge_gemini31pro",
    "pro-llm": args.pro_llm_root or args.input_root
    / "_pro31_llm_fixed30_final_stage/teaal_chia_challenge_gemini31propreview_llm_fixed30",
}



DATASETS = [
    ("Gemini 3.5 Flash", ROOTS["flash"], None),
    ("Gemini 3.1 Pro", ROOTS["pro-alpha"], {"alphaevolve_structured", "alphaevolve_guided"}),
    ("Gemini 3.1 Pro", ROOTS["pro-openevolve"], {"openevolve_structured", "openevolve_guided"}),
    ("Gemini 3.1 Pro", ROOTS["pro-llm"], {"llm_agent_structured", "llm_agent_guided"}),
]


def objective(row):
    value = row.get("metrics", {}).get("objective_value")

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def engine_parts(engine):
    if engine in {"structured_random", "structured_walk"}:
        return engine, "baseline"

    if engine.endswith("_structured"):
        return engine[: -len("_structured")], "structured"

    if engine.endswith("_guided"):
        return engine[: -len("_guided")], "guided"

    return engine, "unknown"


records = []


for model_label, root, permitted_engines in DATASETS:
    if not root.exists():
        raise RuntimeError(f"Missing dataset root: {root}")

    for path in sorted(root.rglob("trusted_results.jsonl")):
        rel = path.relative_to(root).parts

        if not rel:
            continue

        engine = rel[0]

        if (
            permitted_engines is not None
            and engine not in permitted_engines
        ):
            continue

        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

        baselines = [
            row
            for row in rows
            if row.get("role") == "seed_baseline"
        ]

        proposals = [
            row
            for row in rows
            if row.get("role") == "proposal"
        ]

        if len(baselines) != 1:
            raise RuntimeError(
                f"{path}: expected 1 seed_baseline, "
                f"found {len(baselines)}"
            )

        baseline = baselines[0]
        seed_obj = objective(baseline)

        # Exclude unchanged OpenEvolve seeds from proposal counts.
        if engine.startswith("openevolve"):
            cleaned = []

            for row in proposals:
                trace_proposal = (
                    row.get("trace", {}).get("proposal", {})
                )

                label = (
                    row.get("candidate_label")
                    or row.get("proposal_label")
                    or trace_proposal.get("label")
                )

                mutation_ids = (
                    row.get("mutation_ids")
                    or trace_proposal.get("mutation_ids")
                    or []
                )

                row_obj = objective(row)

                is_seed_like = (
                    label == "seed"
                    and not mutation_ids
                    and row_obj == seed_obj
                )

                if not is_seed_like:
                    cleaned.append(row)

            proposals = cleaned

        legal_rows = [
            row
            for row in proposals
            if row.get("status") == "pass"
        ]

        legal_with_objective = [
            row
            for row in legal_rows
            if objective(row) is not None
        ]

        failed_rows = [
            row
            for row in proposals
            if row.get("status") == "fail"
        ]

        error_rows = [
            row
            for row in proposals
            if row.get("status") == "error"
        ]

        legal_objectives = [
            objective(row)
            for row in legal_with_objective
        ]

        if seed_obj is not None:
            improving_objectives = [
                value
                for value in legal_objectives
                if value < seed_obj
            ]
        else:
            improving_objectives = []

        candidate_objectives = []

        if seed_obj is not None:
            candidate_objectives.append(seed_obj)

        candidate_objectives.extend(legal_objectives)

        if candidate_objectives:
            best_seen_obj = min(candidate_objectives)
        else:
            best_seen_obj = None

        best_improvement = None

        if (
            seed_obj is not None
            and seed_obj != 0
            and best_seen_obj is not None
        ):
            best_improvement = (
                seed_obj - best_seen_obj
            ) / seed_obj

            best_improvement = max(
                0.0,
                best_improvement,
            )

        mechanism, feedback = engine_parts(engine)

        kernel = baseline.get("kernel")
        seed = baseline.get("seed")

        if kernel is None and len(rel) >= 3:
            kernel = rel[2]

        if seed is None and len(rel) >= 4:
            seed_name = rel[3]

            try:
                seed = int(seed_name.split("_")[-1])
            except ValueError:
                seed = seed_name

        proposal_count = len(proposals)
        legal_count = len(legal_rows)
        improving_count = len(improving_objectives)

        legal_rate = (
            legal_count / proposal_count
            if proposal_count
            else 0.0
        )

        improving_rate = (
            improving_count / proposal_count
            if proposal_count
            else 0.0
        )

        records.append(
            {
                "model": model_label,
                "engine": engine,
                "mechanism": mechanism,
                "feedback": feedback,
                "kernel": kernel,
                "seed": seed,
                "seed_objective": seed_obj,
                "proposals": proposal_count,
                "legal": legal_count,
                "fail": len(failed_rows),
                "errors": len(error_rows),
                "legal_rate": legal_rate,
                "improving": improving_count,
                "improving_rate": improving_rate,
                "found_improvement": int(
                    improving_count > 0
                ),
                "best_seen_objective": best_seen_obj,
                "best_improvement": best_improvement,
            }
        )


df = pd.DataFrame(records)

print("\n=== FINAL DATASET ===")
print("run-level experiments:", len(df))

print("\nRuns by model:")
print(df.groupby("model").size().to_string())

print("\nRuns by engine:")
print(df.groupby(["model", "engine"]).size().to_string())


if len(df) != 560:
    raise RuntimeError(
        f"Expected 560 run-level experiments, found {len(df)}"
    )


df.to_csv(
    OUT / "run_level.csv",
    index=False,
)


# Overall search summary

search_summary = (
    df.groupby(
        [
            "model",
            "mechanism",
            "feedback",
        ],
        as_index=False,
    )
    .agg(
        runs=("seed", "count"),
        proposals=("proposals", "sum"),
        legal=("legal", "sum"),
        improving=("improving", "sum"),
        runs_with_improvement=(
            "found_improvement",
            "sum",
        ),
        mean_best_improvement=(
            "best_improvement",
            "mean",
        ),
        median_best_improvement=(
            "best_improvement",
            "median",
        ),
        max_best_improvement=(
            "best_improvement",
            "max",
        ),
    )
)


search_summary["legal_pct"] = (
    100.0
    * search_summary["legal"]
    / search_summary["proposals"]
)

search_summary["improving_pct"] = (
    100.0
    * search_summary["improving"]
    / search_summary["proposals"]
)

search_summary["run_success_pct"] = (
    100.0
    * search_summary["runs_with_improvement"]
    / search_summary["runs"]
)

for column in [
    "mean_best_improvement",
    "median_best_improvement",
    "max_best_improvement",
]:
    search_summary[column] *= 100.0


search_summary.to_csv(
    OUT / "search_summary.csv",
    index=False,
)


# Per-kernel summary

kernel_summary = (
    df.groupby(
        [
            "model",
            "mechanism",
            "feedback",
            "kernel",
        ],
        as_index=False,
    )
    .agg(
        runs=("seed", "count"),
        proposals=("proposals", "sum"),
        legal=("legal", "sum"),
        improving=("improving", "sum"),
        runs_with_improvement=(
            "found_improvement",
            "sum",
        ),
        mean_best_improvement=(
            "best_improvement",
            "mean",
        ),
        median_best_improvement=(
            "best_improvement",
            "median",
        ),
    )
)


kernel_summary["legal_pct"] = (
    100.0
    * kernel_summary["legal"]
    / kernel_summary["proposals"]
)

kernel_summary["improving_pct"] = (
    100.0
    * kernel_summary["improving"]
    / kernel_summary["proposals"]
)

kernel_summary["run_success_pct"] = (
    100.0
    * kernel_summary["runs_with_improvement"]
    / kernel_summary["runs"]
)

kernel_summary["mean_best_improvement"] *= 100.0
kernel_summary["median_best_improvement"] *= 100.0


kernel_summary.to_csv(
    OUT / "kernel_summary.csv",
    index=False,
)


# Compare methods used by both models.

agentic = df[
    df["mechanism"].isin(
        [
            "alphaevolve",
            "openevolve",
            "llm_agent",
        ]
    )
].copy()


model_sensitivity = (
    agentic.groupby(
        [
            "mechanism",
            "feedback",
            "model",
        ],
        as_index=False,
    )
    .agg(
        runs=("seed", "count"),
        proposals=("proposals", "sum"),
        legal=("legal", "sum"),
        improving=("improving", "sum"),
        runs_with_improvement=(
            "found_improvement",
            "sum",
        ),
        mean_best_improvement=(
            "best_improvement",
            "mean",
        ),
        median_best_improvement=(
            "best_improvement",
            "median",
        ),
    )
)


model_sensitivity["legal_pct"] = (
    100.0
    * model_sensitivity["legal"]
    / model_sensitivity["proposals"]
)

model_sensitivity["improving_pct"] = (
    100.0
    * model_sensitivity["improving"]
    / model_sensitivity["proposals"]
)

model_sensitivity["run_success_pct"] = (
    100.0
    * model_sensitivity["runs_with_improvement"]
    / model_sensitivity["runs"]
)

model_sensitivity["mean_best_improvement"] *= 100.0
model_sensitivity["median_best_improvement"] *= 100.0


model_sensitivity.to_csv(
    OUT / "model_sensitivity.csv",
    index=False,
)


# Print summary tables.

print("\n=== SEARCH SUMMARY ===\n")

display_columns = [
    "model",
    "mechanism",
    "feedback",
    "runs",
    "proposals",
    "legal_pct",
    "improving_pct",
    "run_success_pct",
    "mean_best_improvement",
    "median_best_improvement",
]

display = search_summary[
    display_columns
].copy()

for column in [
    "legal_pct",
    "improving_pct",
    "run_success_pct",
    "mean_best_improvement",
    "median_best_improvement",
]:
    display[column] = display[column].round(2)

print(display.to_string(index=False))


print("\n=== MODEL SENSITIVITY ===\n")

display = model_sensitivity.copy()

for column in [
    "legal_pct",
    "improving_pct",
    "run_success_pct",
    "mean_best_improvement",
    "median_best_improvement",
]:
    display[column] = display[column].round(2)

print(display.to_string(index=False))


print("\n=== OUTPUT FILES ===")

for filename in [
    "run_level.csv",
    "search_summary.csv",
    "kernel_summary.csv",
    "model_sensitivity.csv",
]:
    print(OUT / filename)
