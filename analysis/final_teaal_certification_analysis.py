import argparse
import json
import math
from collections import Counter, defaultdict
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


BACKENDS = [
    "full",
    "hierarchical",
    "hierarchical_delta_syntactic",
    "hierarchical_delta_egraph",
]

DATASETS = [
    ("flash35", "Gemini 3.5 Flash", ROOTS["flash"], None),
    ("pro31_alpha", "Gemini 3.1 Pro", ROOTS["pro-alpha"], {"alphaevolve_structured", "alphaevolve_guided"}),
    ("pro31_oe", "Gemini 3.1 Pro", ROOTS["pro-openevolve"], {"openevolve_structured", "openevolve_guided"}),
    ("pro31_llm", "Gemini 3.1 Pro", ROOTS["pro-llm"], {"llm_agent_structured", "llm_agent_guided"}),
]


def mechanism_from_strategy(strategy):
    if strategy in {"structured_random", "structured_walk"}:
        return strategy

    if strategy and strategy.endswith("_structured"):
        return strategy[: -len("_structured")]

    if strategy and strategy.endswith("_guided"):
        return strategy[: -len("_guided")]

    return strategy


def feedback_from_strategy(strategy, row):
    value = row.get("feedback_mode")
    if value:
        return value

    if strategy and strategy.endswith("_guided"):
        return "guided"

    if strategy and strategy.endswith("_structured"):
        return "structured"

    return "baseline"


def as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_series(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").dropna()


# Load replay records.

raw_rows = []
replay_file_count = 0

for dataset_id, model_label, root, permitted in DATASETS:
    if not root.exists():
        raise RuntimeError(f"Missing dataset root: {root}")

    files = sorted(root.rglob("certification_replay.jsonl"))
    replay_file_count += len(files)

    for path in files:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

        for row in rows:
            strategy = row.get("search_strategy")

            if permitted is not None and strategy not in permitted:
                continue

            row = dict(row)
            row["_dataset_id"] = dataset_id
            row["_model_label"] = model_label
            row["_source_file"] = str(path)
            row["_mechanism"] = mechanism_from_strategy(strategy)
            row["_feedback"] = feedback_from_strategy(strategy, row)
            raw_rows.append(row)


print("\n=== RAW REPLAY DATASET ===")
print("replay files:", replay_file_count)
print("backend rows:", len(raw_rows))

if replay_file_count != 560:
    raise RuntimeError(
        f"Expected 560 replay files in the final dataset, found {replay_file_count}"
    )


# Match the four checks for each proposal.

groups = defaultdict(list)

for row in raw_rows:
    source_id = row.get("source_evaluation_id")

    if source_id is None:
        source_id = (
            f"fallback:{row.get('source_evaluation_sequence')}:"
            f"{row.get('proposal_index')}:{row.get('candidate_id')}"
        )

    key = (
        row["_dataset_id"],
        row.get("run_id"),
        source_id,
    )
    groups[key].append(row)


bad_groups = []
candidate_records = []
group_is_extra_base = {}

for key, rows in groups.items():
    by_backend = {
        row.get("checker_backend"): row
        for row in rows
    }

    if len(rows) != 4 or set(by_backend) != set(BACKENDS):
        bad_groups.append(
            {
                "key": key,
                "rows": len(rows),
                "backends": sorted(str(x) for x in by_backend),
            }
        )
        continue

    full = by_backend["full"]

    # Exclude unchanged OpenEvolve seeds from proposal counts.
    is_extra_open_seed = (
        full.get("base_search_engine") == "openevolve"
        and full.get("candidate_sha256") == full.get("base_candidate_sha256")
        and not full.get("mutation_ids")
        and full.get("proposal_index") == 1
    )

    group_is_extra_base[key] = is_extra_open_seed

    record = {
        "dataset_id": full["_dataset_id"],
        "model": full["_model_label"],
        "run_id": full.get("run_id"),
        "source_evaluation_id": full.get("source_evaluation_id"),
        "candidate_id": full.get("candidate_id"),
        "candidate_sha256": full.get("candidate_sha256"),
        "kernel": full.get("kernel"),
        "seed": full.get("seed"),
        "search_strategy": full.get("search_strategy"),
        "mechanism": full["_mechanism"],
        "feedback": full["_feedback"],
        "proposal_index": full.get("proposal_index"),
        "is_extra_open_seed": is_extra_open_seed,
    }

    for backend in BACKENDS:
        row = by_backend[backend]
        record[f"{backend}_legal"] = row.get("legal")
        record[f"{backend}_objective"] = as_float(row.get("objective_value"))
        record[f"{backend}_checker_time_s"] = as_float(row.get("checker_time_s"))
        record[f"{backend}_e2e_time_s"] = as_float(row.get("replay_e2e_time_s"))

    candidate_records.append(record)


if bad_groups:
    print("\nBAD MATCHED GROUPS:")
    for item in bad_groups[:20]:
        print(item)
    raise RuntimeError(
        f"{len(bad_groups)} replay groups did not contain exactly "
        "the four expected checker backends."
    )


candidate_df_all = pd.DataFrame(candidate_records)

extra_count = int(candidate_df_all["is_extra_open_seed"].sum())

candidate_df = candidate_df_all[
    ~candidate_df_all["is_extra_open_seed"]
].copy()

print("candidate groups (including OpenEvolve base extras):", len(candidate_df_all))
print("OpenEvolve base-evaluation extras:", extra_count)
print("proposal candidate groups:", len(candidate_df))

# Check proposal counts against the search results when available.
search_run_level = OUT / "run_level.csv"
if search_run_level.exists():
    search_df = pd.read_csv(search_run_level)
    expected_proposals = int(
        pd.to_numeric(search_df["proposals"], errors="raise").sum()
    )
    print("expected proposals from run_level.csv:", expected_proposals)

    if len(candidate_df) != expected_proposals:
        raise RuntimeError(
            "Replay/search proposal-count mismatch: "
            f"replay has {len(candidate_df)}, search analysis has "
            f"{expected_proposals}"
        )
else:
    print(
        "NOTE: run_level.csv was not found, so proposal count was not "
        "cross-checked against the search analysis."
    )


candidate_df.to_csv(
    OUT / "certification_candidate_level.csv",
    index=False,
)


# Collect replay rows for proposals only.

proposal_rows = []

for key, rows in groups.items():
    if key not in group_is_extra_base:
        continue

    if group_is_extra_base[key]:
        continue

    full = next(
        row
        for row in rows
        if row.get("checker_backend") == "full"
    )

    for row in rows:
        out = dict(row)
        out["_full_legal"] = bool(full.get("legal"))
        proposal_rows.append(out)


replay_df = pd.DataFrame(proposal_rows)

print("proposal backend rows:", len(replay_df))
print(
    "expected proposal backend rows:",
    len(candidate_df) * len(BACKENDS),
)

if len(replay_df) != len(candidate_df) * len(BACKENDS):
    raise RuntimeError(
        "Proposal replay row count is not exactly four rows per proposal."
    )


# Exclude proposals that failed before certification was possible.

status_counts = Counter(str(x) for x in replay_df["replay_status"])
print("replay status counts:", dict(status_counts))

unknown_status = replay_df[
    ~replay_df["replay_status"].isin(["ok", "unavailable"])
]
if len(unknown_status) != 0:
    raise RuntimeError(
        f"Found {len(unknown_status)} replay rows with unexpected status"
    )

# Check that all backends report the same replay status.
status_by_candidate = (
    replay_df.groupby(
        ["_dataset_id", "run_id", "source_evaluation_id"],
        dropna=False,
    )["replay_status"]
    .nunique()
)

mixed_status_groups = status_by_candidate[status_by_candidate != 1]
if len(mixed_status_groups) != 0:
    raise RuntimeError(
        f"Found {len(mixed_status_groups)} candidate groups with mixed "
        "replay statuses across backends"
    )

available_replay_df = replay_df[
    replay_df["replay_status"] == "ok"
].copy()

unavailable_replay_df = replay_df[
    replay_df["replay_status"] == "unavailable"
].copy()

# Check saved candidate hashes.
available_sha_bad = available_replay_df[
    available_replay_df["candidate_sha_matches_log"] != True
]
if len(available_sha_bad) != 0:
    raise RuntimeError(
        f"Found {len(available_sha_bad)} actual SHA mismatches among "
        "available replay rows"
    )

# Unavailable candidates have no hash.
unavailable_sha_false = unavailable_replay_df[
    unavailable_replay_df["candidate_sha_matches_log"] == False
]
if len(unavailable_sha_false) != 0:
    raise RuntimeError(
        f"Found {len(unavailable_sha_false)} unavailable rows with an "
        "explicit SHA mismatch"
    )

unavailable_keys = set(
    zip(
        unavailable_replay_df["_dataset_id"],
        unavailable_replay_df["run_id"],
        unavailable_replay_df["source_evaluation_id"],
    )
)

# Check unavailable proposals against the original logs.
precert_error_records = []
trusted_cache = {}
for _, row in unavailable_replay_df.drop_duplicates(
    ["_dataset_id", "run_id", "source_evaluation_id"]
).iterrows():
    replay_path = Path(row["_source_file"])
    trusted_path = replay_path.parent / "trusted_results.jsonl"
    if trusted_path not in trusted_cache:
        trusted_cache[trusted_path] = {
            r.get("evaluation_id"): r
            for r in (
                json.loads(line)
                for line in trusted_path.read_text().splitlines()
                if line.strip()
            )
        }

    source_id = row.get("source_evaluation_id")
    trusted = trusted_cache[trusted_path].get(source_id)
    if trusted is None:
        raise RuntimeError(
            f"Unavailable replay source {source_id} is absent from {trusted_path}"
        )

    metrics = trusted.get("metrics") or {}
    trace = trusted.get("trace") or {}
    is_precert_error = (
        trusted.get("role") == "proposal"
        and trusted.get("status") == "error"
        and trace.get("terminal_status") == "error"
        and float(metrics.get("runs_successfully", 0.0) or 0.0) == 0.0
        and float(metrics.get("objective_available", 0.0) or 0.0) == 0.0
    )
    if not is_precert_error:
        raise RuntimeError(
            "Found an unavailable replay group that is not a verified "
            f"pre-certification pipeline error: {source_id}"
        )

    precert_error_records.append(trusted)

candidate_df["replay_available"] = [
    (dataset_id, run_id, source_id) not in unavailable_keys
    for dataset_id, run_id, source_id in zip(
        candidate_df["dataset_id"],
        candidate_df["run_id"],
        candidate_df["source_evaluation_id"],
    )
]

cert_candidate_df = candidate_df[
    candidate_df["replay_available"]
].copy()

unavailable_candidate_df = candidate_df[
    ~candidate_df["replay_available"]
].copy()

print("replay-eligible materialized candidates:", len(cert_candidate_df))
print("pre-certification pipeline-error proposals:", len(unavailable_candidate_df))
print(
    "replay-eligible proposal pct:",
    round(100.0 * len(cert_candidate_df) / len(candidate_df), 3),
)
print("available SHA mismatches:", len(available_sha_bad))

# Save the replay status with each candidate.
candidate_df.to_csv(
    OUT / "certification_candidate_level.csv",
    index=False,
)


def replay_coverage_summary(frame, group_cols):
    if group_cols:
        summary = (
            frame.groupby(group_cols, as_index=False, dropna=False)
            .agg(
                proposals=("replay_available", "count"),
                replayable=("replay_available", "sum"),
            )
        )
    else:
        summary = pd.DataFrame(
            [{
                "proposals": len(frame),
                "replayable": int(frame["replay_available"].sum()),
            }]
        )

    summary["unavailable"] = (
        summary["proposals"] - summary["replayable"]
    )
    summary["coverage_pct"] = (
        100.0 * summary["replayable"] / summary["proposals"]
    )
    return summary


coverage_overall = replay_coverage_summary(candidate_df, [])
coverage_by_model = replay_coverage_summary(candidate_df, ["model"])
coverage_by_strategy = replay_coverage_summary(
    candidate_df, ["model", "search_strategy"]
)
coverage_by_kernel = replay_coverage_summary(candidate_df, ["kernel"])

coverage_overall.to_csv(
    OUT / "certification_replay_coverage.csv", index=False
)
coverage_by_model.to_csv(
    OUT / "certification_replay_coverage_by_model.csv", index=False
)
coverage_by_strategy.to_csv(
    OUT / "certification_replay_coverage_by_strategy.csv", index=False
)
coverage_by_kernel.to_csv(
    OUT / "certification_replay_coverage_by_kernel.csv", index=False
)


# Compare each backend with the full checker.

agreement_rows = []

for _, candidate in cert_candidate_df.iterrows():
    full_legal = bool(candidate["full_legal"])
    full_obj = candidate["full_objective"]

    for backend in BACKENDS[1:]:
        backend_legal = bool(candidate[f"{backend}_legal"])
        backend_obj = candidate[f"{backend}_objective"]

        agree = full_legal == backend_legal
        false_accept = (not full_legal) and backend_legal
        false_reject = full_legal and (not backend_legal)

        objective_mismatch = False

        if (
            full_legal
            and backend_legal
            and pd.notna(full_obj)
            and pd.notna(backend_obj)
        ):
            objective_mismatch = not math.isclose(
                float(full_obj),
                float(backend_obj),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )

        agreement_rows.append(
            {
                "model": candidate["model"],
                "mechanism": candidate["mechanism"],
                "feedback": candidate["feedback"],
                "search_strategy": candidate["search_strategy"],
                "kernel": candidate["kernel"],
                "backend": backend,
                "agree": int(agree),
                "false_accept": int(false_accept),
                "false_reject": int(false_reject),
                "objective_mismatch": int(objective_mismatch),
            }
        )


agreement_df = pd.DataFrame(agreement_rows)


def agreement_summary(frame, group_cols):
    summary = (
        frame.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            candidates=("agree", "count"),
            agreements=("agree", "sum"),
            false_accepts=("false_accept", "sum"),
            false_rejects=("false_reject", "sum"),
            objective_mismatches=("objective_mismatch", "sum"),
        )
    )

    summary["agreement_pct"] = (
        100.0
        * summary["agreements"]
        / summary["candidates"]
    )
    return summary


agreement_overall = agreement_summary(
    agreement_df,
    ["backend"],
)

agreement_by_model = agreement_summary(
    agreement_df,
    ["model", "backend"],
)

agreement_by_kernel = agreement_summary(
    agreement_df,
    ["kernel", "backend"],
)

agreement_by_strategy = agreement_summary(
    agreement_df,
    ["model", "search_strategy", "backend"],
)


agreement_overall.to_csv(
    OUT / "checker_agreement.csv",
    index=False,
)

agreement_by_model.to_csv(
    OUT / "checker_agreement_by_model.csv",
    index=False,
)

agreement_by_kernel.to_csv(
    OUT / "checker_agreement_by_kernel.csv",
    index=False,
)

agreement_by_strategy.to_csv(
    OUT / "checker_agreement_by_strategy.csv",
    index=False,
)


# Calculate speedups from mean checking times.


def runtime_summary(frame, group_cols):
    rows = []

    if group_cols:
        grouped = frame.groupby(
            group_cols,
            dropna=False,
        )
    else:
        grouped = [((), frame)]

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        prefix = dict(zip(group_cols, group_key))

        full = group[
            group["checker_backend"] == "full"
        ]

        full_checker = numeric_series(full, "checker_time_s")
        full_e2e = numeric_series(full, "replay_e2e_time_s")

        full_checker_mean = full_checker.mean()
        full_e2e_mean = full_e2e.mean()

        for backend in BACKENDS:
            b = group[
                group["checker_backend"] == backend
            ]

            checker = numeric_series(b, "checker_time_s")
            e2e = numeric_series(b, "replay_e2e_time_s")

            if checker.empty or e2e.empty:
                continue

            checker_mean = checker.mean()
            e2e_mean = e2e.mean()

            recorded_checker_speedup = numeric_series(
                b,
                "checker_speedup_vs_full",
            )
            recorded_e2e_speedup = numeric_series(
                b,
                "e2e_speedup_vs_full",
            )

            row = dict(prefix)
            row.update(
                {
                    "backend": backend,
                    "candidates": len(b),
                    "mean_checker_ms": 1000.0 * checker_mean,
                    "median_checker_ms": 1000.0 * checker.median(),
                    "p95_checker_ms": 1000.0 * checker.quantile(0.95),
                    "checker_speedup_ratio_of_means": (
                        full_checker_mean / checker_mean
                    ),
                    "mean_pairwise_checker_speedup": (
                        recorded_checker_speedup.mean()
                    ),
                    "mean_e2e_s": e2e_mean,
                    "median_e2e_s": e2e.median(),
                    "p95_e2e_s": e2e.quantile(0.95),
                    "e2e_speedup_ratio_of_means": (
                        full_e2e_mean / e2e_mean
                    ),
                    "mean_pairwise_e2e_speedup": (
                        recorded_e2e_speedup.mean()
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


runtime_overall = runtime_summary(
    available_replay_df,
    [],
)

runtime_by_model = runtime_summary(
    available_replay_df,
    ["_model_label"],
).rename(
    columns={"_model_label": "model"}
)

runtime_by_strategy = runtime_summary(
    available_replay_df,
    ["_model_label", "search_strategy"],
).rename(
    columns={"_model_label": "model"}
)

runtime_by_kernel = runtime_summary(
    available_replay_df,
    ["kernel"],
)

runtime_by_legality = runtime_summary(
    available_replay_df,
    ["_full_legal"],
).rename(
    columns={"_full_legal": "full_legal"}
)


runtime_overall.to_csv(
    OUT / "certification_runtime.csv",
    index=False,
)

runtime_by_model.to_csv(
    OUT / "certification_runtime_by_model.csv",
    index=False,
)

runtime_by_strategy.to_csv(
    OUT / "certification_runtime_by_strategy.csv",
    index=False,
)

runtime_by_kernel.to_csv(
    OUT / "checker_by_kernel.csv",
    index=False,
)

runtime_by_legality.to_csv(
    OUT / "certification_runtime_by_legality.csv",
    index=False,
)


# Measure reused verification work.

delta_df = available_replay_df[
    available_replay_df["checker_backend"].isin(
        [
            "hierarchical_delta_syntactic",
            "hierarchical_delta_egraph",
        ]
    )
].copy()


def reuse_summary(frame, group_cols):
    rows = []

    if group_cols:
        grouped = frame.groupby(
            group_cols,
            dropna=False,
        )
    else:
        grouped = [((), frame)]

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        prefix = dict(zip(group_cols, group_key))

        for backend in [
            "hierarchical_delta_syntactic",
            "hierarchical_delta_egraph",
        ]:
            b = group[
                group["checker_backend"] == backend
            ]

            if b.empty:
                continue

            hits = pd.to_numeric(
                b["cache_hits"],
                errors="coerce",
            ).fillna(0.0)

            misses = pd.to_numeric(
                b["cache_misses"],
                errors="coerce",
            ).fillna(0.0)

            total = float((hits + misses).sum())

            weighted_cache = (
                float(hits.sum()) / total
                if total
                else None
            )

            legality_hits = pd.to_numeric(
                b["legality_hit_count"],
                errors="coerce",
            ).fillna(0.0)

            legality_misses = pd.to_numeric(
                b["legality_miss_count"],
                errors="coerce",
            ).fillna(0.0)

            legality_total = float(
                (legality_hits + legality_misses).sum()
            )

            extra = float(
                pd.to_numeric(
                    b["extra_reused_legality_count"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

            rewrite_denominator = float(
                pd.to_numeric(
                    b["rewrite_recovery_denominator"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

            # E-graph hits also include rewrite matches.
            syntactic_legality_hits = (
                float(legality_hits.sum()) - extra
            )

            syntactic_legality_rate = (
                syntactic_legality_hits / legality_total
                if legality_total
                else None
            )

            rewrite_legality_rate = (
                float(legality_hits.sum()) / legality_total
                if legality_total
                else None
            )

            rewrite_recovery_fraction = (
                extra / rewrite_denominator
                if rewrite_denominator
                else None
            )

            fallback_count = int(
                b["incremental_fallback_reason"]
                .notna()
                .sum()
            )

            mean_recorded_cache_hit_rate = pd.to_numeric(
                b["cache_hit_rate"],
                errors="coerce",
            ).mean()

            row = dict(prefix)
            row.update(
                {
                    "backend": backend,
                    "candidates": len(b),
                    "obligations": int(total),
                    "weighted_cache_hit_pct": (
                        100.0 * weighted_cache
                        if weighted_cache is not None
                        else None
                    ),
                    "mean_recorded_cache_hit_pct": (
                        100.0 * mean_recorded_cache_hit_rate
                        if pd.notna(mean_recorded_cache_hit_rate)
                        else None
                    ),
                    "syntactic_legality_reuse_pct": (
                        100.0 * syntactic_legality_rate
                        if syntactic_legality_rate is not None
                        else None
                    ),
                    "rewrite_legality_reuse_pct": (
                        100.0 * rewrite_legality_rate
                        if rewrite_legality_rate is not None
                        else None
                    ),
                    "additional_rewrite_reuse": int(extra),
                    "rewrite_recovered_pct_of_syntactic_misses": (
                        100.0 * rewrite_recovery_fraction
                        if rewrite_recovery_fraction is not None
                        else None
                    ),
                    "fallback_count": fallback_count,
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


reuse_overall = reuse_summary(
    delta_df,
    [],
)

reuse_by_model = reuse_summary(
    delta_df,
    ["_model_label"],
).rename(
    columns={"_model_label": "model"}
)

reuse_by_strategy = reuse_summary(
    delta_df,
    ["_model_label", "search_strategy"],
).rename(
    columns={"_model_label": "model"}
)

reuse_by_kernel = reuse_summary(
    delta_df,
    ["kernel"],
)


reuse_overall.to_csv(
    OUT / "reuse_summary.csv",
    index=False,
)

reuse_by_model.to_csv(
    OUT / "reuse_by_model.csv",
    index=False,
)

reuse_by_strategy.to_csv(
    OUT / "reuse_by_strategy.csv",
    index=False,
)

reuse_by_kernel.to_csv(
    OUT / "reuse_by_kernel.csv",
    index=False,
)


# Count reuse sources and fallbacks.

source_counts = Counter(
    str(value)
    for value in delta_df[
        "incremental_delta_source"
    ].dropna()
)

fallback_counts = Counter(
    str(value)
    for value in delta_df[
        "incremental_fallback_reason"
    ].dropna()
)


# Print summary tables.

print("\n=== CERTIFICATION REPLAY ELIGIBILITY ===\n")

display = coverage_overall.copy()
display["coverage_pct"] = display["coverage_pct"].round(3)
print(display.to_string(index=False))

print("\nPre-certification pipeline errors by model/search strategy:")
unavailable_display = coverage_by_strategy[
    coverage_by_strategy["unavailable"] > 0
].copy()
unavailable_display["coverage_pct"] = (
    unavailable_display["coverage_pct"].round(3)
)
if len(unavailable_display):
    print(unavailable_display.to_string(index=False))
else:
    print("none")


print("\n=== FULL VS. TRUSTED BACKENDS ===\n")

display = agreement_overall.copy()
display["agreement_pct"] = display["agreement_pct"].round(6)
print(display.to_string(index=False))


print("\n=== CERTIFICATION RUNTIME ===\n")

runtime_columns = [
    "backend",
    "candidates",
    "mean_checker_ms",
    "checker_speedup_ratio_of_means",
    "mean_e2e_s",
    "e2e_speedup_ratio_of_means",
]

display = runtime_overall[runtime_columns].copy()

for column in [
    "mean_checker_ms",
    "checker_speedup_ratio_of_means",
    "mean_e2e_s",
    "e2e_speedup_ratio_of_means",
]:
    display[column] = display[column].round(3)

print(display.to_string(index=False))


print("\n=== CERTIFICATION RUNTIME BY FULL LEGALITY ===\n")

legality_columns = [
    "full_legal",
    "backend",
    "candidates",
    "mean_checker_ms",
    "checker_speedup_ratio_of_means",
    "mean_e2e_s",
    "e2e_speedup_ratio_of_means",
]

display = runtime_by_legality[legality_columns].copy()

for column in [
    "mean_checker_ms",
    "checker_speedup_ratio_of_means",
    "mean_e2e_s",
    "e2e_speedup_ratio_of_means",
]:
    display[column] = display[column].round(3)

print(display.to_string(index=False))


print("\n=== INCREMENTAL / REWRITE REUSE ===\n")

display = reuse_overall.copy()

for column in [
    "weighted_cache_hit_pct",
    "mean_recorded_cache_hit_pct",
    "syntactic_legality_reuse_pct",
    "rewrite_legality_reuse_pct",
    "rewrite_recovered_pct_of_syntactic_misses",
]:
    display[column] = display[column].round(2)

print(display.to_string(index=False))


print("\n=== DELTA SOURCE COUNTS ===")
for key, value in sorted(source_counts.items()):
    print(f"{key}: {value}")


print("\n=== FALLBACK COUNTS ===")
if fallback_counts:
    for key, value in sorted(fallback_counts.items()):
        print(f"{key}: {value}")
else:
    print("none")


print("\n=== OUTPUT FILES ===")

output_files = [
    "certification_candidate_level.csv",
    "certification_replay_coverage.csv",
    "certification_replay_coverage_by_model.csv",
    "certification_replay_coverage_by_strategy.csv",
    "certification_replay_coverage_by_kernel.csv",
    "checker_agreement.csv",
    "checker_agreement_by_model.csv",
    "checker_agreement_by_kernel.csv",
    "checker_agreement_by_strategy.csv",
    "certification_runtime.csv",
    "certification_runtime_by_model.csv",
    "certification_runtime_by_strategy.csv",
    "certification_runtime_by_legality.csv",
    "checker_by_kernel.csv",
    "reuse_summary.csv",
    "reuse_by_model.csv",
    "reuse_by_strategy.csv",
    "reuse_by_kernel.csv",
]

for filename in output_files:
    print(OUT / filename)
