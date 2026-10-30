from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _nested(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    numeric = _float(value)
    return None if numeric is None else int(numeric)


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else sum(clean) / len(clean)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
        rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _find_numeric(value: Any, aliases: tuple[str, ...]) -> float | None:
    if isinstance(value, Mapping):
        lowered = {str(key).lower(): item for key, item in value.items()}
        for alias in aliases:
            if alias in lowered:
                candidate = lowered[alias]
                if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                    return float(candidate)
        for child in value.values():
            found = _find_numeric(child, aliases)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_numeric(child, aliases)
            if found is not None:
                return found
    return None


def _candidate_flat(row: Mapping[str, Any], seed_objective: float | None) -> dict[str, Any]:
    metrics = dict(row.get("metrics", {}))
    trace = dict(row.get("trace", {}))
    certificate = _nested(row, "evidence", "certificate", default={}) or {}
    execution = dict(certificate.get("execution", {})) if isinstance(certificate, Mapping) else {}
    timings = dict(row.get("stage_timings_s", {}))
    objective_value = _float(metrics.get("objective_value"))
    legal = (_float(metrics.get("legal")) or 0.0) > 0.0
    objective_available = (_float(metrics.get("objective_available")) or 0.0) > 0.0
    relative_improvement = None
    improving = False
    if legal and objective_available and objective_value is not None and seed_objective not in (None, 0.0):
        relative_improvement = (seed_objective - objective_value) / seed_objective
        improving = objective_value < seed_objective
    stage_hits = execution.get("stage_hit_ids") or []
    stage_misses = execution.get("stage_miss_ids") or []
    interface_hits = execution.get("interface_hit_ids") or []
    interface_misses = execution.get("interface_miss_ids") or []
    legality_hits = execution.get("legality_hit_ids") or []
    legality_misses = execution.get("legality_miss_ids") or []
    return {
        "run_id": row.get("run_id"),
        "run_instance_id": row.get("run_instance_id"),
        "experiment_name": row.get("experiment_name"),
        "kernel": row.get("kernel"),
        "search_strategy": row.get("search_strategy"),
        "base_search_engine": row.get("base_search_engine"),
        "feedback_mode": row.get("feedback_mode"),
        "seed": row.get("seed"),
        "model": row.get("model"),
        "provider": row.get("provider"),
        "role": row.get("role"),
        "evaluation_origin": row.get("evaluation_origin"),
        "evaluation_index": row.get("evaluation_index"),
        "evaluation_sequence": row.get("evaluation_sequence", row.get("evaluation_index")),
        "evaluation_local_index": row.get("evaluation_local_index"),
        "evaluation_id": row.get("evaluation_id"),
        "candidate_id": row.get("candidate_id"),
        "candidate_label": row.get("candidate_label"),
        "proposal_label": row.get("proposal_label"),
        "status": row.get("status"),
        "candidate_sha256": row.get("candidate_sha256") or trace.get("candidate_sha256"),
        "catalog_sha256": row.get("catalog_sha256") or trace.get("catalog_sha256"),
        "legal": metrics.get("legal"),
        "runs_successfully": metrics.get("runs_successfully"),
        "objective_available": metrics.get("objective_available"),
        "objective_value": metrics.get("objective_value"),
        "relative_improvement": relative_improvement,
        "improving": 1 if improving else 0,
        "combined_score": metrics.get("combined_score"),
        "failed_checks": metrics.get("failed_checks"),
        "pipeline_wall_time_s": row.get("pipeline_wall_time_s"),
        "elapsed_since_run_start_s": row.get("elapsed_since_run_start_s"),
        "prepare_time_s": timings.get("prepare"),
        "verify_time_s": timings.get("verify"),
        "evaluate_time_s": timings.get("evaluate"),
        "feedback_time_s": timings.get("feedback"),
        "checker_backend": certificate.get("checker_backend") if isinstance(certificate, Mapping) else None,
        "certificate_size_bytes": _nested(row, "evidence", "certificate_size_bytes"),
        "certificate_validation_time_s": _nested(row, "evidence", "certificate_validation_time_s"),
        "required_obligation_count": certificate.get("required_obligation_count") if isinstance(certificate, Mapping) else None,
        "reused_obligation_count": certificate.get("reused_obligation_count") if isinstance(certificate, Mapping) else None,
        "freshly_checked_obligation_count": certificate.get("freshly_checked_obligation_count") if isinstance(certificate, Mapping) else None,
        "checker_mode": execution.get("mode"),
        "cache_hits": execution.get("cache_hits"),
        "cache_misses": execution.get("cache_misses"),
        "total_obligations_processed": execution.get("total_obligations_processed"),
        "cache_hit_rate": execution.get("cache_hit_rate"),
        "stage_hit_count": len(stage_hits),
        "stage_miss_count": len(stage_misses),
        "interface_hit_count": len(interface_hits),
        "interface_miss_count": len(interface_misses),
        "legality_hit_count": len(legality_hits),
        "legality_miss_count": len(legality_misses),
        "syntactic_legality_reuse_rate": execution.get("syntactic_legality_reuse_rate"),
        "rewrite_legality_reuse_rate": execution.get("rewrite_legality_reuse_rate"),
        "extra_reused_legality_count": execution.get("extra_reused_legality_count"),
        "incremental_fallback_reason": execution.get("incremental_fallback_reason"),
        "incremental_delta_source": execution.get("incremental_delta_source"),
        "mutation_ids": json.dumps(
            row.get("mutation_ids")
            or _nested(row, "trace", "proposal", "mutation_ids", default=[])
            or _nested(row, "feedback", "mutation_ids", default=[])
        ),
        "failed_categories": json.dumps(_nested(row, "feedback", "failed_categories", default=[])),
    }


def _failure_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = {
        "run_id": row.get("run_id"),
        "run_instance_id": row.get("run_instance_id"),
        "kernel": row.get("kernel"),
        "search_strategy": row.get("search_strategy"),
        "base_search_engine": row.get("base_search_engine"),
        "feedback_mode": row.get("feedback_mode"),
        "seed": row.get("seed"),
        "model": row.get("model"),
        "evaluation_index": row.get("evaluation_index"),
        "evaluation_id": row.get("evaluation_id"),
        "candidate_id": row.get("candidate_id"),
    }
    result: list[dict[str, Any]] = []
    categories = _nested(row, "feedback", "failed_categories", default=[]) or []
    if not categories and row.get("status") == "error":
        diagnostics = row.get("diagnostics", []) or []
        if diagnostics:
            categories = [str(diagnostics[0].get("code", "pipeline_error")).split(".", 1)[0]]
        else:
            categories = ["pipeline_error"]
    for category in categories:
        result.append({**base, "failure_category": str(category)})
    return result


def _seed_objective(candidates: list[dict[str, Any]]) -> float | None:
    for row in candidates:
        if row.get("role") != "seed_baseline":
            continue
        value = _float(_nested(row, "metrics", "objective_value"))
        if value is not None and _float(_nested(row, "metrics", "legal")) == 1.0:
            return value
    return None


def _ordered_proposals(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in candidates if row.get("role") == "proposal"],
        key=lambda row: int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0),
    )


def _progress_metrics(candidates: list[dict[str, Any]], seed_objective: float | None) -> dict[str, Any]:
    proposals = _ordered_proposals(candidates)
    first_legal = None
    first_improving = None
    for index, row in enumerate(proposals, start=1):
        legal = (_float(_nested(row, "metrics", "legal")) or 0.0) > 0.0
        available = (_float(_nested(row, "metrics", "objective_available")) or 0.0) > 0.0
        objective = _float(_nested(row, "metrics", "objective_value"))
        if first_legal is None and legal and available and objective is not None:
            first_legal = (index, row)
        if (
            first_improving is None
            and legal
            and available
            and objective is not None
            and seed_objective is not None
            and objective < seed_objective
        ):
            first_improving = (index, row)
    return {
        "proposal_index_to_first_legal": None if first_legal is None else first_legal[0],
        "time_to_first_legal_s": None if first_legal is None else _float(first_legal[1].get("elapsed_since_run_start_s")),
        "proposal_index_to_first_improvement": None if first_improving is None else first_improving[0],
        "time_to_first_improvement_s": None if first_improving is None else _float(first_improving[1].get("elapsed_since_run_start_s")),
    }


def _aggregate_candidate_rows(candidate_rows: list[dict[str, Any]], run_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_cost = {
        str(row.get("run_instance_id") or row.get("run_id")): _float(row.get("total_recorded_cost_usd"))
        for row in run_summaries
    }

    def build(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in candidate_rows:
            if row.get("role") != "proposal":
                continue
            key = tuple(row.get(name) for name in keys)
            groups.setdefault(key, []).append(row)
        output: list[dict[str, Any]] = []
        for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
            valid = [row for row in rows if (_float(row.get("runs_successfully")) or 0.0) > 0.0]
            legal = [row for row in rows if (_float(row.get("legal")) or 0.0) > 0.0]
            improving = [row for row in rows if int(row.get("improving") or 0) == 1]
            unique = {str(row.get("candidate_sha256")) for row in valid if row.get("candidate_sha256")}
            improvements = [_float(row.get("relative_improvement")) for row in legal]
            improvements = [value for value in improvements if value is not None]
            objectives = [_float(row.get("objective_value")) for row in legal]
            objectives = [value for value in objectives if value is not None]
            run_ids = {str(row.get("run_instance_id") or row.get("run_id")) for row in rows}
            costs = [run_cost.get(run_id) for run_id in run_ids]
            costs = [value for value in costs if value is not None]
            total_cost = sum(costs) if costs else None
            record = {name: value for name, value in zip(keys, key)}
            record.update({
                "runs": len(run_ids),
                "seeds": len({row.get("seed") for row in rows}),
                "proposals": len(rows),
                "valid_proposals": len(valid),
                "unique_candidates": len(unique),
                "legal_candidates": len(legal),
                "improving_candidates": len(improving),
                "valid_rate": len(valid) / len(rows) if rows else None,
                "unique_rate": len(unique) / len(valid) if valid else None,
                "legal_rate": len(legal) / len(rows) if rows else None,
                "improving_rate": len(improving) / len(rows) if rows else None,
                "best_relative_improvement": max(improvements) if improvements else None,
                "unique_legal_objective_values": len(set(objectives)),
                "legal_objective_range": (max(objectives) - min(objectives)) if objectives else None,
                "mean_pipeline_time_s": _mean(_float(row.get("pipeline_wall_time_s")) for row in rows),
                "mean_verify_time_s": _mean(_float(row.get("verify_time_s")) for row in rows),
                "mean_cache_hit_rate": _mean(_float(row.get("cache_hit_rate")) for row in rows),
                "mean_syntactic_legality_reuse_rate": _mean(_float(row.get("syntactic_legality_reuse_rate")) for row in rows),
                "mean_rewrite_legality_reuse_rate": _mean(_float(row.get("rewrite_legality_reuse_rate")) for row in rows),
                "extra_reused_legality_count": sum(int(_float(row.get("extra_reused_legality_count")) or 0) for row in rows),
                "total_recorded_cost_usd": total_cost,
                "cost_per_legal_candidate_usd": (total_cost / len(legal)) if total_cost is not None and legal else None,
                "cost_per_improving_candidate_usd": (total_cost / len(improving)) if total_cost is not None and improving else None,
            })
            output.append(record)
        return output

    return (
        build(("kernel", "search_strategy", "model")),
        build(("search_strategy", "model")),
    )


def _run_summary(manifest: Mapping[str, Any], search_result: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    proposals = _ordered_proposals(candidates)
    seed_objective = _seed_objective(candidates)

    valid = [row for row in proposals if (_float(_nested(row, "metrics", "runs_successfully")) or 0.0) > 0.0]
    legal = [row for row in proposals if (_float(_nested(row, "metrics", "legal")) or 0.0) > 0.0]
    scored = [
        (row, _float(_nested(row, "metrics", "objective_value")))
        for row in legal
        if (_float(_nested(row, "metrics", "objective_available")) or 0.0) > 0.0
    ]
    scored = [(row, value) for row, value in scored if value is not None]
    improving: list[tuple[dict[str, Any], float]] = []
    if seed_objective is not None:
        improving = [(row, value) for row, value in scored if value < seed_objective]
    best_pair = min(scored, key=lambda pair: pair[1]) if scored else None
    best_objective = None if best_pair is None else best_pair[1]
    best_improvement = None
    time_to_best = None
    proposal_index_to_best = None
    evaluation_count_to_best = None
    time_from_first_proposal_to_best = None
    if best_objective is not None and seed_objective not in (None, 0.0):
        best_improvement = (seed_objective - best_objective) / seed_objective
        best_row = best_pair[0]
        time_to_best = _float(best_row.get("elapsed_since_run_start_s"))
        target_sequence = int(best_row.get("evaluation_sequence") or best_row.get("evaluation_index") or 0)
        ordered = sorted(
            candidates,
            key=lambda row: int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0),
        )
        evaluation_count_to_best = next(
            (index for index, row in enumerate(ordered, start=1)
             if int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0) == target_sequence),
            None,
        )
        proposal_index_to_best = next(
            (index for index, row in enumerate(proposals, start=1)
             if int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0) == target_sequence),
            None,
        )
        if proposals:
            first_proposal_time = _float(proposals[0].get("elapsed_since_run_start_s"))
            if time_to_best is not None and first_proposal_time is not None:
                time_from_first_proposal_to_best = max(0.0, time_to_best - first_proposal_time)

    unique = {
        str(_nested(row, "trace", "candidate_sha256") or row.get("candidate_sha256"))
        for row in valid
        if _nested(row, "trace", "candidate_sha256") or row.get("candidate_sha256")
    }
    objectives = [value for _, value in scored]
    cost = dict(search_result.get("cost", {}))
    total_cost = _float(cost.get("total_recorded_cost_usd"))
    raw_metrics = _nested(search_result, "search_result", "metrics", default={}) or {}
    progress = _progress_metrics(candidates, seed_objective)
    failed_with_successor = 0
    failure_to_next_legal = 0
    for previous, following in zip(proposals, proposals[1:]):
        previous_legal = (_float(_nested(previous, "metrics", "legal")) or 0.0) > 0.0
        following_legal = (_float(_nested(following, "metrics", "legal")) or 0.0) > 0.0
        if not previous_legal:
            failed_with_successor += 1
            if following_legal:
                failure_to_next_legal += 1
    reasoning_tokens = _find_numeric(raw_metrics, ("reasoning_tokens", "thoughts_token_count"))
    cache_read_tokens = _find_numeric(raw_metrics, ("cache_read", "cache_read_tokens", "cached_content_token_count"))
    model_turns = _find_numeric(raw_metrics, ("num_turns", "model_turns"))
    tool_calls = _find_numeric(raw_metrics, ("tool_calls", "num_tool_calls"))
    provider_retries = _find_numeric(raw_metrics, ("provider_retries", "retry_count", "retries"))
    api_failures = _find_numeric(raw_metrics, ("api_failures", "api_failure_count", "provider_failures"))
    return {
        "run_id": manifest.get("run_id"),
        "run_instance_id": manifest.get("run_instance_id"),
        "experiment_name": manifest.get("experiment_name"),
        "kernel": manifest.get("kernel"),
        "search_strategy": manifest.get("search_strategy"),
        "base_search_engine": manifest.get("base_search_engine"),
        "feedback_mode": manifest.get("feedback_mode"),
        "seed": manifest.get("seed"),
        "model": manifest.get("model"),
        "provider": manifest.get("provider"),
        "checker_backend": manifest.get("checker_backend"),
        "objective_backend": manifest.get("objective_backend"),
        "proposal_budget": manifest.get("proposal_budget"),
        "search_iteration_count": _nested(search_result, "search_result", "metrics", "search_iteration_count"),
        "search_population_size": _nested(search_result, "search_result", "metrics", "search_population_size"),
        "search_population_size_source": _nested(search_result, "search_result", "metrics", "search_population_size_source"),
        "run_status": manifest.get("status"),
        "run_wall_time_s": manifest.get("wall_time_s"),
        "seed_objective": seed_objective,
        "proposals": len(proposals),
        "valid_proposals": len(valid),
        "unique_candidates": len(unique),
        "legal_candidates": len(legal),
        "improving_candidates": len(improving),
        "valid_rate": (len(valid) / len(proposals)) if proposals else None,
        "unique_rate": (len(unique) / len(valid)) if valid else None,
        "legal_rate": (len(legal) / len(proposals)) if proposals else None,
        "improving_rate": (len(improving) / len(proposals)) if proposals else None,
        "failure_to_next_legal_transitions": failure_to_next_legal,
        "failed_proposals_with_successor": failed_with_successor,
        "failure_to_next_legal_rate": (failure_to_next_legal / failed_with_successor) if failed_with_successor else None,
        "best_objective": best_objective,
        "best_relative_improvement": best_improvement,
        "unique_legal_objective_values": len(set(objectives)),
        "legal_objective_range": (max(objectives) - min(objectives)) if objectives else None,
        "time_to_best_s": time_to_best,
        "proposal_index_to_best": proposal_index_to_best,
        "evaluation_count_to_best": evaluation_count_to_best,
        "time_from_first_proposal_to_best_s": time_from_first_proposal_to_best,
        **progress,
        "mean_pipeline_time_s": _mean(_float(row.get("pipeline_wall_time_s")) for row in proposals),
        "mean_prepare_time_s": _mean(_float(_nested(row, "stage_timings_s", "prepare")) for row in proposals),
        "mean_verify_time_s": _mean(_float(_nested(row, "stage_timings_s", "verify")) for row in proposals),
        "mean_evaluate_time_s": _mean(_float(_nested(row, "stage_timings_s", "evaluate")) for row in proposals),
        "mean_feedback_time_s": _mean(_float(_nested(row, "stage_timings_s", "feedback")) for row in proposals),
        "mean_certificate_validation_time_s": _mean(_float(_nested(row, "evidence", "certificate_validation_time_s")) for row in proposals),
        "mean_certificate_size_bytes": _mean(_float(_nested(row, "evidence", "certificate_size_bytes")) for row in proposals),
        "mean_cache_hit_rate": _mean(
            _float(_nested(row, "evidence", "certificate", "execution", "cache_hit_rate"))
            for row in proposals
        ),
        "input_tokens": cost.get("input_tokens"),
        "output_tokens": cost.get("output_tokens"),
        "total_tokens": cost.get("total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "model_turns": model_turns,
        "tool_calls": tool_calls,
        "provider_retries": provider_retries,
        "api_failures": api_failures,
        "reported_llm_cost_usd": cost.get("reported_llm_cost_usd"),
        "estimated_llm_cost_usd": cost.get("estimated_llm_cost_usd"),
        "total_recorded_cost_usd": total_cost,
        "manual_cloud_credit_used_usd": cost.get("manual_cloud_credit_used_usd"),
        "billing_source": cost.get("billing_source"),
        "pricing_as_of": cost.get("pricing_as_of"),
        "cost_per_legal_candidate_usd": (total_cost / len(legal)) if total_cost is not None and legal else None,
        "cost_per_improving_candidate_usd": (total_cost / len(improving)) if total_cost is not None and improving else None,
    }


def _certification_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("kernel"),
            row.get("search_strategy"),
            row.get("base_search_engine"),
            row.get("feedback_mode"),
            row.get("model"),
            row.get("checker_backend"),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        ok = [row for row in items if row.get("replay_status") == "ok"]
        agreements = [row for row in ok if row.get("agrees_with_full") is not None]
        extra = sum(int(_float(row.get("extra_reused_legality_count")) or 0) for row in ok)
        denom = sum(int(_float(row.get("rewrite_recovery_denominator")) or 0) for row in ok)
        output.append({
            "kernel": key[0],
            "search_strategy": key[1],
            "base_search_engine": key[2],
            "feedback_mode": key[3],
            "model": key[4],
            "checker_backend": key[5],
            "replay_rows": len(items),
            "successful_replays": len(ok),
            "agreement_with_full_rate": (
                sum(1 for row in agreements if row.get("agrees_with_full") is True) / len(agreements)
                if agreements else None
            ),
            "false_accepts_vs_full": sum(1 for row in ok if row.get("false_accept_vs_full") is True),
            "false_rejects_vs_full": sum(1 for row in ok if row.get("false_reject_vs_full") is True),
            "mean_checker_time_s": _mean(_float(row.get("checker_time_s")) for row in ok),
            "mean_replay_e2e_time_s": _mean(_float(row.get("replay_e2e_time_s")) for row in ok),
            "mean_checker_speedup_vs_full": _mean(_float(row.get("checker_speedup_vs_full")) for row in ok),
            "mean_e2e_speedup_vs_full": _mean(_float(row.get("e2e_speedup_vs_full")) for row in ok),
            "mean_cache_hit_rate": _mean(_float(row.get("cache_hit_rate")) for row in ok),
            "cache_hits": sum(int(_float(row.get("cache_hits")) or 0) for row in ok),
            "cache_misses": sum(int(_float(row.get("cache_misses")) or 0) for row in ok),
            "stage_hits": sum(int(_float(row.get("stage_hit_count")) or 0) for row in ok),
            "stage_misses": sum(int(_float(row.get("stage_miss_count")) or 0) for row in ok),
            "interface_hits": sum(int(_float(row.get("interface_hit_count")) or 0) for row in ok),
            "interface_misses": sum(int(_float(row.get("interface_miss_count")) or 0) for row in ok),
            "legality_hits": sum(int(_float(row.get("legality_hit_count")) or 0) for row in ok),
            "legality_misses": sum(int(_float(row.get("legality_miss_count")) or 0) for row in ok),
            "mean_syntactic_legality_reuse_rate": _mean(_float(row.get("syntactic_legality_reuse_rate")) for row in ok),
            "mean_rewrite_legality_reuse_rate": _mean(_float(row.get("rewrite_legality_reuse_rate")) for row in ok),
            "rewrite_only_additional_obligations_reused": extra,
            "rewrite_recovery_denominator": denom,
            "rewrite_recovered_fraction_of_syntactic_misses": extra / denom if denom else 0.0,
            "incremental_fallbacks": sum(1 for row in ok if row.get("incremental_fallback_reason")),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="results/teaal_summary")
    args = parser.parse_args()

    runs_root = Path(args.runs).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifests = sorted(runs_root.rglob("run_manifest.json"))
    if not manifests:
        raise SystemExit(f"no run_manifest.json files found under {runs_root}")

    all_candidates: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_replay: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        run_dir = manifest_path.parent
        candidate_path = Path(str(manifest.get("result_log") or run_dir / "trusted_results.jsonl"))
        if not candidate_path.is_absolute():
            candidate_path = run_dir / candidate_path
        if not candidate_path.exists():
            candidate_path = run_dir / "trusted_results.jsonl"
        candidates = _read_jsonl(candidate_path)
        search_path = run_dir / "search_result.json"
        search_result = _read_json(search_path) if search_path.exists() else {}
        annotation_path = run_dir / "cost_annotation.json"
        if annotation_path.exists():
            annotation = _read_json(annotation_path)
            cost = dict(search_result.get("cost", {}))
            cost.update({key: value for key, value in annotation.items() if key not in {"schema_version", "source", "annotated_at_utc"}})
            search_result = {**search_result, "cost": cost}
        summaries.append(_run_summary(manifest, search_result, candidates))
        seed_objective = _seed_objective(candidates)
        for row in candidates:
            all_candidates.append(_candidate_flat(row, seed_objective))
            all_failures.extend(_failure_rows(row))
        all_replay.extend(_read_jsonl(run_dir / "certification_replay.jsonl"))

    aggregate, model_summary = _aggregate_candidate_rows(all_candidates, summaries)
    cert_summary = _certification_summary(all_replay)
    _write_csv(output / "run_summary.csv", summaries)
    _write_csv(output / "aggregate_summary.csv", aggregate)
    _write_csv(output / "model_summary.csv", model_summary)
    _write_csv(output / "candidate_evaluations.csv", all_candidates)
    _write_csv(output / "failure_taxonomy.csv", all_failures)
    _write_csv(output / "certification_replay.csv", all_replay)
    _write_csv(output / "certification_summary.csv", cert_summary)
    _write_csv(
        output / "cost_usage_summary.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "run_id", "run_instance_id", "kernel", "search_strategy", "seed",
                    "model", "provider", "run_wall_time_s", "input_tokens",
                    "output_tokens", "reasoning_tokens", "cache_read_tokens", "total_tokens",
                    "model_turns", "tool_calls", "provider_retries", "api_failures",
                    "reported_llm_cost_usd", "estimated_llm_cost_usd", "total_recorded_cost_usd",
                    "manual_cloud_credit_used_usd", "billing_source", "pricing_as_of",
                    "legal_candidates", "improving_candidates", "cost_per_legal_candidate_usd",
                    "cost_per_improving_candidate_usd",
                )
            }
            for row in summaries
        ],
    )
    print(f"runs: {len(summaries)}")
    print(f"candidate evaluations: {len(all_candidates)}")
    print(f"certification replay rows: {len(all_replay)}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
