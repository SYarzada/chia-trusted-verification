from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from .backend import TeaalEvolveBackend
from .worker import _build_catalog, _candidate_dict, _candidate_sha, _load_api, _seed_candidate


CERTIFICATION_BACKENDS = (
    "full",
    "hierarchical",
    "hierarchical_delta_syntactic",
    "hierarchical_delta_egraph",
)


def _rebase_archived_backend_settings(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Update saved file paths for the current checkout.

    Keep paths that still exist. Replace missing paths with local files.
    """
    rebased = dict(settings)
    repo_root = Path(__file__).resolve().parents[4]
    local_config_dir = repo_root / "configs"

    archived_config_dir = Path(str(rebased.get("_config_dir", ""))).expanduser()
    if not archived_config_dir.is_dir():
        rebased["_config_dir"] = str(local_config_dir)

    def rebase_path_field(key: str, fallback: Path | None = None) -> None:
        raw = rebased.get(key)
        if raw is None:
            return
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            return
        if path.exists():
            return
        parts = path.parts
        if "teaal_evolve" in parts:
            index = parts.index("teaal_evolve")
            candidate = repo_root.joinpath(*parts[index:])
            if candidate.exists():
                rebased[key] = str(candidate)
                return
        if fallback is not None and fallback.exists():
            rebased[key] = str(fallback)

    rebase_path_field("root", repo_root / "teaal_evolve")
    rebase_path_field("kernel_file")
    rebase_path_field("architecture_file")
    return rebased


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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _timed_check(api: Mapping[str, Any], candidate: Any, **kwargs: Any):
    started = time.perf_counter()
    result = api["check_candidate_with_backend"](candidate, **kwargs)
    return result, time.perf_counter() - started


def _timed_objective(api: Mapping[str, Any], candidate: Any, settings: Mapping[str, Any]):
    started = time.perf_counter()
    result = api["compute_objective_with_backend"](
        candidate,
        backend_name=str(settings.get("objective_backend", "monolithic_bottleneck")),
        shapes=dict(settings.get("objective_shapes", {})) or None,
    )
    elapsed = time.perf_counter() - started
    value = None if result.value is None else float(result.value)
    return value, str(result.status), elapsed


def replay_run(run_dir: str | Path, *, force: bool = False) -> Path:
    """Check saved proposals with all four certification methods.

    Each proposal starts from the legal seed and uses its saved mutation IDs.
    This does not rerun the model or search.
    """

    run_path = Path(run_dir).expanduser().resolve()
    output = run_path / "certification_replay.jsonl"
    if output.exists() and output.stat().st_size > 0 and not force:
        return output

    manifest = _read_json(run_path / "run_manifest.json")
    snapshot = dict(manifest.get("config_snapshot", {}))
    backend_settings = dict(snapshot.get("backend_settings", {}))
    if not backend_settings:
        raise ValueError(f"run manifest has no backend_settings: {run_path}")

    backend_settings = _rebase_archived_backend_settings(backend_settings)
    worker_settings = TeaalEvolveBackend()._worker_settings(backend_settings)
    api = _load_api(worker_settings)
    catalog = _build_catalog(api, worker_settings)
    by_id = {str(row["mutation_id"]): row for row in catalog}
    seed_candidate = _seed_candidate(api, worker_settings)
    seed_payload = _candidate_dict(seed_candidate)
    seed_sha = _candidate_sha(seed_payload)
    seed_hier = api["check_candidate_with_backend"](
        seed_candidate,
        backend_name="hierarchical",
        include_bottleneck_objective=False,
        knob_space=str(worker_settings.get("knob_space", "rich")),
    )
    seed_report = seed_hier.candidate_report

    result_log = Path(str(manifest.get("result_log") or run_path / "trusted_results.jsonl"))
    if not result_log.is_absolute():
        result_log = run_path / result_log
    if not result_log.exists():
        result_log = run_path / "trusted_results.jsonl"
    candidates = _read_jsonl(result_log)
    proposals = [row for row in candidates if row.get("role") == "proposal"]
    proposals.sort(
        key=lambda row: int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0)
    )

    replay_rows: list[dict[str, Any]] = []
    for proposal_index, source_row in enumerate(proposals, start=1):
        mutation_ids = [str(item) for item in (source_row.get("mutation_ids") or [])]
        common = {
            "record_type": "certification_replay",
            "schema_version": 1,
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
            "proposal_index": proposal_index,
            "source_evaluation_sequence": source_row.get("evaluation_sequence", source_row.get("evaluation_index")),
            "source_evaluation_id": source_row.get("evaluation_id"),
            "candidate_id": source_row.get("candidate_id"),
            "logged_candidate_sha256": source_row.get("candidate_sha256"),
            "mutation_ids": mutation_ids,
            "base_candidate_sha256": seed_sha,
            "delta_base": "fixed_legal_seed",
        }

        if not source_row.get("candidate_sha256"):
            for backend_name in CERTIFICATION_BACKENDS:
                replay_rows.append(
                    {
                        **common,
                        "checker_backend": backend_name,
                        "replay_status": "unavailable",
                        "replay_error_type": "ProposalNotMaterialized",
                        "replay_error": "trusted proposal did not materialize a TeAAL candidate",
                    }
                )
            continue

        try:
            selected = [by_id[mutation_id] for mutation_id in mutation_ids]
            if len(selected) != len(mutation_ids):
                raise ValueError("proposal contains mutation IDs absent from the archived catalog")
            materialize_started = time.perf_counter()
            candidate = (
                api["apply_knob_specs_atomic"](
                    seed_candidate,
                    [str(row["knob_spec"]) for row in selected],
                    label=f"replay-{proposal_index}",
                    knob_space=str(worker_settings.get("knob_space", "rich")),
                )
                if selected
                else seed_candidate.clone(label=f"replay-{proposal_index}")
            )
            materialize_time_s = time.perf_counter() - materialize_started
            candidate_sha = _candidate_sha(_candidate_dict(candidate))
            trusted_delta = None
            if mutation_ids:
                from src.trans.certify.incremental_pipeline import verify_trusted_knob_delta

                trusted_delta = verify_trusted_knob_delta(
                    seed_candidate,
                    candidate,
                    [str(row["knob_spec"]) for row in selected],
                    knob_space=str(worker_settings.get("knob_space", "rich")),
                    source="chia_archived_mutation_ids",
                    base_candidate_fingerprint=seed_sha,
                    mutated_candidate_fingerprint=candidate_sha,
                    base_report=seed_report,
                )
        except Exception as exc:
            for backend_name in CERTIFICATION_BACKENDS:
                replay_rows.append(
                    {
                        **common,
                        "checker_backend": backend_name,
                        "replay_status": "unavailable",
                        "replay_error_type": type(exc).__name__,
                        "replay_error": str(exc),
                    }
                )
            continue

        results: dict[str, dict[str, Any]] = {}
        for backend_name in CERTIFICATION_BACKENDS:
            kwargs: dict[str, Any] = {
                "backend_name": backend_name,
                "include_bottleneck_objective": False,
                "knob_space": str(worker_settings.get("knob_space", "rich")),
            }
            if backend_name.startswith("hierarchical_delta_"):
                kwargs.update(
                    {
                        "base_candidate": seed_candidate,
                        "base_report": seed_report,
                        "trusted_knob_delta": trusted_delta,
                    }
                )
            try:
                checked, checker_time_s = _timed_check(api, candidate, **kwargs)
                objective_value = None
                objective_status = "skipped_illegal"
                objective_time_s = 0.0
                if bool(checked.candidate_report.is_legal):
                    objective_value, objective_status, objective_time_s = _timed_objective(
                        api, candidate, worker_settings
                    )
                extra = len(checked.extra_reused_legality_ids)
                remaining_rewrite_misses = len(checked.legality_miss_ids)
                recovered_denominator = extra + remaining_rewrite_misses
                results[backend_name] = {
                    **common,
                    "checker_backend": backend_name,
                    "replay_status": "ok",
                    "candidate_sha256": candidate_sha,
                    "candidate_sha_matches_log": bool(
                        not source_row.get("candidate_sha256")
                        or str(source_row.get("candidate_sha256")) == candidate_sha
                    ),
                    "legal": bool(checked.candidate_report.is_legal),
                    "checker_time_s": checker_time_s,
                    "materialize_time_s": materialize_time_s,
                    "objective_time_s": objective_time_s,
                    "replay_e2e_time_s": materialize_time_s + checker_time_s + objective_time_s,
                    "objective_status": objective_status,
                    "objective_value": objective_value,
                    "execution_mode": checked.execution_mode,
                    "cache_hits": checked.total_hits,
                    "cache_misses": checked.total_misses,
                    "total_obligations_processed": checked.total_obligations_processed,
                    "cache_hit_rate": checked.hit_rate,
                    "stage_hit_count": len(checked.stage_hit_ids),
                    "stage_miss_count": len(checked.stage_miss_ids),
                    "interface_hit_count": len(checked.interface_hit_ids),
                    "interface_miss_count": len(checked.interface_miss_ids),
                    "legality_hit_count": len(checked.legality_hit_ids),
                    "legality_miss_count": len(checked.legality_miss_ids),
                    "syntactic_legality_reuse_rate": checked.syntactic_legality_reuse_rate,
                    "rewrite_legality_reuse_rate": checked.rewrite_legality_reuse_rate,
                    "extra_reused_legality_count": extra,
                    "rewrite_recovery_denominator": recovered_denominator,
                    "rewrite_recovered_fraction_of_syntactic_misses": (
                        extra / recovered_denominator if recovered_denominator else 0.0
                    ),
                    "incremental_fallback_reason": checked.incremental_fallback_reason,
                    "incremental_delta_source": checked.incremental_delta_source,
                }
            except Exception as exc:
                results[backend_name] = {
                    **common,
                    "checker_backend": backend_name,
                    "replay_status": "error",
                    "candidate_sha256": candidate_sha,
                    "replay_error_type": type(exc).__name__,
                    "replay_error": str(exc),
                }

        full = results.get("full", {})
        full_legal = full.get("legal") if full.get("replay_status") == "ok" else None
        full_checker_time = full.get("checker_time_s")
        full_e2e_time = full.get("replay_e2e_time_s")
        for backend_name in CERTIFICATION_BACKENDS:
            row = results[backend_name]
            if row.get("replay_status") == "ok" and full_legal is not None:
                legal = bool(row.get("legal"))
                row["agrees_with_full"] = legal == bool(full_legal)
                row["false_accept_vs_full"] = bool(legal and not bool(full_legal))
                row["false_reject_vs_full"] = bool((not legal) and bool(full_legal))
                row["checker_speedup_vs_full"] = (
                    float(full_checker_time) / float(row["checker_time_s"])
                    if full_checker_time is not None and float(row["checker_time_s"]) > 0
                    else None
                )
                row["e2e_speedup_vs_full"] = (
                    float(full_e2e_time) / float(row["replay_e2e_time_s"])
                    if full_e2e_time is not None and float(row["replay_e2e_time_s"]) > 0
                    else None
                )
            replay_rows.append(row)

    _write_jsonl(output, replay_rows)
    return output
