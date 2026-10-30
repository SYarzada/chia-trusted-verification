from __future__ import annotations

from dataclasses import asdict
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import importlib.metadata
import platform
import sys
import threading
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
import time
from typing import Any, Mapping

from .config import ExperimentConfig, base_search_engine, search_feedback_mode
from .types import PipelineResult, SearchProblem, SearchResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def append_jsonl(path: str | Path, value: Any, lock: threading.Lock | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n"

    def _append() -> None:
        with target.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    if lock is None:
        _append()
        return
    with lock:
        _append()



def append_jsonl_sequenced(
    path: str | Path,
    value: Mapping[str, Any],
    lock: threading.Lock | None = None,
) -> dict[str, Any]:
    """Append a JSONL record with a unique sequence number across processes.

    A file lock keeps sequence numbers unique for parallel workers.
    Both evaluation_sequence and evaluation_index use the global number.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    sequence_path = target.with_suffix(target.suffix + ".seq")

    def _append() -> dict[str, Any]:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                sequence = 0
                if sequence_path.exists():
                    raw = sequence_path.read_text(encoding="utf-8").strip()
                    if raw:
                        sequence = int(raw)
                sequence += 1
                sequence_path.write_text(str(sequence) + "\n", encoding="utf-8")

                record = dict(value)
                record["evaluation_sequence"] = sequence
                record["evaluation_index"] = sequence
                line = json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n"
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                return record
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    if lock is None:
        return _append()
    with lock:
        return _append()


def persist_candidate_source(result_log: str | Path, source: str) -> tuple[str, str]:
    """Save candidate source beside the result log and return its path and SHA256."""
    log_path = Path(result_log).resolve()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    directory = log_path.parent / "candidate_programs"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.py"
    if not target.exists():
        temporary = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(source, encoding="utf-8")
        try:
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    try:
        relative = target.relative_to(log_path.parent)
        return str(relative), digest
    except ValueError:
        return str(target), digest


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("chia", "skydiscover", "openevolve", "alpha-evolve"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }

def build_run_metadata(config: ExperimentConfig, problem: SearchProblem) -> dict[str, Any]:
    backend = dict(config.backend_settings)
    search_settings = _read_yaml_mapping(config.search_config)
    model = _effective_model(config.engine, config.model, search_settings)
    provider = _provider_name(config.engine, search_settings)
    if base_search_engine(config.engine) == "openevolve" and model and "/" in model:
        provider = model.split("/", 1)[0]
    return {
        "experiment_name": config.experiment_name,
        "run_id": config.run_id,
        "backend": config.backend_name,
        "kernel": backend.get("kernel_label"),
        "search_strategy": config.engine,
        "base_search_engine": base_search_engine(config.engine),
        "feedback_mode": search_feedback_mode(config.engine),
        "seed": config.seed,
        "model": model,
        "provider": provider,
        "proposal_budget": config.iterations,
        "checker_backend": backend.get("checker_backend"),
        "objective_backend": backend.get("objective_backend"),
        "knob_space": backend.get("knob_space"),
        "min_mutations": backend.get("min_mutations"),
        "max_mutations": backend.get("max_mutations"),
        "catalog_sha256": problem.metadata.get("catalog_sha256"),
        "catalog_size": problem.metadata.get("catalog_size"),
        "intent_id": config.intent.intent_id,
        "artifact_type": config.intent.artifact_type,
        "output_dir": config.output_dir,
        "result_log": config.result_log,
        "search_config": config.search_config,
        "cost_tracking": dict(config.cost_tracking),
    }



def _effective_model(
    engine: str,
    override: str | None,
    settings: Mapping[str, Any],
) -> str | None:
    if override:
        return str(override)
    base_engine = base_search_engine(engine)
    if base_engine == "llm_agent":
        agent = settings.get("agent", {})
        if isinstance(agent, Mapping) and agent.get("model"):
            return str(agent["model"])
    if base_engine == "alphaevolve":
        alpha = settings.get("alphaevolve", {})
        if isinstance(alpha, Mapping):
            generation = alpha.get("generation_settings", {})
            if isinstance(generation, Mapping):
                models = generation.get("models", [])
                if isinstance(models, list) and models and isinstance(models[0], Mapping):
                    name = models[0].get("name")
                    if name:
                        return str(name)
    llm = settings.get("llm", {})
    if isinstance(llm, Mapping):
        models = llm.get("models", [])
        if isinstance(models, list) and models and isinstance(models[0], Mapping):
            name = models[0].get("name")
            if name:
                return str(name)
    return None

def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        return {}


def _provider_name(engine: str, settings: Mapping[str, Any]) -> str | None:
    base_engine = base_search_engine(engine)
    if base_engine == "llm_agent":
        agent = settings.get("agent", {})
        if isinstance(agent, Mapping):
            value = agent.get("provider")
            return None if value is None else str(value)
    llm = settings.get("llm", {})
    if isinstance(llm, Mapping):
        models = llm.get("models", [])
        if isinstance(models, list) and models and isinstance(models[0], Mapping):
            name = models[0].get("name")
            if name:
                return str(name).split("/", 1)[0]
    if base_engine == "alphaevolve":
        return "alphaevolve"
    return None


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


def summarize_usage(raw_metrics: Any, cost_tracking: Mapping[str, Any]) -> dict[str, Any]:
    input_tokens = _find_numeric(
        raw_metrics,
        ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count"),
    )
    output_tokens = _find_numeric(
        raw_metrics,
        ("output_tokens", "completion_tokens", "output_token_count", "completion_token_count"),
    )
    total_tokens = _find_numeric(raw_metrics, ("total_tokens", "total_token_count"))
    reported_cost = _find_numeric(
        raw_metrics,
        ("cost_usd", "total_cost_usd", "llm_cost_usd", "api_cost_usd"),
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    input_rate = _optional_float(cost_tracking.get("input_per_million_tokens_usd"))
    output_rate = _optional_float(cost_tracking.get("output_per_million_tokens_usd"))
    estimated_llm = None
    if input_tokens is not None and output_tokens is not None and input_rate is not None and output_rate is not None:
        estimated_llm = (
            input_tokens / 1_000_000.0 * input_rate
            + output_tokens / 1_000_000.0 * output_rate
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reported_llm_cost_usd": reported_cost,
        "estimated_llm_cost_usd": estimated_llm,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def build_cost_summary(
    *,
    raw_metrics: Any,
    cost_tracking: Mapping[str, Any],
    wall_time_s: float,
) -> dict[str, Any]:
    usage = summarize_usage(raw_metrics, cost_tracking)
    compute_hourly = _optional_float(cost_tracking.get("compute_hourly_usd"))
    estimated_compute = None if compute_hourly is None else wall_time_s / 3600.0 * compute_hourly
    manual_compute = _optional_float(cost_tracking.get("manual_compute_cost_usd"))
    manual_storage = _optional_float(cost_tracking.get("manual_storage_cost_usd"))
    manual_other = _optional_float(cost_tracking.get("manual_other_cost_usd"))
    cloud_credit_used = _optional_float(cost_tracking.get("manual_cloud_credit_used_usd"))

    llm_cost = usage["reported_llm_cost_usd"]
    llm_source = "reported"
    manual_llm = _optional_float(cost_tracking.get("manual_llm_cost_usd"))
    if llm_cost is None and manual_llm is not None:
        llm_cost = manual_llm
        llm_source = "manual"
    if llm_cost is None:
        llm_cost = usage["estimated_llm_cost_usd"]
        llm_source = "estimated_from_tokens" if llm_cost is not None else "unavailable"

    components = [
        item
        for item in (
            llm_cost,
            manual_compute if manual_compute is not None else estimated_compute,
            manual_storage,
            manual_other,
        )
        if item is not None
    ]
    total = sum(components) if components else None
    return {
        **usage,
        "llm_cost_source": llm_source,
        "manual_llm_cost_usd": manual_llm,
        "estimated_compute_cost_usd": estimated_compute,
        "manual_compute_cost_usd": manual_compute,
        "manual_storage_cost_usd": manual_storage,
        "manual_other_cost_usd": manual_other,
        "manual_cloud_credit_used_usd": cloud_credit_used,
        # Report cloud credits separately from costs.
        "total_recorded_cost_usd": total,
        "currency": str(cost_tracking.get("currency", "USD")),
        "billing_source": cost_tracking.get("billing_source"),
        "pricing_as_of": cost_tracking.get("pricing_as_of"),
        "note": cost_tracking.get("note"),
    }



def _read_candidate_records(path: str | Path, run_instance_id: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record_type") != "candidate_evaluation":
            continue
        if str(row.get("run_instance_id", "")) != str(run_instance_id):
            continue
        rows.append(dict(row))
    rows.sort(
        key=lambda row: (
            int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0),
            str(row.get("timestamp_utc", "")),
        )
    )
    return rows


def _candidate_metric(row: Mapping[str, Any], name: str) -> float | None:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    return _optional_float(metrics.get(name))


def _candidate_source_from_record(row: Mapping[str, Any], result_log: str | Path) -> str | None:
    value = row.get("candidate_source_file")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(result_log).resolve().parent / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def reconcile_search_result(
    result: SearchResult,
    *,
    result_log: str | Path,
    run_instance_id: str,
) -> SearchResult:
    """Select the best candidate using trusted evaluation results.

    Use the trusted log instead of relying on the search engine's final result.
    """

    rows = _read_candidate_records(result_log, run_instance_id)
    if not rows:
        return result

    proposals = [row for row in rows if row.get("role") == "proposal"]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "pass":
            continue
        legal = _candidate_metric(row, "legal")
        available = _candidate_metric(row, "objective_available")
        score = _candidate_metric(row, "combined_score")
        objective = _candidate_metric(row, "objective_value")
        if legal != 1.0 or available != 1.0 or score is None or objective is None:
            continue
        eligible.append(row)

    metrics = dict(result.metrics)
    metrics["search_total_candidate_evaluations"] = len(rows)
    metrics["search_proposal_evaluations"] = len(proposals)
    metrics["engine_reported_best_score"] = float(result.best_score)
    if result.best_solution:
        metrics["engine_reported_best_source_sha256"] = hashlib.sha256(
            result.best_solution.encode("utf-8")
        ).hexdigest()

    if not eligible:
        metrics["best_trusted_available"] = False
        return SearchResult(
            engine=result.engine,
            best_score=result.best_score,
            best_solution=result.best_solution,
            metrics=metrics,
            output_dir=result.output_dir,
        )

    def rank_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        score = _candidate_metric(row, "combined_score") or float("-inf")
        objective = _candidate_metric(row, "objective_value") or float("inf")
        seq = int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0)
        return score, -objective, -seq

    best = max(eligible, key=rank_key)
    best_seq = int(best.get("evaluation_sequence") or best.get("evaluation_index") or 0)
    best_score = float(_candidate_metric(best, "combined_score") or 0.0)
    best_objective = float(_candidate_metric(best, "objective_value") or 0.0)
    best_source = _candidate_source_from_record(best, result_log)

    proposal_index = None
    if best.get("role") == "proposal":
        proposal_index = 0
        for row in proposals:
            proposal_index += 1
            if (
                int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0)
                == best_seq
            ):
                break

    evaluation_count_to_best = next(
        (index for index, row in enumerate(rows, start=1)
         if int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0) == best_seq),
        None,
    )
    elapsed = _optional_float(best.get("elapsed_since_run_start_s"))
    first_proposal_elapsed = (
        _optional_float(proposals[0].get("elapsed_since_run_start_s")) if proposals else None
    )
    from_first = None
    if elapsed is not None and first_proposal_elapsed is not None:
        from_first = max(0.0, elapsed - first_proposal_elapsed)

    first_legal = next(
        (row for row in proposals
         if row.get("status") == "pass"
         and _candidate_metric(row, "legal") == 1.0
         and _candidate_metric(row, "objective_available") == 1.0),
        None,
    )
    first_legal_index = None
    if first_legal is not None:
        target_seq = int(first_legal.get("evaluation_sequence") or first_legal.get("evaluation_index") or 0)
        first_legal_index = next(
            (index for index, row in enumerate(proposals, start=1)
             if int(row.get("evaluation_sequence") or row.get("evaluation_index") or 0) == target_seq),
            None,
        )

    metrics.update(
        {
            "best_trusted_available": True,
            "best_trusted_score": best_score,
            "best_trusted_objective_value": best_objective,
            "best_trusted_evaluation_sequence": best_seq,
            "best_trusted_evaluation_count": evaluation_count_to_best,
            "best_trusted_proposal_index": proposal_index,
            "best_trusted_time_to_best_s": elapsed,
            "best_trusted_time_from_first_proposal_s": from_first,
            "best_trusted_candidate_id": best.get("candidate_id"),
            "best_trusted_candidate_source_sha256": best.get("candidate_source_sha256"),
            "best_trusted_mutation_ids": list(best.get("mutation_ids") or []),
            "first_legal_proposal_index": first_legal_index,
            "first_legal_time_s": (
                _optional_float(first_legal.get("elapsed_since_run_start_s"))
                if first_legal is not None
                else None
            ),
        }
    )
    trusted_source_sha = best.get("candidate_source_sha256")
    engine_source_sha = metrics.get("engine_reported_best_source_sha256")
    metrics["engine_reported_best_matches_trusted_best"] = bool(
        trusted_source_sha and engine_source_sha and trusted_source_sha == engine_source_sha
    )

    return SearchResult(
        engine=result.engine,
        best_score=best_score,
        best_solution=best_source if best_source is not None else result.best_solution,
        metrics=metrics,
        output_dir=result.output_dir,
    )


class RunRecorder:
    def __init__(self, config: ExperimentConfig, problem: SearchProblem) -> None:
        self.config = config
        self.problem = problem
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.search_result_path = self.output_dir / "search_result.json"
        self.metadata = build_run_metadata(config, problem)
        self.metadata["run_instance_id"] = f"run_{uuid.uuid4().hex}"
        self.started_at = utc_now()
        self.started_epoch = time.time()
        self.started_perf = time.perf_counter()
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            **self.metadata,
            "status": "running",
            "started_at_utc": self.started_at,
            "config_snapshot": config.to_dict(),
            "search_problem": {
                "file_name": problem.file_name,
                "metadata": dict(problem.metadata),
                "guidance": problem.guidance.to_dict(),
            },
            "search_config_snapshot": _read_yaml_mapping(config.search_config),
            "environment": environment_metadata(),
        }
        write_json(self.manifest_path, self._manifest)

    def finish(self, result: SearchResult) -> None:
        wall_time_s = time.perf_counter() - self.started_perf
        cost = build_cost_summary(
            raw_metrics=result.metrics,
            cost_tracking=self.config.cost_tracking,
            wall_time_s=wall_time_s,
        )
        search_record = {
            "schema_version": 1,
            **self.metadata,
            "completed_at_utc": utc_now(),
            "wall_time_s": wall_time_s,
            "search_result": result.to_dict(),
            "cost": cost,
        }
        write_json(self.search_result_path, search_record)
        self._manifest.update(
            {
                "status": "completed",
                "completed_at_utc": search_record["completed_at_utc"],
                "wall_time_s": wall_time_s,
                "cost": cost,
                "search_result_file": str(self.search_result_path),
            }
        )
        write_json(self.manifest_path, self._manifest)

    def fail(self, exc: BaseException) -> None:
        wall_time_s = time.perf_counter() - self.started_perf
        self._manifest.update(
            {
                "status": "error",
                "completed_at_utc": utc_now(),
                "wall_time_s": wall_time_s,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        write_json(self.manifest_path, self._manifest)

    def evaluator_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "run_started_at_utc": self.started_at,
            "run_started_epoch": self.started_epoch,
        }


def candidate_record(
    *,
    run_metadata: Mapping[str, Any],
    evaluation_index: int,
    role: str,
    evaluation_origin: str,
    result: PipelineResult,
    pipeline_wall_time_s: float,
    elapsed_since_run_start_s: float | None,
    candidate_source_file: str | None = None,
    candidate_source_sha256: str | None = None,
) -> dict[str, Any]:
    trace = dict(result.trace)
    proposal = dict(trace.get("proposal", {}) or {})
    mutation_ids = list(
        proposal.get("mutation_ids")
        or result.feedback.get("mutation_ids", [])
        or []
    )
    candidate_sha256 = trace.get("candidate_sha256")
    evaluation_id = f"eval_{uuid.uuid4().hex}"
    candidate_id = (
        f"cand_{str(candidate_sha256)[:12]}"
        if candidate_sha256
        else f"cand_{evaluation_id[5:17]}"
    )
    display_label = "seed" if not mutation_ids else candidate_id
    return {
        "record_type": "candidate_evaluation",
        "schema_version": 2,
        "timestamp_utc": utc_now(),
        **dict(run_metadata),
        # Keep the local index for debugging. Sequenced logging assigns a global index.
        "evaluation_index": evaluation_index,
        "evaluation_local_index": evaluation_index,
        "evaluation_id": evaluation_id,
        "candidate_id": candidate_id,
        "candidate_label": display_label,
        "proposal_label": proposal.get("label"),
        "mutation_ids": mutation_ids,
        "role": role,
        "evaluation_origin": evaluation_origin,
        "pipeline_wall_time_s": pipeline_wall_time_s,
        "elapsed_since_run_start_s": elapsed_since_run_start_s,
        "candidate_source_file": candidate_source_file,
        "candidate_source_sha256": candidate_source_sha256,
        **result.to_dict(),
        "candidate_sha256": candidate_sha256,
        "catalog_sha256": trace.get("catalog_sha256"),
    }
