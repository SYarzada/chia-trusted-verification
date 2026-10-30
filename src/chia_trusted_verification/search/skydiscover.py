from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib.util
import inspect
import json
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.config import ExperimentConfig, base_search_engine
from chia_trusted_verification.types import SearchProblem, SearchResult

from .base import SearchStrategy


_VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_VERTEX_AUTH_LOCK = threading.Lock()
_VERTEX_CREDENTIALS: Any = None
_VERTEX_CREDENTIALS_PROJECT: str | None = None


def vertex_adc_access_token() -> str:
    global _VERTEX_CREDENTIALS, _VERTEX_CREDENTIALS_PROJECT

    from google.auth import default
    from google.auth.transport.requests import Request

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or None
    with _VERTEX_AUTH_LOCK:
        if _VERTEX_CREDENTIALS is None or _VERTEX_CREDENTIALS_PROJECT != project:
            credentials, discovered_project = default(scopes=[_VERTEX_SCOPE])
            effective_project = project or discovered_project
            if effective_project and hasattr(credentials, "with_quota_project"):
                credentials = credentials.with_quota_project(effective_project)
            _VERTEX_CREDENTIALS = credentials
            _VERTEX_CREDENTIALS_PROJECT = project
        if not _VERTEX_CREDENTIALS.valid:
            _VERTEX_CREDENTIALS.refresh(Request())
        token = _VERTEX_CREDENTIALS.token
    if not token:
        raise RuntimeError("Google ADC did not produce a Vertex AI access token")
    return str(token)


def _vertex_openai_base_url(project: str, location: str) -> str:
    project = project.strip()
    location = location.strip() or "global"
    if not project:
        raise RuntimeError("OpenEvolve on Vertex AI requires GOOGLE_CLOUD_PROJECT")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def _vertex_model_name(model: str) -> str:
    value = str(model).strip()
    if value.startswith("google-vertex/"):
        value = value.removeprefix("google-vertex/")
    if value.startswith("gemini/"):
        value = value.removeprefix("gemini/")
    if value.startswith("google/"):
        return value
    if value.startswith("gemini-"):
        return f"google/{value}"
    raise ValueError(
        "OpenEvolve's Vertex ADC path expects a Gemini model such as "
        "google/gemini-2.5-flash"
    )


def _openevolve_vertex_endpoint() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("VERTEX_LOCATION")
        or "global"
    ).strip()
    return _vertex_openai_base_url(project, location)


def _configure_openevolve_vertex_adc(config: Any) -> str:
    endpoint = _openevolve_vertex_endpoint()
    llm = getattr(config, "llm", None)
    if llm is None:
        raise RuntimeError("SkyDiscover config does not expose an llm section")
    setattr(llm, "api_base", endpoint)
    if hasattr(llm, "api_key"):
        llm.api_key = None

    collections = []
    for name in ("models", "evaluator_models", "guide_models"):
        models = getattr(llm, name, None)
        if models:
            collections.append(models)
    if not collections:
        raise RuntimeError("OpenEvolve config does not define any LLM models")

    seen: set[int] = set()
    for models in collections:
        for model in models:
            if id(model) in seen:
                continue
            seen.add(id(model))
            model.name = _vertex_model_name(model.name)
            model.api_base = endpoint
            model.api_key = None
    return endpoint


def _configure_openevolve_process_pool(config: Any) -> None:
    # Set process spawning directly to avoid hangs after Ray starts.
    # SkyDiscover does not forward this setting automatically.
    config.max_tasks_per_child = 1


def _apply_vertex_adc_to_openevolve_config(config: Any, endpoint: str) -> None:
    llm = getattr(config, "llm", None)
    if llm is None:
        raise RuntimeError("OpenEvolve config does not expose an llm section")

    # Spawned workers need a serializable token, not a callable provider.
    # Refresh the ADC token and save it in both model configurations.
    token = vertex_adc_access_token()
    llm.api_base = endpoint
    llm.api_key = token

    collections = []
    for name in ("models", "evaluator_models", "guide_models"):
        models = getattr(llm, name, None)
        if models:
            collections.append(models)
    if not collections:
        raise RuntimeError("OpenEvolve config does not define any LLM models")

    seen: set[int] = set()
    for models in collections:
        for model in models:
            if id(model) in seen:
                continue
            seen.add(id(model))
            model.name = _vertex_model_name(model.name)
            model.api_base = endpoint
            model.api_key = token


@contextmanager
def _openevolve_vertex_environment(endpoint: str):
    names = ("OPENAI_API_BASE", "OPENAI_BASE_URL")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = endpoint
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value



def _trusted_evaluator_spawn_spec(
    evaluator: TrustedVerificationEvaluator,
) -> dict[str, Any]:
    """Return evaluator settings for spawned workers."""
    return {
        "backend_name": evaluator.backend_name,
        "backend_settings": evaluator.backend_settings,
        "backend_resources": evaluator.backend_resources,
        "intent": evaluator.intent.to_dict(),
        "engine": evaluator.engine,
        "initial_source": evaluator.initial_source,
        "result_log": str(evaluator.result_log),
        "run_metadata": evaluator.run_metadata,
    }


def _write_spawn_safe_evaluator_shim(spec: dict[str, Any]) -> str:
    """Write an evaluator module for spawned workers."""
    payload = json.dumps(spec, sort_keys=True)
    fd, path = tempfile.mkstemp(suffix=".py", prefix="chia_spawn_eval_")
    source = (
        '"""Auto-generated spawn-safe CHIA trusted evaluator."""\n\n'
        'import json\n'
        'from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator\n'
        'from chia_trusted_verification.types import VerificationIntent\n\n'
        f'_SPEC = json.loads({payload!r})\n\n'
        '_EVALUATOR = TrustedVerificationEvaluator(\n'
        '    backend_name=_SPEC["backend_name"],\n'
        '    backend_settings=_SPEC["backend_settings"],\n'
        '    backend_resources=_SPEC["backend_resources"],\n'
        '    intent=VerificationIntent.from_dict(_SPEC["intent"]),\n'
        '    engine=_SPEC["engine"],\n'
        '    initial_source=_SPEC["initial_source"],\n'
        '    result_log=_SPEC["result_log"],\n'
        '    run_metadata=_SPEC["run_metadata"],\n'
        '    use_chia=True,\n'
        ')\n\n'
        'def evaluate(program_path):\n'
        '    return _EVALUATOR(program_path)\n'
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(source)
    return path


@contextmanager
def _spawn_safe_evolve_flows_evaluator_bridge(
    evaluator: TrustedVerificationEvaluator,
):
    """Use a spawn-safe evaluator for OpenEvolve workers."""
    import evolve_flows.evolver.bridge as evolve_bridge

    original_writer = evolve_bridge._write_chia_evaluator_shim
    spec = _trusted_evaluator_spawn_spec(evaluator)

    def write_shim(_key: str) -> str:
        return _write_spawn_safe_evaluator_shim(spec)

    evolve_bridge._write_chia_evaluator_shim = write_shim
    try:
        yield
    finally:
        evolve_bridge._write_chia_evaluator_shim = original_writer


@contextmanager
def _openevolve_vertex_bridge(endpoint: str):
    from skydiscover.extras import external as external_backends
    from skydiscover.extras.external import openevolve_backend

    original_map_config = openevolve_backend._map_config
    original_runner = external_backends._REGISTRY.get("openevolve")
    if original_runner is None:
        raise RuntimeError("SkyDiscover OpenEvolve backend is not registered")

    def map_config(config: Any, iterations: int | None, output_dir: str):
        mapped = original_map_config(config, iterations, output_dir)
        _configure_openevolve_process_pool(mapped)
        _apply_vertex_adc_to_openevolve_config(mapped, endpoint)
        return mapped

    parameters = inspect.signature(original_runner).parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    async def compatible_runner(*args: Any, **kwargs: Any):
        if not accepts_var_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return await original_runner(*args, **kwargs)

    openevolve_backend._map_config = map_config
    external_backends._REGISTRY["openevolve"] = compatible_runner
    try:
        with _openevolve_vertex_environment(endpoint):
            yield
    finally:
        external_backends._REGISTRY["openevolve"] = original_runner
        openevolve_backend._map_config = original_map_config


@dataclass
class _EvolveFlowsEvaluation:
    metrics: Any


class _TrustedEvolveFlowsEvaluator:
    def __init__(
        self,
        evaluator: TrustedVerificationEvaluator,
        *,
        output_dir: str,
    ) -> None:
        self._evaluator = evaluator
        self.output_dir = output_dir
        self._log_path = str(evaluator.result_log)
        initial = getattr(evaluator, "initial_source", None)
        self._initial_source = None if initial is None else str(initial)
        self._search_seed_seen = False

    async def evaluate_program(
        self,
        source_code: str,
        program_id: str | None = None,
    ) -> _EvolveFlowsEvaluation:
        del program_id
        role = "proposal"
        if (
            not self._search_seed_seen
            and self._initial_source is not None
            and source_code.strip() == self._initial_source.strip()
        ):
            # The search engine may evaluate the seed before new proposals.
            # CHIA already logs it as seed_baseline, so exclude it from proposal counts.
            role = "search_seed"
            self._search_seed_seen = True
        result = self._evaluator.evaluate_source(source_code, role=role)
        return _EvolveFlowsEvaluation(metrics=self._evaluator.format_engine_result(result))


def _infer_population_size(raw: Any, output_dir: str | Path) -> tuple[int, str]:
    population = getattr(raw, "population", None)
    if population:
        try:
            return len(population), "evolve_flows_result"
        except TypeError:
            pass

    checkpoints = Path(output_dir) / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints.exists():
        for path in checkpoints.glob("checkpoint_*"):
            try:
                iteration = int(path.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((iteration, path))
    if candidates:
        _, latest = max(candidates, key=lambda item: item[0])

        # Use current checkpoint metadata to count the live population.
        # Old program files may remain from earlier runs.
        metadata_path = latest / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = None
            if isinstance(metadata, dict):
                islands = metadata.get("islands")
                if isinstance(islands, list):
                    program_ids: set[str] = set()
                    valid_islands = True
                    for island in islands:
                        if not isinstance(island, list):
                            valid_islands = False
                            break
                        program_ids.update(str(program_id) for program_id in island)
                    if valid_islands:
                        return len(program_ids), "latest_checkpoint_metadata"

        programs = latest / "programs"
        if programs.exists():
            return len(list(programs.glob("*.json"))), "latest_checkpoint_files"

    iteration_count = int(getattr(raw, "iteration_count", 0) or 0)
    return (iteration_count + 1 if iteration_count else 0), "iteration_count_fallback"


class EvolutionSearchStrategy(SearchStrategy):
    def __init__(self, engine: str, *, name: str | None = None) -> None:
        if engine not in {"openevolve", "alphaevolve"}:
            raise ValueError(f"unsupported evolutionary search engine: {engine}")
        self.engine = engine
        self.name = name or engine

    def run(
        self,
        *,
        problem: SearchProblem,
        initial_program: Path,
        evaluator: TrustedVerificationEvaluator,
        config: ExperimentConfig,
    ) -> SearchResult:
        try:
            from evolve_flows.evolver.bridge import run_skydiscover
            from evolve_flows.evolver.types import EvolverInput
        except ImportError as exc:
            raise RuntimeError(
                "OpenEvolve and AlphaEvolve require evolve-flows, SkyDiscover, "
                "and their optional search dependencies. Install this project "
                "with the search extra."
            ) from exc

        raw_config = yaml.safe_load(Path(config.search_config).read_text(encoding="utf-8")) or {}
        if not isinstance(raw_config, dict):
            raise TypeError("search configuration must be a YAML mapping")
        configured = _inject_problem_guidance(raw_config, problem, self.engine, config.model)
        # AlphaEvolve counts the seed toward its program limit.
        # Add one to keep the generated proposal budget the same.
        configured["max_iterations"] = int(config.iterations) + (
            1 if self.engine == "alphaevolve" else 0
        )
        _inject_reproducibility(configured, config.seed)
        _apply_process_reproducibility(config.seed)

        config_content = yaml.safe_dump(configured, sort_keys=False)
        evolver_input = EvolverInput(
            config_path=str(Path(config.search_config).resolve()),
            initial_program=initial_program.read_text(encoding="utf-8"),
            config_content=config_content,
        )
        evolve_evaluator = _TrustedEvolveFlowsEvaluator(
            evaluator,
            output_dir=config.output_dir,
        )

        if self.engine == "openevolve":
            bridge = _openevolve_vertex_bridge(_openevolve_vertex_endpoint())
            evaluator_bridge = _spawn_safe_evolve_flows_evaluator_bridge(evaluator)
        else:
            bridge = _alphaevolve_structured_feedback_bridge()
            evaluator_bridge = nullcontext()

        with bridge, evaluator_bridge:
            raw = run_skydiscover(
                evolver_input,
                _unexpected_native_stage,
                _unexpected_native_stage,
                _unexpected_native_stage,
                evaluator=evolve_evaluator,
            )

        terminal_status = str(getattr(raw, "terminal_status", "") or "")
        error_message = getattr(raw, "error_message", None)
        if terminal_status == "error":
            raise RuntimeError(
                f"{self.name} search failed through evolve-flows/SkyDiscover: "
                f"{error_message or 'unknown error'}"
            )

        best_metrics = dict(getattr(raw, "best_metrics", {}) or {})
        population_size, population_source = _infer_population_size(raw, config.output_dir)
        metrics: dict[str, Any] = dict(best_metrics)
        metrics.update(
            {
                "requested_seed": config.seed,
                "search_terminal_status": terminal_status,
                "search_iteration_count": int(getattr(raw, "iteration_count", 0) or 0),
                "search_population_size": population_size,
                "search_population_size_source": population_source,
            }
        )
        metrics_log_path = getattr(raw, "metrics_log_path", None)
        if metrics_log_path:
            metrics["search_metrics_log_path"] = str(metrics_log_path)

        return SearchResult(
            engine=self.name,
            best_score=_score_from_metrics(best_metrics),
            best_solution=str(getattr(raw, "best_program", "") or ""),
            metrics=metrics,
            output_dir=config.output_dir,
        )


SkyDiscoverSearchStrategy = EvolutionSearchStrategy


class OpenEvolveSearchStrategy(EvolutionSearchStrategy):
    def __init__(self, *, name: str = "openevolve") -> None:
        super().__init__("openevolve", name=name)


class AlphaEvolveSearchStrategy(EvolutionSearchStrategy):
    def __init__(self, *, name: str = "alphaevolve") -> None:
        super().__init__("alphaevolve", name=name)


def _unexpected_native_stage(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "OpenEvolve and AlphaEvolve should be routed by evolve-flows through "
        "SkyDiscover's external-backend registry"
    )


def _score_from_metrics(metrics: dict[str, Any]) -> float:
    try:
        from skydiscover.utils.metrics import get_score

        return float(get_score(metrics) or 0.0)
    except Exception:
        for key in ("combined_score", "score", "fitness"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return 0.0


def _inject_problem_guidance(
    raw_config: dict[str, Any],
    problem: SearchProblem,
    engine: str,
    model_override: str | None = None,
) -> dict[str, Any]:
    configured = json.loads(json.dumps(raw_config))
    search = configured.setdefault("search", {})
    if not isinstance(search, dict):
        raise TypeError("search configuration must be a mapping")
    search["type"] = engine

    prompt = configured.setdefault("prompt", {})
    if not isinstance(prompt, dict):
        raise TypeError("search prompt configuration must be a mapping")
    prompt["system_message"] = problem.guidance.system_message
    if problem.guidance.evaluator_system_message:
        prompt["evaluator_system_message"] = problem.guidance.evaluator_system_message

    if engine == "openevolve" and model_override:
        llm = configured.setdefault("llm", {})
        if not isinstance(llm, dict):
            raise TypeError("search llm configuration must be a mapping")
        models = llm.setdefault("models", [])
        if not isinstance(models, list):
            raise TypeError("search llm.models configuration must be a list")
        base_model = dict(models[0]) if models and isinstance(models[0], dict) else {}
        base_model["name"] = _vertex_model_name(model_override)
        base_model["weight"] = 1.0
        llm["models"] = [base_model]

    if engine == "alphaevolve":
        configured["file_suffix"] = Path(problem.file_name).suffix or ".txt"
        alpha = configured.setdefault("alphaevolve", {})
        if not isinstance(alpha, dict):
            raise TypeError("alphaevolve configuration must be a mapping")
        alpha["problem_description"] = (
            problem.guidance.problem_description or problem.guidance.system_message
        )

        generation = alpha.setdefault("generation_settings", {})
        if not isinstance(generation, dict):
            raise TypeError("alphaevolve.generation_settings must be a mapping")

        # Show the full program, including fixed search-space definitions.
        # EVOLVE-BLOCK still marks the editable region.
        generation["include_full_program_in_prompt"] = True
        generation.setdefault("context", problem.guidance.system_message)

        if model_override:
            model_name = model_override.split("/", 1)[-1]
            generation["models"] = [{"name": model_name, "weight": 1.0}]
    return configured


@contextmanager
def _alphaevolve_structured_feedback_bridge():
    from skydiscover.extras.external import alphaevolve_backend

    original = alphaevolve_backend._make_alphaevolve_evaluator
    alphaevolve_backend._make_alphaevolve_evaluator = _make_alphaevolve_evaluator
    try:
        yield
    finally:
        alphaevolve_backend._make_alphaevolve_evaluator = original


def _make_alphaevolve_evaluator(evaluator_path: str, monitor_callback=None):
    spec = importlib.util.spec_from_file_location("_chia_trusted_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user_evaluate = module.evaluate
    counter = 0

    def evaluate(program: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        files = program.get("content", {}).get("files", [])
        if not files:
            return {
                "scores": {"scores": []},
                "insights": _alphaevolve_error_insights(
                    "AlphaEvolve candidate contains no files"
                ),
            }
        code = str(files[0].get("content", ""))
        suffix = Path(str(files[0].get("path", "candidate.py"))).suffix or ".py"
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            prefix="chia_alpha_",
            delete=False,
            encoding="utf-8",
        )
        try:
            temporary.write(code)
            temporary.close()
            raw = user_evaluate(temporary.name)
            if not isinstance(raw, dict):
                raw = dict(raw)
            metrics = {
                str(key): float(value)
                for key, value in raw.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            trusted_fields = {
                key.removeprefix("__"): _decode_json_or_text(value)
                for key, value in raw.items()
                if str(key).startswith("__trusted_")
            }
            insights = _alphaevolve_insights_from_trusted_fields(
                trusted_fields, metrics=metrics
            )
            counter += 1
            if monitor_callback is not None:
                try:
                    from skydiscover.search.base_database import Program

                    monitor_callback(
                        Program(
                            id=f"chia-{counter}",
                            solution=code,
                            language="python",
                            metrics=metrics,
                            iteration_found=counter,
                            generation=counter,
                        ),
                        counter,
                    )
                except Exception:
                    pass
            return {
                "scores": {
                    "scores": _alphaevolve_optimization_scores(metrics)
                },
                "insights": insights,
            }
        except Exception as exc:
            return {
                "scores": {"scores": []},
                "insights": _alphaevolve_error_insights(
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        finally:
            try:
                os.unlink(temporary.name)
            except OSError:
                pass

    return evaluate


def _alphaevolve_insights_from_trusted_fields(
    trusted_fields: dict[str, Any],
    *,
    metrics: dict[str, float] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Format trusted results as AlphaEvolve insights.

    Keep full evidence in the local log and return concise search feedback.
    """
    items: list[dict[str, str]] = []

    backend = trusted_fields.get("trusted_backend")
    intent_id = trusted_fields.get("trusted_intent_id")
    status = trusted_fields.get("trusted_status")
    if any(value is not None for value in (backend, intent_id, status)):
        status_payload = {
            "backend": backend,
            "intent_id": intent_id,
            "status": status,
        }
        items.append(
            {
                "label": "trusted_verification",
                "text": json.dumps(status_payload, sort_keys=True),
            }
        )

    feedback = trusted_fields.get("trusted_feedback_json")
    if feedback not in (None, {}, [], ""):
        items.append(
            {
                "label": "trusted_feedback",
                "text": _alphaevolve_compact_feedback_insight(feedback),
            }
        )

    diagnostics = trusted_fields.get("trusted_diagnostics_json")
    if diagnostics not in (None, {}, [], ""):
        items.append(
            {
                "label": "trusted_diagnostics",
                "text": _alphaevolve_insight_text(diagnostics),
            }
        )

    if metrics:
        items.append(
            {
                "label": "trusted_metrics",
                "text": json.dumps(metrics, sort_keys=True),
            }
        )

    return {"insights": items}


def _alphaevolve_optimization_scores(
    metrics: dict[str, float],
) -> list[dict[str, float]]:
    """Return the trusted higher-is-better score for AlphaEvolve.

    Other metrics are diagnostics, not optimization targets.
    """
    if "combined_score" in metrics:
        return [{"metric": "combined_score", "score": metrics["combined_score"]}]
    if "score" in metrics:
        return [{"metric": "score", "score": metrics["score"]}]
    return []


def _alphaevolve_error_insights(message: str) -> dict[str, list[dict[str, str]]]:
    return {"insights": [{"label": "evaluation_error", "text": str(message)}]}


def _alphaevolve_compact_feedback_insight(value: Any) -> str:
    """Return concise search and repair feedback for AlphaEvolve.

    Full results remain in trusted_results.jsonl.
    """
    decoded = _decode_json_or_text(value)
    if not isinstance(decoded, dict):
        return _alphaevolve_insight_text(decoded)

    mutation_ids = decoded.get("mutation_ids")
    summary = str(decoded.get("summary") or "").strip()
    action = str(decoded.get("suggested_action") or "").strip()
    failed_ids = [str(item) for item in decoded.get("failed_check_ids", []) if item]
    messages = [str(item).strip() for item in decoded.get("messages", []) if str(item).strip()]
    objective = decoded.get("objective")

    lines: list[str] = []
    if isinstance(mutation_ids, list):
        formatted = ", ".join(str(item) for item in mutation_ids) or "<seed>"
        lines.append(f"Candidate mutations: {formatted}")
    if summary:
        lines.append(summary)
    if failed_ids:
        lines.append("Failed trusted checks: " + ", ".join(failed_ids))
    if messages:
        lines.append("Failure detail: " + " | ".join(messages[:3]))
    if isinstance(objective, dict) and objective.get("status") == "ok":
        lines.append(f"Trusted objective value: {objective.get('value')}")
    if action:
        lines.append("Next action: " + action)

    return "\n".join(lines) if lines else _alphaevolve_insight_text(decoded)


def _alphaevolve_insight_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _decode_json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _inject_reproducibility(configured: dict[str, Any], seed: int) -> None:
    for name in ("seed", "random_seed"):
        if name in configured:
            configured[name] = seed
    search = configured.get("search")
    if isinstance(search, dict):
        for name in ("seed", "random_seed"):
            if name in search:
                search[name] = seed


def _apply_process_reproducibility(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _apply_reproducibility(seed: int, config: Any) -> None:
    _apply_process_reproducibility(seed)
    for name in ("seed", "random_seed"):
        if hasattr(config, name):
            try:
                setattr(config, name, seed)
            except Exception:
                pass
    search = getattr(config, "search", None)
    if search is not None:
        for name in ("seed", "random_seed"):
            if hasattr(search, name):
                try:
                    setattr(search, name, seed)
                except Exception:
                    pass


def _simple_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _simple_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_simple_value(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _simple_value(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _simple_value(vars(value))
        except Exception:
            pass
    return str(value)
