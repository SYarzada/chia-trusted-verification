from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from chia_trusted_verification.types import VerificationIntent


ENGINE_ALIASES = {
    "structured_random": "structured_random",
    "structured_walk": "structured_walk",
    "openevolve": "openevolve",
    "openevolve_structured": "openevolve",
    "openevolve_guided": "openevolve",
    "alphaevolve": "alphaevolve",
    "alphaevolve_structured": "alphaevolve",
    "alphaevolve_guided": "alphaevolve",
    "llm_agent": "llm_agent",
    "llm_agent_structured": "llm_agent",
    "llm_agent_guided": "llm_agent",
}
SUPPORTED_ENGINES = frozenset(ENGINE_ALIASES)


def base_search_engine(engine: str) -> str:
    try:
        return ENGINE_ALIASES[str(engine)]
    except KeyError as exc:
        raise ValueError(f"unsupported search engine/policy: {engine!r}") from exc


def search_feedback_mode(engine: str) -> str:
    value = str(engine)
    return "score_only" if value.endswith("_structured") else "guided"
PIPELINE_STAGES = ("prepare", "verify", "evaluate", "feedback")


def _resolve(base: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


@dataclass(frozen=True)
class ExperimentConfig:
    engine: str
    backend_name: str
    iterations: int
    model: str | None
    experiment_name: str
    run_id: str
    seed: int
    output_dir: str
    result_log: str
    search_config: str
    backend_settings: dict[str, Any]
    backend_resources: dict[str, dict[str, float]] = field(default_factory=dict)
    cost_tracking: dict[str, Any] = field(default_factory=dict)
    intent: VerificationIntent = field(
        default_factory=lambda: VerificationIntent("default", "artifact")
    )

    def __post_init__(self) -> None:
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported search engine {self.engine!r}; "
                f"expected one of {sorted(SUPPORTED_ENGINES)}"
            )
        if not self.backend_name:
            raise ValueError("backend_name must be non-empty")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not self.experiment_name:
            raise ValueError("experiment_name must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        for stage, resources in self.backend_resources.items():
            if stage not in PIPELINE_STAGES:
                raise ValueError(f"unknown backend resource stage {stage!r}")
            for name, amount in resources.items():
                if not name or float(amount) <= 0:
                    raise ValueError(f"invalid resource requirement {name!r}: {amount!r}")

    def resources_for(self, stage: str) -> dict[str, float]:
        return dict(self.backend_resources.get(stage, {}))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["intent"] = self.intent.to_dict()
        return value


def load_experiment_config(
    path: str | Path,
    *,
    engine_override: str | None = None,
    backend_override: str | None = None,
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("experiment configuration must be a YAML mapping")

    experiment = dict(raw.get("experiment", {}))
    backend = dict(raw.get("backend", {}))
    verification = dict(raw.get("verification", {}))
    search = dict(raw.get("search", {}))
    base = config_path.parent

    engine = str(engine_override or experiment.get("engine", "openevolve"))
    backend_name = str(backend_override or backend.get("name", "teaal_evolve"))
    search_configs = dict(search.get("configs", {}))
    search_config = search_configs.get(engine)
    if search_config is None:
        search_config = search_configs.get(base_search_engine(engine))
    if search_config is None:
        raise ValueError(
            f"search.configs has no entry for {engine!r} or its base engine "
            f"{base_search_engine(engine)!r}"
        )

    output_dir = _resolve(base, str(experiment.get("output_dir", "../runs/default")))
    raw_result_log = experiment.get("result_log")
    result_log = (
        str((Path(output_dir) / "trusted_results.jsonl").resolve())
        if raw_result_log is None
        else _resolve(base, str(raw_result_log))
    )

    settings = dict(backend.get("settings", {}))
    settings["_config_dir"] = str(base)
    raw_resources = dict(backend.get("resources", {}))
    resources: dict[str, dict[str, float]] = {}
    for stage in PIPELINE_STAGES:
        stage_resources = raw_resources.get(stage, {})
        if stage_resources is None:
            stage_resources = {}
        if not isinstance(stage_resources, Mapping):
            raise TypeError(f"backend.resources.{stage} must be a mapping")
        resources[stage] = {
            str(name): float(amount) for name, amount in stage_resources.items()
        }

    raw_intent = dict(verification.get("intent", {}))
    if not raw_intent:
        raise ValueError("verification.intent must be provided")
    intent = VerificationIntent.from_dict(raw_intent)

    kernel_label = str(settings.get("kernel_label", "artifact"))
    seed = int(experiment.get("seed", 0))
    experiment_name = str(experiment.get("name", "chia_trusted_verification"))
    run_id = str(
        experiment.get(
            "run_id",
            f"{experiment_name}-{kernel_label}-{engine}-seed{seed}",
        )
    )
    cost_tracking = dict(experiment.get("cost_tracking", {}))

    return ExperimentConfig(
        engine=engine,
        backend_name=backend_name,
        iterations=int(experiment.get("iterations", 20)),
        model=(None if experiment.get("model") is None else str(experiment["model"])),
        experiment_name=experiment_name,
        run_id=run_id,
        seed=seed,
        output_dir=output_dir,
        result_log=result_log,
        search_config=_resolve(base, str(search_config)),
        backend_settings=settings,
        backend_resources=resources,
        cost_tracking=cost_tracking,
        intent=intent,
    )
