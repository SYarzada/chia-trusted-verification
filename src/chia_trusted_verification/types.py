from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PIPELINE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class VerificationIntent:
    intent_id: str
    artifact_type: str
    requirements: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationIntent":
        return cls(
            intent_id=str(value.get("intent_id", "default")),
            artifact_type=str(value.get("artifact_type", "artifact")),
            requirements=tuple(str(item) for item in value.get("requirements", [])),
            parameters=dict(value.get("parameters", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchGuidance:
    system_message: str = (
        "Propose an improved candidate while preserving the candidate representation "
        "and constraints defined by the initial artifact."
    )
    evaluator_system_message: str | None = None
    problem_description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchProblem:
    initial_source: str
    file_name: str = "initial_program.py"
    metadata: dict[str, Any] = field(default_factory=dict)
    guidance: SearchGuidance = field(default_factory=SearchGuidance)


@dataclass(frozen=True)
class SearchResult:
    engine: str
    best_score: float
    best_solution: str
    metrics: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineResult:
    backend: str
    intent_id: str
    status: str
    metrics: dict[str, float]
    feedback: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    stage_timings_s: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineResult":
        return cls(
            backend=str(value.get("backend", "unknown")),
            intent_id=str(value.get("intent_id", "default")),
            status=str(value.get("status", "error")),
            metrics={str(k): float(v) for k, v in value.get("metrics", {}).items()},
            feedback=dict(value.get("feedback", {})),
            evidence=dict(value.get("evidence", {})),
            trace=dict(value.get("trace", {})),
            diagnostics=[dict(item) for item in value.get("diagnostics", []) if isinstance(item, Mapping)],
            stage_timings_s={str(k): float(v) for k, v in value.get("stage_timings_s", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
