

from __future__ import annotations

from typing import Dict

from src.trans.model.bottleneck import BottleneckAnalysisModel

from .candidate import CandidateArtifacts
from .obligation import ObjectiveResult



def compute_bottleneck_objective(
    artifacts: CandidateArtifacts,
    shapes: Dict[str, int],
) -> ObjectiveResult:
    

    assert artifacts.elaborated_binding is not None

    try:
        model = BottleneckAnalysisModel(artifacts.elaborated_binding, shapes)
        total_latency, breakdown = model.get_latency()
        normalized = {
            ", ".join(group_name): {
                component: float(latency)
                for component, latency in component_breakdown.items()
            }
            for group_name, component_breakdown in breakdown.items()
        }
        return ObjectiveResult(
            enabled=True,
            objective_name="bottleneck_latency",
            status="ok",
            value=float(total_latency),
            breakdown=normalized,
        )
    except Exception as exc:                                              
        return ObjectiveResult(
            enabled=True,
            objective_name="bottleneck_latency",
            status="error",
            error_type=type(exc).__name__,
            message=str(exc),
        )
