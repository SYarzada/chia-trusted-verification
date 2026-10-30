

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LegalityStageResult:
    

    stage_name: str
    ok: bool
    error_type: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class StageObligation:
    

    obligation_id: str
    einsum_name: str
    instruction_name: str
    instruction_position: int
    engine: str
    output_tensor: str
    input_tensors: List[str]
    bound_ranks: List[str]
    unbound_ranks: List[str]
    status: str = "pass"
    detail: str = "Bound Einsum extracted successfully"


@dataclass(frozen=True)
class InterfaceObligation:
    

    obligation_id: str
    tensor_name: str
    producer_einsum: str
    consumer_einsum: str
    producer_instruction: str
    consumer_instruction: str
    producer_position: int
    consumer_position: int
    explicit_reshape_required: bool
    status: str
    detail: str


@dataclass(frozen=True)
class LocalLegalityObligation:
    

    obligation_id: str
    knob_id: str
    kind: str = "tensor_component_group"
    tensor_name: str = ""
    stage_name: str = ""
    consumer_name: str = ""
    current_value: str = ""
    allowed_values: List[str] = field(default_factory=list)
    binding_tensor_indices: List[int] = field(default_factory=list)
    status: str = "pass"
    detail: str = ""


@dataclass(frozen=True)
class ObjectiveResult:
    

    enabled: bool = False
    objective_name: Optional[str] = None
    status: str = "disabled"
    value: Optional[float] = None
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_type: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class CandidateReport:
    

    candidate_label: str
    is_legal: bool
    legality_obligations: List[LegalityStageResult]
    local_legality_obligations: List[LocalLegalityObligation]
    stage_obligations: List[StageObligation]
    interface_obligations: List[InterfaceObligation]
    objective: ObjectiveResult

    def num_local_legality_obligations(self) -> int:
        return len(self.local_legality_obligations)

    def num_stage_obligations(self) -> int:
        return len(self.stage_obligations)

    def num_interface_obligations(self) -> int:
        return len(self.interface_obligations)

    def has_valid_objective(self) -> bool:
        return self.objective.enabled and self.objective.status == "ok" and self.objective.value is not None
