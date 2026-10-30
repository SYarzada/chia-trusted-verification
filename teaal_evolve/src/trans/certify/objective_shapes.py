

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:                                                         
    from .candidate import Candidate, CandidateArtifacts


class ObjectiveShapeResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ObjectiveShapePreset:
    

    kernel_key: str
    preset_name: str
    shapes: Mapping[str, int]
    provenance: str


@dataclass(frozen=True)
class FlattenRelation:
    output_rank: str
    input_ranks: Tuple[str, ...]
    einsum: str


@dataclass(frozen=True)
class UniformShapeRelation:
    source_rank: str
    upper_rank: str
    lower_rank: str
    lower_shape: int | str
    einsum: str


@dataclass
class ObjectiveShapeResolutionReport:
    

    kernel_key: str
    preset_name: Optional[str]
    preset_provenance: Optional[str]
    shapes: Dict[str, int]
    explicit_overrides: Dict[str, int] = field(default_factory=dict)
    derived_shapes: Dict[str, int] = field(default_factory=dict)
    unresolved_symbols: List[str] = field(default_factory=list)
    consistency_errors: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.unresolved_symbols and not self.consistency_errors

    def format_error(self) -> str:
        details: List[str] = []
        if self.unresolved_symbols:
            details.append(
                "unresolved objective ranks: " + ", ".join(self.unresolved_symbols)
            )
        if self.consistency_errors:
            details.append("shape consistency errors: " + "; ".join(self.consistency_errors))
        if self.preset_name:
            details.append(f"preset={self.preset_name!r}")
        details.append(
            "known shapes={" + ", ".join(
                f"{rank}={self.shapes[rank]}" for rank in sorted(self.shapes)
            ) + "}"
        )
        return f"Kernel {self.kernel_key!r} has " + "; ".join(details)


                                                                          
                                                       
_OBJECTIVE_SHAPE_PRESETS: Dict[str, ObjectiveShapePreset] = {
    "attention": ObjectiveShapePreset(
        kernel_key="attention",
        preset_name="attention-seq512-head128",
                                                                    
                                                             
        shapes={"P3": 2, "M2": 1},
        provenance=(
            "example/experiments/scripts/attention.py, seq_len=512; "
            "remaining tile/head shapes come from attention.yaml"
        ),
    ),
    "one_pass_attention": ObjectiveShapePreset(
        kernel_key="one_pass_attention",
        preset_name="one-pass-attention-seq2048-head128",
                                                               
                                                                          
                                                                       
        shapes={"P4": 1, "M3": 4},
        provenance=(
            "one_pass_attention.yaml tiled 2048-token evaluation shape: "
            "P4=1 and M3=4; recurrent extents are derived"
        ),
    ),
    "one_pass_attention_no_pipelining": ObjectiveShapePreset(
        kernel_key="one_pass_attention_no_pipelining",
        preset_name="one-pass-attention-no-pipeline-seq2048-head128",
        shapes={"P4": 1, "M3": 4},
        provenance=(
            "one_pass_attention_no_pipelining.yaml tiled 2048-token evaluation "
            "shape: P4=1 and M3=4; recurrent extents are derived"
        ),
    ),
    "matmul": ObjectiveShapePreset(
        kernel_key="matmul",
        preset_name="matmul-512x1024x2048",
                                                                             
                                              
        shapes={"K1": 1, "M1": 1, "N1": 1},
        provenance=(
            "matmul.yaml K0=1024, M0=512, N0=2048 with unit outer tile counts"
        ),
    ),
    "mamba": ObjectiveShapePreset(
        kernel_key="mamba",
        preset_name="mamba-d768-seq512-state16-batch1",
                                                                 
                                                             
        shapes={"D1": 6, "D10": 6, "N": 16, "B": 1, "T1": 4},
        provenance=(
            "example/experiments/scripts/mamba.py with d_model=768, "
            "seq_len=512, N=16, B=1"
        ),
    ),
                                                              
    "vector_scalar_overlap": ObjectiveShapePreset(
        kernel_key="vector_scalar_overlap",
        preset_name="vector-scalar-overlap-default",
        shapes={"P": 128},
        provenance="existing certification objective default; PI is derived",
    ),
    "unary": ObjectiveShapePreset(
        kernel_key="unary",
        preset_name="unary-yaml-shapes",
        shapes={},
        provenance="all independent shapes are declared in unary.yaml",
    ),
    "unary_reduce_one_tile": ObjectiveShapePreset(
        kernel_key="unary_reduce_one_tile",
        preset_name="unary-reduce-yaml-shapes",
        shapes={},
        provenance="all independent shapes are declared in unary_reduce_one_tile.yaml",
    ),
}

                                                                        
_KNOWN_KERNEL_KEYS: Tuple[str, ...] = tuple(
    sorted(_OBJECTIVE_SHAPE_PRESETS, key=len, reverse=True)
)

_UNIFORM_RE = re.compile(r"^uniform_shape\(([^)]+)\)$")


def infer_objective_kernel_key(candidate: "Candidate") -> str:
    

    for source in getattr(candidate, "source_files", ()):
        stem = str(source).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem in _OBJECTIVE_SHAPE_PRESETS:
            return stem

    label = str(getattr(candidate, "label", "")).lower().replace("-", "_")
    for key in _KNOWN_KERNEL_KEYS:
        if key in label:
            return key

    spec = getattr(candidate, "spec_dict", {})
    options = spec.get("options", {}) if isinstance(spec, dict) else {}
    kernel_name = str(options.get("kernel-name", ""))
    if kernel_name != "attn":
        return kernel_name or "unknown"

    declaration = spec.get("einsum", {}).get("declaration", {})
    if not isinstance(declaration, dict) or "RM" not in declaration:
        return "attention"

                                                                              
                                                                       
    shape = spec.get("einsum", {}).get("shape", {})
    if isinstance(shape, dict) and int(shape.get("P3", 0) or 0) == 8:
        return "one_pass_attention_no_pipelining"
    return "one_pass_attention"


def get_objective_shape_preset(candidate: "Candidate") -> Optional[ObjectiveShapePreset]:
    return _OBJECTIVE_SHAPE_PRESETS.get(infer_objective_kernel_key(candidate))


def get_spec_objective_shapes(candidate: "Candidate") -> Dict[str, int]:
    

    spec_shapes = getattr(candidate, "spec_dict", {}).get("einsum", {}).get("shape", {})
    result: Dict[str, int] = {}
    if not isinstance(spec_shapes, dict):
        return result
    for rank, value in spec_shapes.items():
        try:
            converted = int(value)
        except (TypeError, ValueError):
            continue
        if converted <= 0:
            continue
        result[str(rank)] = converted
    return result


def _parse_partition_roots(raw: Any) -> Tuple[str, ...]:
    text = str(raw).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return tuple(part.strip() for part in text.split(",") if part.strip())


def extract_shape_relations(
    candidate: "Candidate",
) -> Tuple[List[FlattenRelation], List[UniformShapeRelation]]:
    

    partitioning = (
        getattr(candidate, "spec_dict", {})
        .get("mapping", {})
        .get("partitioning", {})
    )
    flatten: List[FlattenRelation] = []
    uniform: List[UniformShapeRelation] = []
    if not isinstance(partitioning, dict):
        return flatten, uniform

    seen_flatten: Set[Tuple[str, Tuple[str, ...]]] = set()
    seen_uniform: Set[Tuple[str, str, str, str]] = set()

    for einsum, parts in partitioning.items():
        if not isinstance(parts, dict):
            continue
        for raw_roots, raw_directives in parts.items():
            roots = _parse_partition_roots(raw_roots)
            directives = list(raw_directives or [])
            if len(roots) > 1:
                if any(str(directive).strip() == "flatten()" for directive in directives):
                    relation = FlattenRelation("".join(roots), roots, str(einsum))
                    key = (relation.output_rank, relation.input_ranks)
                    if key not in seen_flatten:
                        flatten.append(relation)
                        seen_flatten.add(key)
                continue

            if len(roots) != 1:
                continue
            root = roots[0]
            uniform_directives: List[int | str] = []
            for directive in directives:
                match = _UNIFORM_RE.match(str(directive).strip())
                if not match:
                    continue
                arg = match.group(1).strip()
                try:
                    lower_shape: int | str = int(arg)
                except ValueError:
                    lower_shape = arg
                uniform_directives.append(lower_shape)

            source = root
            count = len(uniform_directives)
            for index, lower_shape in enumerate(uniform_directives):
                upper = root + str(count - index)
                lower = root + str(count - index - 1)
                if index < count - 1:
                    lower += "I"
                relation = UniformShapeRelation(
                    source_rank=source,
                    upper_rank=upper,
                    lower_rank=lower,
                    lower_shape=lower_shape,
                    einsum=str(einsum),
                )
                key = (source, upper, lower, str(lower_shape))
                if key not in seen_uniform:
                    uniform.append(relation)
                    seen_uniform.add(key)
                source = lower

    return flatten, uniform


def _declared_ranks(candidate: "Candidate") -> Set[str]:
    declaration = (
        getattr(candidate, "spec_dict", {})
        .get("einsum", {})
        .get("declaration", {})
    )
    ranks: Set[str] = set()
    if isinstance(declaration, dict):
        for tensor_ranks in declaration.values():
            if isinstance(tensor_ranks, Sequence) and not isinstance(tensor_ranks, (str, bytes)):
                ranks.update(str(rank) for rank in tensor_ranks)
    return ranks


def _set_derived(
    shapes: Dict[str, int],
    derived: Dict[str, int],
    rank: str,
    value: int,
    errors: List[str],
    reason: str,
) -> bool:
    if value <= 0:
        errors.append(f"{reason} derives non-positive {rank}={value}")
        return False
    existing = shapes.get(rank)
    if existing is not None:
        if existing != value:
            errors.append(
                f"{reason} requires {rank}={value}, but the resolved value is {existing}"
            )
        return False
    shapes[rank] = value
    derived[rank] = value
    return True


def derive_objective_shapes(
    candidate: "Candidate",
    base_shapes: Mapping[str, int],
) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    

    shapes = {str(rank): int(value) for rank, value in base_shapes.items()}
    derived: Dict[str, int] = {}
    errors: List[str] = []
    flatten_relations, uniform_relations = extract_shape_relations(candidate)
    declared = _declared_ranks(candidate)

                                                                               
                                                                               
                                                                  
    recurrence_relations = [
        (rank, rank[:-1])
        for rank in sorted(declared)
        if rank.endswith("I") and rank[:-1]
    ]

    changed = True
    while changed:
        changed = False

        for recurrent, base in recurrence_relations:
            if recurrent not in shapes and base in shapes:
                changed |= _set_derived(
                    shapes,
                    derived,
                    recurrent,
                    shapes[base] + 1,
                    errors,
                    f"recurrent extent {recurrent}={base}+1",
                )

        for relation in uniform_relations:
            lower_value: Optional[int]
            if isinstance(relation.lower_shape, int):
                lower_value = relation.lower_shape
                changed |= _set_derived(
                    shapes,
                    derived,
                    relation.lower_rank,
                    lower_value,
                    errors,
                    f"{relation.einsum} uniform_shape({lower_value})",
                )
            else:
                lower_value = shapes.get(relation.lower_shape)
                if lower_value is not None:
                    changed |= _set_derived(
                        shapes,
                        derived,
                        relation.lower_rank,
                        lower_value,
                        errors,
                        f"{relation.einsum} uniform_shape({relation.lower_shape})",
                    )

            source_value = shapes.get(relation.source_rank)
            upper_value = shapes.get(relation.upper_rank)
            lower_rank_value = shapes.get(relation.lower_rank)

            if source_value is not None and lower_rank_value is not None:
                if source_value % lower_rank_value != 0:
                    message = (
                        f"{relation.einsum}: {relation.source_rank}={source_value} is not "
                        f"divisible by {relation.lower_rank}={lower_rank_value}"
                    )
                    if message not in errors:
                        errors.append(message)
                else:
                    changed |= _set_derived(
                        shapes,
                        derived,
                        relation.upper_rank,
                        source_value // lower_rank_value,
                        errors,
                        f"{relation.einsum} uniform partition",
                    )
            elif upper_value is not None and lower_rank_value is not None:
                changed |= _set_derived(
                    shapes,
                    derived,
                    relation.source_rank,
                    upper_value * lower_rank_value,
                    errors,
                    f"{relation.einsum} inverse uniform partition",
                )
            elif source_value is not None and upper_value is not None:
                if source_value % upper_value != 0:
                    message = (
                        f"{relation.einsum}: {relation.source_rank}={source_value} is not "
                        f"divisible by {relation.upper_rank}={upper_value}"
                    )
                    if message not in errors:
                        errors.append(message)
                else:
                    changed |= _set_derived(
                        shapes,
                        derived,
                        relation.lower_rank,
                        source_value // upper_value,
                        errors,
                        f"{relation.einsum} inverse uniform partition",
                    )

        for relation in flatten_relations:
            inputs = [shapes.get(rank) for rank in relation.input_ranks]
            if all(value is not None for value in inputs):
                product = 1
                for value in inputs:
                    assert value is not None
                    product *= value
                changed |= _set_derived(
                    shapes,
                    derived,
                    relation.output_rank,
                    product,
                    errors,
                    f"{relation.einsum} flatten({', '.join(relation.input_ranks)})",
                )

                                            
    errors = list(dict.fromkeys(errors))
    return shapes, derived, errors


def resolve_objective_shape_report(
    candidate: "Candidate",
    explicit_shapes: Optional[Mapping[str, int]] = None,
    *,
    required_symbols: Optional[Iterable[str]] = None,
) -> ObjectiveShapeResolutionReport:
    

    kernel_key = infer_objective_kernel_key(candidate)
    preset = get_objective_shape_preset(candidate)
    base = get_spec_objective_shapes(candidate)
    if preset is not None:
        base.update({str(rank): int(value) for rank, value in preset.shapes.items()})

    overrides: Dict[str, int] = {}
    if explicit_shapes is not None:
        for rank, value in explicit_shapes.items():
            converted = int(value)
            if converted <= 0:
                raise ObjectiveShapeResolutionError(
                    f"Objective shape {rank} must be positive, got {converted}"
                )
            overrides[str(rank)] = converted
        base.update(overrides)

    shapes, derived, errors = derive_objective_shapes(candidate, base)
    unresolved: List[str] = []
    if required_symbols is not None:
        unresolved = sorted(
            str(symbol) for symbol in set(required_symbols) if str(symbol) not in shapes
        )

    return ObjectiveShapeResolutionReport(
        kernel_key=kernel_key,
        preset_name=preset.preset_name if preset else None,
        preset_provenance=preset.provenance if preset else None,
        shapes=shapes,
        explicit_overrides=overrides,
        derived_shapes=derived,
        unresolved_symbols=unresolved,
        consistency_errors=errors,
    )


def required_objective_symbols(artifacts: "CandidateArtifacts") -> List[str]:
    

    mapping = getattr(artifacts, "elaborated_mapping", None)
    if mapping is None:
        return []
    return sorted({str(symbol) for symbol in mapping.get_symbols()})


def validate_objective_shape_report(report: ObjectiveShapeResolutionReport) -> None:
    if not report.is_complete:
        raise ObjectiveShapeResolutionError(report.format_error())


def objective_shape_presets() -> Dict[str, ObjectiveShapePreset]:
    

    return dict(_OBJECTIVE_SHAPE_PRESETS)
