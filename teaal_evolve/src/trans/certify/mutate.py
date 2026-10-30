

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .candidate import Candidate
from .schema_utils import set_component_name
from .knobs import (
    CandidateKnob,
    KNOB_SPACE_RESTRICTED,
    extract_knob_map,
    parse_engine_instruction_value,
    parse_rank_list_value,
    parse_spacetime_value,
)


def set_knob_value(
    candidate: Candidate,
    knob_id: str,
    value: str,
    label: Optional[str] = None,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> Candidate:
    

    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    if knob_id not in knob_map:
        raise KeyError(f"Unknown knob_id: {knob_id}")

    knob = knob_map[knob_id]
    _validate_knob_value(knob, value)

    mutated = candidate.clone(label=label or f"{candidate.label}-mutated")

    _apply_resolved_knob(mutated, knob, value)

    mutated.knobs = dict(candidate.knobs)
    mutated.knobs[knob_id] = value
    return mutated




def apply_knob_specs_atomic(
    candidate: Candidate,
    knob_specs: Sequence[str],
    label: Optional[str] = None,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> Candidate:
    

    if not knob_specs:
        raise ValueError("knob_specs must not be empty")

    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    assignments: List[Tuple[CandidateKnob, str]] = []
    seen_knob_ids = set()

    for raw_spec in knob_specs:
        knob_id, explicit_value = _parse_knob_spec(raw_spec)
        if knob_id in seen_knob_ids:
            raise ValueError(
                f"Duplicate knob assignment in atomic mutation: {knob_id}"
            )
        seen_knob_ids.add(knob_id)

        if knob_id not in knob_map:
            raise KeyError(f"Unknown knob_id: {knob_id}")
        knob = knob_map[knob_id]

        if explicit_value is None:
            value = _next_knob_value(knob)
        else:
            value = explicit_value
            _validate_knob_value(knob, value)

        assignments.append((knob, value))

    mutated = candidate.clone(label=label or f"{candidate.label}-mutated")
    mutated.knobs = dict(candidate.knobs)

    for knob, value in assignments:
        _apply_resolved_knob(mutated, knob, value)
        mutated.knobs[knob.knob_id] = value

    return mutated

def step_knob_value(
    candidate: Candidate,
    knob_id: str,
    label: Optional[str] = None,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> Candidate:
    

    knob_map = extract_knob_map(candidate, knob_space=knob_space)
    if knob_id not in knob_map:
        raise KeyError(f"Unknown knob_id: {knob_id}")

    knob = knob_map[knob_id]
    new_value = _next_knob_value(knob)
    return set_knob_value(
        candidate,
        knob_id=knob_id,
        value=new_value,
        label=label or f"{candidate.label}-{knob_id.split('::')[0]}-step",
        knob_space=knob_space,
    )


def toggle_onchip_tensor_component(
    candidate: Candidate,
    tensor_name: str,
    label: Optional[str] = None,
    *,
    knob_space: str = KNOB_SPACE_RESTRICTED,
) -> Candidate:
    

    knob_id = f"binding.tensor.component::{tensor_name}"
    knob_map = extract_knob_map(candidate, knob_space=knob_space)

    if knob_id not in knob_map:
        raise KeyError(f"No grouped on-chip tensor component knob found for {tensor_name}")

    knob = knob_map[knob_id]
    if knob.current_value == "mixed":
        raise ValueError(
            f"Cannot safely toggle grouped knob {knob_id} because its current value is mixed"
        )

    onchip_alternatives = [
        value
        for value in knob.allowed_values
        if value in {"SBUF", "PSUM"} and value != knob.current_value
    ]
    if not onchip_alternatives:
        raise ValueError(f"No on-chip alternative value available for knob {knob_id}")

    new_value = onchip_alternatives[0]
    return set_knob_value(
        candidate=candidate,
        knob_id=knob_id,
        value=new_value,
        label=label or f"{candidate.label}-{tensor_name}-{new_value.lower()}",
        knob_space=knob_space,
    )




def _parse_knob_spec(knob_spec: str) -> Tuple[str, Optional[str]]:
    if not isinstance(knob_spec, str) or not knob_spec.strip():
        raise ValueError(f"Invalid knob specification: {knob_spec!r}")
    if "=" not in knob_spec:
        return knob_spec, None
    knob_id, value = knob_spec.split("=", 1)
    if not knob_id or not value:
        raise ValueError(
            "Explicit knob specification must have form '<knob_id>=<value>', "
            f"got {knob_spec!r}"
        )
    return knob_id, value


def _next_knob_value(knob: CandidateKnob) -> str:
    if knob.current_value == "mixed":
        raise ValueError(
            f"Cannot step knob {knob.knob_id} because its current value is mixed"
        )
    if knob.current_value not in knob.allowed_values:
        raise ValueError(
            f"Current value {knob.current_value!r} is not present in "
            f"allowed_values for {knob.knob_id}"
        )
    if len(knob.allowed_values) < 2:
        raise ValueError(f"No alternative value available for knob {knob.knob_id}")

    current_index = knob.allowed_values.index(knob.current_value)
    next_index = (current_index + 1) % len(knob.allowed_values)
    return knob.allowed_values[next_index]


def _apply_resolved_knob(
    mutated: Candidate,
    knob: CandidateKnob,
    value: str,
) -> None:
    if knob.kind == "tensor_component_group":
        _apply_tensor_component_group(mutated, knob, value)
    elif knob.kind == "engine_instruction":
        _apply_engine_instruction(mutated, knob, value)
    elif knob.kind == "mapping_loop_order":
        _apply_loop_order(mutated, knob, value)
    elif knob.kind == "mapping_spacetime":
        _apply_spacetime(mutated, knob, value)
    elif knob.kind == "format_rank_order":
        _apply_format_rank_order(mutated, knob, value)
    else:
        raise ValueError(f"Unsupported knob kind: {knob.kind}")

def _validate_knob_value(knob: CandidateKnob, value: str) -> None:
    if value not in knob.allowed_values:
        raise ValueError(
            f"Illegal value {value!r} for knob {knob.knob_id}; "
            f"allowed values are {knob.allowed_values}"
        )


def _apply_tensor_component_group(mutated: Candidate, knob: CandidateKnob, value: str) -> None:
    tensor_bindings = mutated.spec_dict["binding"]["tensor"]
    for index in knob.binding_tensor_indices:
        set_component_name(tensor_bindings[index], value)


def _apply_engine_instruction(mutated: Candidate, knob: CandidateKnob, value: str) -> None:
    component, instruction = parse_engine_instruction_value(value)
    engine_bindings = mutated.spec_dict["binding"]["engine"]
    for index in knob.binding_tensor_indices:
        set_component_name(engine_bindings[index], component)
        engine_bindings[index]["instruction"] = instruction


def _apply_loop_order(mutated: Candidate, knob: CandidateKnob, value: str) -> None:
    stage_name = knob.metadata["stage_name"]
    mutated.spec_dict["mapping"]["loop-order"][stage_name] = parse_rank_list_value(value)


def _apply_spacetime(mutated: Candidate, knob: CandidateKnob, value: str) -> None:
    stage_name = knob.metadata["stage_name"]
    space_ranks, time_ranks = parse_spacetime_value(value)
    mutated.spec_dict["mapping"]["spacetime"][stage_name] = {
        "space": space_ranks,
        "time": time_ranks,
    }


def _apply_format_rank_order(mutated: Candidate, knob: CandidateKnob, value: str) -> None:
    tensor_name = knob.metadata["tensor_name"]
    consumer_name = knob.metadata["consumer_name"]
    mutated.spec_dict["format"][tensor_name]["rank-order"][consumer_name] = parse_rank_list_value(value)
