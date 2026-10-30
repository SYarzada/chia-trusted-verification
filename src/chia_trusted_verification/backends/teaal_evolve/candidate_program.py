from __future__ import annotations

import ast
from copy import deepcopy
import pprint
from typing import Any, Mapping, Sequence


class CandidateProgramError(ValueError):
    """Raised when an evolved program violates the bounded proposal contract."""


def extract_candidate_payload(source: str) -> dict[str, Any]:
    """Read a literal ``CANDIDATE`` assignment without executing candidate code."""

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CandidateProgramError(f"candidate program is not valid Python: {exc}") from exc

    assignments: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CANDIDATE"
            for target in node.targets
        ):
            assignments.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CANDIDATE"
            and node.value is not None
        ):
            assignments.append(node.value)

    if len(assignments) != 1:
        raise CandidateProgramError(
            "candidate program must contain exactly one top-level CANDIDATE assignment"
        )
    try:
        payload = ast.literal_eval(assignments[0])
    except (ValueError, TypeError) as exc:
        raise CandidateProgramError("CANDIDATE must be a literal dictionary") from exc
    if not isinstance(payload, dict):
        raise CandidateProgramError("CANDIDATE must be a dictionary")
    return deepcopy(payload)


def render_initial_program(
    catalog: Sequence[Mapping[str, Any]],
    *,
    min_mutations: int,
    max_mutations: int,
) -> str:
    public_catalog = [
        {
            "mutation_id": str(row["mutation_id"]),
            "knob_id": str(row.get("knob_id", "")),
            "knob_spec": str(row.get("knob_spec", "")),
            "family": str(row.get("family", "")),
            "stage": str(row.get("stage", "")),
        }
        for row in catalog
    ]
    return f'''"""Bounded TeAAL candidate proposed by a CHIA search strategy."""

from copy import deepcopy

MUTATION_CATALOG = {pprint.pformat(public_catalog, width=100, sort_dicts=True)}
VALID_MUTATION_IDS = tuple(row["mutation_id"] for row in MUTATION_CATALOG)
MIN_MUTATIONS = {min_mutations}
MAX_MUTATIONS = {max_mutations}

# Search contract: edit only the literal CANDIDATE value inside this block.
# Select MIN_MUTATIONS..MAX_MUTATIONS unique ids. Do not alter the catalog.
# EVOLVE-BLOCK-START
CANDIDATE = {{
    "label": "seed",
    "mutation_ids": [],
}}
# EVOLVE-BLOCK-END


def propose_candidate():
    return deepcopy(CANDIDATE)


def get_candidate():
    return propose_candidate()
'''
