"""Trusted TeAAL certification and bounded-mutation API.

The public surface is intentionally small. Historical benchmark, paid-run,
full-spec, and experimental objective exports were removed when the CHIA loop
became the supported orchestration path.
"""

from .candidate import Candidate, CandidateArtifacts, ParserBundle
from .checker_backend import (
    CHECKER_BACKEND_FULL,
    CHECKER_BACKEND_HIERARCHICAL,
    CHECKER_BACKEND_HIERARCHICAL_DELTA_EGRAPH,
    CHECKER_BACKEND_HIERARCHICAL_DELTA_SYNTACTIC,
    CheckerBackendResult,
    check_candidate_with_backend,
)
from .hierarchical_pipeline import certify_candidate_hierarchical
from .knobs import KNOB_SPACE_RESTRICTED, KNOB_SPACE_RICH
from .mutate import apply_knob_specs_atomic
from .mutation_profiles import MutationOption, get_mutation_options
from .objective_backend import (
    OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK,
    compute_objective_with_backend,
)
from .obligation import CandidateReport, ObjectiveResult
from .pipeline import certify_candidate

__all__ = [
    "CHECKER_BACKEND_FULL",
    "CHECKER_BACKEND_HIERARCHICAL",
    "CHECKER_BACKEND_HIERARCHICAL_DELTA_EGRAPH",
    "CHECKER_BACKEND_HIERARCHICAL_DELTA_SYNTACTIC",
    "KNOB_SPACE_RESTRICTED",
    "KNOB_SPACE_RICH",
    "OBJECTIVE_BACKEND_MONOLITHIC_BOTTLENECK",
    "Candidate",
    "CandidateArtifacts",
    "CandidateReport",
    "CheckerBackendResult",
    "MutationOption",
    "ObjectiveResult",
    "ParserBundle",
    "apply_knob_specs_atomic",
    "certify_candidate",
    "certify_candidate_hierarchical",
    "check_candidate_with_backend",
    "compute_objective_with_backend",
    "get_mutation_options",
]
