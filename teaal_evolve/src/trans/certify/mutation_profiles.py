from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence, Tuple


SEVERITY_MILD = "mild"
SEVERITY_SEVERE = "severe"

FOCUS_BALANCED = "balanced"
FOCUS_OBJECTIVE = "objective"

BUCKET_LEGALITY = "legality"
BUCKET_OBJECTIVE = "objective"

PURPOSE_SEARCH = "search"
PURPOSE_SEARCH_SURROGATE_LEGACY = "search_surrogate_legacy"
PURPOSE_SEARCH_MONOLITHIC = "search_monolithic"
PURPOSE_CHECKER_STRESS = "checker_stress"

EXPECTED_UNKNOWN = "unknown"
EXPECTED_LEGAL = "legal"
EXPECTED_ILLEGAL = "illegal"

OBJECTIVE_UNKNOWN = "unknown"
OBJECTIVE_IMPROVING = "improving"
OBJECTIVE_WORSENING = "worsening"
OBJECTIVE_NEUTRAL = "neutral"
OBJECTIVE_UNAVAILABLE = "unavailable"

FOCUS_CHECKER_STRESS = "checker_stress"


@dataclass(frozen=True)
class MutationOption:
    knob_spec: str
    severity: str
    family: str
    stage: str = ""
    note: str = ""
    bucket: str = BUCKET_LEGALITY
    purposes: Tuple[str, ...] = (PURPOSE_SEARCH, PURPOSE_CHECKER_STRESS)
    expected_legality: str = EXPECTED_UNKNOWN
    objective_direction: str = OBJECTIVE_UNKNOWN
    evidence_source: str = ""


@dataclass(frozen=True)
class MutationProfile:
    profile_name: str
    kernel_name: str
    num_changes: int
    num_severe: int
    knob_sequence: List[str]
    focus: str = FOCUS_BALANCED
    purpose: str | None = None


@dataclass(frozen=True)
class ProfileScheduleEntry:
    num_changes: int
    num_severe: int
    weight: float
    focus: str = FOCUS_BALANCED


UNARY_OPTIONS: List[MutationOption] = [
    # ------------------------------------------------------------------
    # Legality / balanced profile.
    # ------------------------------------------------------------------
    MutationOption(
        "binding.tensor.component::ZS=PSUM",
        SEVERITY_MILD,
        "component",
        "ZS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "binding.tensor.component::AS=PSUM",
        SEVERITY_MILD,
        "component",
        "AS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "mapping.loop_order::ZS=[M,N]",
        SEVERITY_MILD,
        "loop_order",
        "ZS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::ZH=[M,N]",
        SEVERITY_MILD,
        "format",
        "ZS->ZH",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::ZH=[N]",
        SEVERITY_SEVERE,
        "format",
        "ZS->ZH",
        bucket=BUCKET_LEGALITY,
    ),

    # ------------------------------------------------------------------
    # Objective-sensitive profile.
    # ------------------------------------------------------------------
    MutationOption(
        "mapping.spacetime::ZS=space=[M];time=[N]",
        SEVERITY_MILD,
        "spacetime",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "mapping.spacetime::AS=space=[M];time=[N]",
        SEVERITY_MILD,
        "spacetime",
        "AS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::ZS=HBM",
        SEVERITY_SEVERE,
        "component",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::AS=HBM",
        SEVERITY_SEVERE,
        "component",
        "AS",
        bucket=BUCKET_OBJECTIVE,
    ),
]


UNARY_REDUCE_OPTIONS: List[MutationOption] = [
    # ------------------------------------------------------------------
    # Legality / balanced profile.
    # ------------------------------------------------------------------
    MutationOption(
        "binding.tensor.component::RS=PSUM",
        SEVERITY_MILD,
        "component",
        "RS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "binding.tensor.component::ZS=PSUM",
        SEVERITY_MILD,
        "component",
        "ZS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "mapping.loop_order::RS=[K,N,M]",
        SEVERITY_MILD,
        "loop_order",
        "RS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::RS=[K,N,M]",
        SEVERITY_MILD,
        "format",
        "ZS->RS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::RS=[M,N]",
        SEVERITY_SEVERE,
        "format",
        "ZS->RS",
        bucket=BUCKET_LEGALITY,
    ),

    # ------------------------------------------------------------------
    # Objective-sensitive profile.
    # ------------------------------------------------------------------
    MutationOption(
        "mapping.spacetime::RS=space=[K];time=[M,N]",
        SEVERITY_MILD,
        "spacetime",
        "RS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "mapping.spacetime::ZS=space=[K];time=[M,N]",
        SEVERITY_MILD,
        "spacetime",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::RS=HBM",
        SEVERITY_SEVERE,
        "component",
        "RS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::ZS=HBM",
        SEVERITY_SEVERE,
        "component",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
]


VECTOR_SCALAR_OVERLAP_OPTIONS: List[MutationOption] = [
    # ------------------------------------------------------------------
    # Legality / balanced profile.
    # ------------------------------------------------------------------
    MutationOption(
        "binding.tensor.component::BS=SBUF",
        SEVERITY_MILD,
        "component",
        "BS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "binding.tensor.component::ZS=PSUM",
        SEVERITY_MILD,
        "component",
        "ZS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "mapping.loop_order::BS=[P,M,N,Q]",
        SEVERITY_MILD,
        "loop_order",
        "BS",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::ZH=[N,M,PI,Q]",
        SEVERITY_MILD,
        "format",
        "ZS->ZH",
        bucket=BUCKET_LEGALITY,
    ),
    MutationOption(
        "format.rank_order::ZS::ZH=[Q,PI,M]",
        SEVERITY_SEVERE,
        "format",
        "ZS->ZH",
        bucket=BUCKET_LEGALITY,
    ),

    # ------------------------------------------------------------------
    # Objective-sensitive profile.
    #
    # Important: These are chosen based on insights from testing.
    # The hierarchical score changed for BS=SBUF and ZS=PSUM, while the
    # spacetime BS mutation tested did not move the score.
    # ------------------------------------------------------------------
    MutationOption(
        "binding.tensor.component::BS=SBUF",
        SEVERITY_MILD,
        "component",
        "BS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::ZS=PSUM",
        SEVERITY_MILD,
        "component",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::AT=SBUF",
        SEVERITY_MILD,
        "component",
        "AT",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "format.rank_order::ZS::ZH=[Q,PI,M]",
        SEVERITY_SEVERE,
        "format",
        "ZS->ZH",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "format.rank_order::AT::BS=[Q,P,M]",
        SEVERITY_SEVERE,
        "format",
        "AT->BS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::BS=HBM",
        SEVERITY_SEVERE,
        "component",
        "BS",
        bucket=BUCKET_OBJECTIVE,
    ),
    MutationOption(
        "binding.tensor.component::ZS=HBM",
        SEVERITY_SEVERE,
        "component",
        "ZS",
        bucket=BUCKET_OBJECTIVE,
    ),
]


def _evidence_option(
    knob_spec: str,
    family: str,
    stage: str,
    *,
    severity: str = SEVERITY_MILD,
    expected_legality: str,
    objective_direction: str,
    purposes: Tuple[str, ...] = (PURPOSE_SEARCH, PURPOSE_CHECKER_STRESS),
    note: str = "Measured in the exhaustive single-mutation characterization.",
) -> MutationOption:
    return MutationOption(
        knob_spec=knob_spec,
        severity=severity,
        family=family,
        stage=stage,
        note=note,
        bucket=(BUCKET_OBJECTIVE if PURPOSE_SEARCH in purposes else BUCKET_LEGALITY),
        purposes=purposes,
        expected_legality=expected_legality,
        objective_direction=objective_direction,
        evidence_source="mutation_characterization_new_kernels_v2",
    )


ATTENTION_OPTIONS: List[MutationOption] = [
    # Evidence-derived search operators from the exhaustive single-mutation
    # characterization. Search profiles deliberately span improvements,
    # degradations, neutral moves, and a small number of illegal stress cases.
    _evidence_option(
        "binding.tensor.component::QKS=PSUM", "component", "QKS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::SN=PSUM", "component", "SN",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::GM=PSUM", "component", "GM",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::SDR=PSUM", "component", "SDR",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::SNTS=PSUM", "component", "SNTS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::VTS=PSUM", "component", "VTS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::QKP=SBUF", "component", "QKP",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "binding.tensor.component::SNTP=SBUF", "component", "SNTP",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::VTP=space=[F];time=[M2,M1,M0]", "spacetime", "VTP",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::QS=space=[P1P0];time=[P3P2,E]", "spacetime", "QS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::SDR=space=[P0];time=[P3P2,P1]", "spacetime", "SDR",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::QKP=space=[M1M0];time=[P3P2,M2,P0,E,P1]",
        "spacetime", "QKP",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::AV=space=[F];time=[P3P2,P0,P1]", "spacetime", "AV",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "mapping.spacetime::SNTS=space=[P1];time=[P3P2,M2,M1,P0,M0]",
        "spacetime", "SNTS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "mapping.spacetime::VTS=space=[F];time=[M2,M1,M0]", "spacetime", "VTS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    # Representative illegal cases retained for legal-yield/filtering metrics.
    _evidence_option(
        "binding.tensor.component::QS=PSUM", "component", "QS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "mapping.loop_order::KS", "loop_order", "KS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "format.rank_order::QS::QS", "format", "QS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
]


def _one_pass_attention_options() -> List[MutationOption]:
    return [
        _evidence_option(
            "binding.tensor.component::RM=PSUM", "component", "RM",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::RD=PSUM", "component", "RD",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::RNV=PSUM", "component", "RNV",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::BQKS=PSUM", "component", "BQKS",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::PRM=PSUM", "component", "PRM",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::SLN=PSUM", "component", "SLN",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_IMPROVING,
        ),
        _evidence_option(
            "binding.tensor.component::BQKP=SBUF", "component", "BQKP",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "binding.tensor.component::SLNTP=SBUF", "component", "SLNTP",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::QS=space=[P3P2P1P0];time=[P4,E]",
            "spacetime", "QS",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::KS=space=[M2M1M0];time=[P4,M3,E]",
            "spacetime", "KS",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::RDR=space=[P3P2P0];time=[P4,P1]",
            "spacetime", "RDR",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::RM_update=space=[P0];time=[P4,M3,M2P3P2,P1]",
            "spacetime", "RM_update",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::BQKP=space=[M1M0];time=[P4,M3,M2P3P2,P0,E,P1]",
            "spacetime", "BQKP",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_WORSENING,
        ),
        _evidence_option(
            "mapping.spacetime::AVH=space=[F];time=[P4,P3P2,P0,P1]",
            "spacetime", "AVH",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_NEUTRAL,
        ),
        _evidence_option(
            "mapping.spacetime::SLNTS=space=[P1];time=[P4,M3,M2P3P2,P0,M1,M0]",
            "spacetime", "SLNTS",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_NEUTRAL,
        ),
        _evidence_option(
            "mapping.spacetime::SLNTVS=space=[F];time=[P4,M3,M2P3P2,P0,P1]",
            "spacetime", "SLNTVS",
            expected_legality=EXPECTED_LEGAL,
            objective_direction=OBJECTIVE_NEUTRAL,
        ),
        _evidence_option(
            "binding.tensor.component::QS=PSUM", "component", "QS",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_ILLEGAL,
            objective_direction=OBJECTIVE_UNAVAILABLE,
        ),
        _evidence_option(
            "mapping.loop_order::KS", "loop_order", "KS",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_ILLEGAL,
            objective_direction=OBJECTIVE_UNAVAILABLE,
        ),
        _evidence_option(
            "format.rank_order::QS::QS", "format", "QS",
            severity=SEVERITY_SEVERE,
            expected_legality=EXPECTED_ILLEGAL,
            objective_direction=OBJECTIVE_UNAVAILABLE,
        ),
    ]


ONE_PASS_ATTENTION_OPTIONS = _one_pass_attention_options()
ONE_PASS_ATTENTION_NO_PIPELINING_OPTIONS = _one_pass_attention_options()


MATMUL_OPTIONS: List[MutationOption] = [
    _evidence_option(
        "mapping.spacetime::BS=space=[N0];time=[M1,N1,K1,K01,K00]",
        "spacetime", "BS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::ZP=SBUF", "component", "ZP",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::ZH=space=[N0];time=[M1,N1,M01,M00]",
        "spacetime", "ZH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::ZS_init=space=[N0];time=[M1,N1,M01,M00]",
        "spacetime", "ZS_init",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::AS=space=[M0];time=[M1,N1,K1,K01,K00]",
        "spacetime", "AS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::AH=space=[M1M0];time=[K1K0]", "spacetime", "AH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::BH=space=[N1N0];time=[K1K0]", "spacetime", "BH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::Z=space=[N1N0];time=[M1M0]", "spacetime", "Z",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::ZS=space=[N00];time=[M1,N1,K1,M01,N01,M00]",
        "spacetime", "ZS",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "mapping.spacetime::ZP=space=[N00];time=[M1,N1,K1,M01,N01,K01,K00,M00]",
        "spacetime", "ZP",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "binding.tensor.component::AS=PSUM", "component", "AS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "mapping.loop_order::ZS", "loop_order", "ZS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "format.rank_order::ZP::ZS", "format", "ZP->ZS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
]


MAMBA_OPTIONS: List[MutationOption] = [
    _evidence_option(
        "mapping.spacetime::BH=space=[N];time=[B,T1T0]", "spacetime", "BH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::DH=space=[D1D0];time=[B,T1T0]", "spacetime", "DH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::Z=space=[D1D0];time=[B,T1T0]", "spacetime", "Z",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::H=PSUM", "component", "H",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::AH=space=[N];time=[D1D0]", "spacetime", "AH",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::EDA=PSUM", "component", "EDA",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::DT=PSUM", "component", "DT",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "binding.tensor.component::DUB=PSUM", "component", "DUB",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_IMPROVING,
    ),
    _evidence_option(
        "mapping.spacetime::DT=space=[T0];time=[B,T1,D11,D10,D0]",
        "spacetime", "DT",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::DP=space=[D0];time=[B,T1,D11,D10,T0]",
        "spacetime", "DP",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::I_init=space=[D11ND10];time=[B,D0]",
        "spacetime", "I_init",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::I_update=space=[D10];time=[B,T1,D11,N,D0]",
        "spacetime", "I_update",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::AS=space=[N];time=[B,D1,D0]", "spacetime", "AS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::BS=space=[N];time=[B,T1,T0]", "spacetime", "BS",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_WORSENING,
    ),
    _evidence_option(
        "mapping.spacetime::DA=space=[T0];time=[B,T1,D11,N,D10,D0]",
        "spacetime", "DA",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "mapping.spacetime::HC=space=[D0];time=[B,T1,D11,N,D10,T0]",
        "spacetime", "HC",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "mapping.spacetime::ZS_update=space=[D0];time=[B,T1,D11,N,D10,T0]",
        "spacetime", "ZS_update",
        expected_legality=EXPECTED_LEGAL,
        objective_direction=OBJECTIVE_NEUTRAL,
    ),
    _evidence_option(
        "binding.tensor.component::H=HBM", "component", "H",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "mapping.loop_order::ZS_update", "loop_order", "ZS_update",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
    _evidence_option(
        "format.rank_order::EDA::EDA", "format", "EDA",
        severity=SEVERITY_SEVERE,
        expected_legality=EXPECTED_ILLEGAL,
        objective_direction=OBJECTIVE_UNAVAILABLE,
    ),
]

# Profile keys use evaluation kernel labels.
KERNEL_MUTATION_OPTIONS: Dict[str, List[MutationOption]] = {
    "unary": UNARY_OPTIONS,
    "unary_reduce_one_tile": UNARY_REDUCE_OPTIONS,
    "vector_scalar_overlap": VECTOR_SCALAR_OVERLAP_OPTIONS,
    "attention": ATTENTION_OPTIONS,
    "one_pass_attention": ONE_PASS_ATTENTION_OPTIONS,
    "one_pass_attention_no_pipelining": ONE_PASS_ATTENTION_NO_PIPELINING_OPTIONS,
    "attn": ATTENTION_OPTIONS,
    "matmul": MATMUL_OPTIONS,
    "mamba": MAMBA_OPTIONS,
}


_DEFAULT_BALANCED = [
    ProfileScheduleEntry(num_changes=1, num_severe=0, weight=0.30, focus=FOCUS_BALANCED),
    ProfileScheduleEntry(num_changes=2, num_severe=0, weight=0.30, focus=FOCUS_BALANCED),
    ProfileScheduleEntry(num_changes=2, num_severe=1, weight=0.30, focus=FOCUS_BALANCED),
    ProfileScheduleEntry(num_changes=3, num_severe=1, weight=0.10, focus=FOCUS_BALANCED),
]

_DEFAULT_SEARCH = [
    ProfileScheduleEntry(num_changes=1, num_severe=0, weight=0.50, focus=FOCUS_OBJECTIVE),
    ProfileScheduleEntry(num_changes=2, num_severe=0, weight=0.35, focus=FOCUS_OBJECTIVE),
    ProfileScheduleEntry(num_changes=2, num_severe=1, weight=0.15, focus=FOCUS_OBJECTIVE),
]

_DEFAULT_CHECKER_STRESS = [
    ProfileScheduleEntry(num_changes=1, num_severe=0, weight=0.25, focus=FOCUS_CHECKER_STRESS),
    ProfileScheduleEntry(num_changes=1, num_severe=1, weight=0.25, focus=FOCUS_CHECKER_STRESS),
    ProfileScheduleEntry(num_changes=2, num_severe=1, weight=0.35, focus=FOCUS_CHECKER_STRESS),
    ProfileScheduleEntry(num_changes=3, num_severe=1, weight=0.15, focus=FOCUS_CHECKER_STRESS),
]

DEFAULT_PROFILE_SCHEDULES: Dict[str, List[ProfileScheduleEntry]] = {
    key: list(_DEFAULT_BALANCED) for key in KERNEL_MUTATION_OPTIONS
}
OBJECTIVE_PROFILE_SCHEDULES: Dict[str, List[ProfileScheduleEntry]] = {
    key: list(_DEFAULT_SEARCH) for key in KERNEL_MUTATION_OPTIONS
}
CHECKER_STRESS_PROFILE_SCHEDULES: Dict[str, List[ProfileScheduleEntry]] = {
    key: list(_DEFAULT_CHECKER_STRESS) for key in KERNEL_MUTATION_OPTIONS
}


def resolve_profile_key(kernel_label: str, kernel_name: str) -> str:
    """Prefer a label-specific profile, falling back to options.kernel-name."""
    if kernel_label in KERNEL_MUTATION_OPTIONS:
        return kernel_label
    if kernel_name in KERNEL_MUTATION_OPTIONS:
        return kernel_name
    raise ValueError(
        f"No mutation profile for kernel label {kernel_label!r} "
        f"or kernel name {kernel_name!r}"
    )


def get_mutation_options(
    kernel_name: str,
    *,
    purpose: str | None = None,
) -> List[MutationOption]:
    if kernel_name not in KERNEL_MUTATION_OPTIONS:
        raise ValueError(f"Unsupported kernel_name for mutation profiles: {kernel_name}")
    options = list(KERNEL_MUTATION_OPTIONS[kernel_name])
    if purpose is None:
        return options
    if purpose not in {
        PURPOSE_SEARCH,
        PURPOSE_SEARCH_SURROGATE_LEGACY,
        PURPOSE_SEARCH_MONOLITHIC,
        PURPOSE_CHECKER_STRESS,
    }:
        raise ValueError(f"Unsupported mutation profile purpose: {purpose}")

    if purpose == PURPOSE_SEARCH_SURROGATE_LEGACY:
        purpose = PURPOSE_SEARCH
    return [option for option in options if purpose in option.purposes]


def _choose_from_options(
    options: List[MutationOption],
    *,
    count: int,
    preferred_bucket: str | None = None,
) -> List[MutationOption]:
    if count <= 0:
        return []

    if preferred_bucket is None:
        if count > len(options):
            raise ValueError("Not enough options available")
        return options[:count]

    preferred = [opt for opt in options if opt.bucket == preferred_bucket]
    other = [opt for opt in options if opt.bucket != preferred_bucket]

    chosen: List[MutationOption] = []
    chosen.extend(preferred[:count])
    if len(chosen) < count:
        chosen.extend(other[: count - len(chosen)])

    if len(chosen) < count:
        raise ValueError("Not enough options available")

    return chosen


def purpose_for_focus(focus: str) -> str:
    if focus == FOCUS_OBJECTIVE:
        return PURPOSE_SEARCH
    if focus in {FOCUS_BALANCED, FOCUS_CHECKER_STRESS}:
        return PURPOSE_CHECKER_STRESS
    raise ValueError(f"Unsupported mutation profile focus: {focus}")


def build_profile_from_options(
    kernel_name: str,
    options: Sequence[MutationOption],
    num_changes: int,
    num_severe: int,
    *,
    focus: str = FOCUS_BALANCED,
    purpose: str | None = None,
) -> MutationProfile:
    """Build a profile shape from an explicit, frozen option list."""

    if num_changes <= 0:
        raise ValueError("num_changes must be positive")
    if num_severe < 0:
        raise ValueError("num_severe cannot be negative")
    if num_severe > num_changes:
        raise ValueError("num_severe cannot exceed num_changes")

    mild = [opt for opt in options if opt.severity == SEVERITY_MILD]
    severe = [opt for opt in options if opt.severity == SEVERITY_SEVERE]
    num_mild = num_changes - num_severe
    preferred_bucket = BUCKET_OBJECTIVE if focus == FOCUS_OBJECTIVE else None

    chosen_mild = _choose_from_options(
        mild,
        count=num_mild,
        preferred_bucket=preferred_bucket,
    )
    chosen_severe = _choose_from_options(
        severe,
        count=num_severe,
        preferred_bucket=preferred_bucket,
    )
    chosen = chosen_mild + chosen_severe
    return MutationProfile(
        profile_name=f"{kernel_name}:{num_changes}chg:{num_severe}sev:{focus}",
        kernel_name=kernel_name,
        num_changes=num_changes,
        num_severe=num_severe,
        knob_sequence=[opt.knob_spec for opt in chosen],
        focus=focus,
        purpose=purpose or purpose_for_focus(focus),
    )


def build_profile(
    kernel_name: str,
    num_changes: int,
    num_severe: int,
    *,
    focus: str = FOCUS_BALANCED,
    purpose: str | None = None,
) -> MutationProfile:
    if num_changes <= 0:
        raise ValueError("num_changes must be positive")
    if num_severe < 0:
        raise ValueError("num_severe cannot be negative")
    if num_severe > num_changes:
        raise ValueError("num_severe cannot exceed num_changes")

    resolved_purpose = purpose or purpose_for_focus(focus)
    options = get_mutation_options(kernel_name, purpose=resolved_purpose)
    return build_profile_from_options(
        kernel_name,
        options,
        num_changes,
        num_severe,
        focus=focus,
        purpose=resolved_purpose,
    )


def get_default_profile_schedule(kernel_name: str) -> List[ProfileScheduleEntry]:
    if kernel_name not in DEFAULT_PROFILE_SCHEDULES:
        raise ValueError(f"Unsupported kernel_name for profile schedule: {kernel_name}")
    return list(DEFAULT_PROFILE_SCHEDULES[kernel_name])


def get_objective_profile_schedule(kernel_name: str) -> List[ProfileScheduleEntry]:
    if kernel_name not in OBJECTIVE_PROFILE_SCHEDULES:
        raise ValueError(f"Unsupported kernel_name for objective profile schedule: {kernel_name}")
    return list(OBJECTIVE_PROFILE_SCHEDULES[kernel_name])


def get_search_profile_schedule(kernel_name: str) -> List[ProfileScheduleEntry]:
    return get_objective_profile_schedule(kernel_name)


def get_checker_stress_profile_schedule(kernel_name: str) -> List[ProfileScheduleEntry]:
    if kernel_name not in CHECKER_STRESS_PROFILE_SCHEDULES:
        raise ValueError(
            f"Unsupported kernel_name for checker-stress profile schedule: {kernel_name}"
        )
    return list(CHECKER_STRESS_PROFILE_SCHEDULES[kernel_name])


def sample_profile(
    kernel_name: str,
    schedule: Sequence[ProfileScheduleEntry],
    rng: Random,
    *,
    options: Sequence[MutationOption] | None = None,
    purpose: str | None = None,
) -> MutationProfile:
    if not schedule:
        raise ValueError("Profile schedule must not be empty")

    total_weight = sum(entry.weight for entry in schedule)
    if total_weight <= 0:
        raise ValueError("Profile schedule weights must sum to a positive value")

    pick = rng.random() * total_weight
    cumulative = 0.0
    chosen = schedule[-1]

    for entry in schedule:
        cumulative += entry.weight
        if pick <= cumulative:
            chosen = entry
            break

    if options is None:
        return build_profile(
            kernel_name=kernel_name,
            num_changes=chosen.num_changes,
            num_severe=chosen.num_severe,
            focus=chosen.focus,
            purpose=purpose,
        )

    return MutationProfile(
        profile_name=(
            f"{kernel_name}:{chosen.num_changes}chg:"
            f"{chosen.num_severe}sev:{chosen.focus}"
        ),
        kernel_name=kernel_name,
        num_changes=chosen.num_changes,
        num_severe=chosen.num_severe,
        knob_sequence=[],
        focus=chosen.focus,
        purpose=purpose or purpose_for_focus(chosen.focus),
    )
