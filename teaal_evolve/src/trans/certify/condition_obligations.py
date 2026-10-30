

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:                                                                          
    from sympy import sstr                
except Exception:                    
    sstr = str                

from src.ir.cond import EqualCondition, LessThanEqualCondition

from .candidate import CandidateArtifacts
from .obligation import LocalLegalityObligation


@dataclass(frozen=True)
class SerializedCondition:
    

    kind: str
    operator: str
    lhs: str
    rhs: str
    text: str
    supported: bool


def serialize_condition(condition: Any) -> SerializedCondition:
    

    if isinstance(condition, EqualCondition):
        lhs = _render_expr(condition.get_lhs())
        rhs = _render_expr(condition.get_rhs())
        text = f"eq({lhs},{rhs})"
        return SerializedCondition(
            kind="symbolic_condition_equal",
            operator="==",
            lhs=lhs,
            rhs=rhs,
            text=text,
            supported=True,
        )

    if isinstance(condition, LessThanEqualCondition):
        lhs = _render_expr(condition.get_lhs())
        rhs = _render_expr(condition.get_rhs())
        text = f"leq({lhs},{rhs})"
        return SerializedCondition(
            kind="symbolic_condition_leq",
            operator="<=",
            lhs=lhs,
            rhs=rhs,
            text=text,
            supported=True,
        )

    text = f"unsupported({type(condition).__name__})"
    return SerializedCondition(
        kind="symbolic_condition_unknown",
        operator="?",
        lhs="",
        rhs="",
        text=text,
        supported=False,
    )


def extract_symbolic_condition_obligations(
    artifacts: CandidateArtifacts,
) -> List[LocalLegalityObligation]:
    

    obligations: List[LocalLegalityObligation] = []

    for source_name, source in _condition_sources(artifacts):
        conditions = _safe_get_conds(source)
        seen: set[str] = set()
        local_index = 0

        for condition in conditions:
            serialized = serialize_condition(condition)
            fingerprint = _fingerprint(serialized.text)
            dedup_key = f"{serialized.kind}:{serialized.text}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            status = "pass" if serialized.supported else "fail"
            detail = (
                f"Supported symbolic condition extracted from {source_name}: "
                f"{serialized.lhs} {serialized.operator} {serialized.rhs}"
                if serialized.supported
                else f"Unsupported symbolic condition extracted from {source_name}: {serialized.text}"
            )

            obligation_id = f"cond:{source_name}:{local_index}:{fingerprint}"
            obligations.append(
                LocalLegalityObligation(
                    obligation_id=obligation_id,
                    knob_id=f"condition:{source_name}:{local_index}:{fingerprint}",
                    kind=serialized.kind,
                    stage_name=source_name,
                    current_value=serialized.text,
                    allowed_values=[serialized.text],
                    status=status,
                    detail=detail,
                )
            )
            local_index += 1

    return obligations


def count_symbolic_condition_obligations(obligations: Sequence[LocalLegalityObligation]) -> int:
    

    return sum(1 for obligation in obligations if obligation.kind.startswith("symbolic_condition_"))


def _condition_sources(artifacts: CandidateArtifacts) -> Iterable[Tuple[str, object]]:
    

    candidates: List[Tuple[str, Optional[object]]] = [
        ("einsum", artifacts.elaborated_einsum),
        ("mapping", artifacts.elaborated_mapping),
        ("format", artifacts.elaborated_format),
        ("binding", artifacts.elaborated_binding),
    ]

    for name, source in candidates:
        if source is not None and callable(getattr(source, "get_conds", None)):
            yield name, source


def _safe_get_conds(source: object) -> List[Any]:
    

    get_conds = getattr(source, "get_conds", None)
    if not callable(get_conds):
        return []

    conditions = get_conds()
    if conditions is None:
        return []
    if isinstance(conditions, list):
        return list(conditions)
    return list(conditions)


def _render_expr(expr: Any) -> str:
    

    try:
        return str(sstr(expr))
    except Exception:                                          
        return str(expr)


def _fingerprint(text: str) -> str:
    

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
