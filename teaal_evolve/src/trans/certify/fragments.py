

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .obligation import LocalLegalityObligation


class Expr:
    pass


@dataclass(frozen=True)
class Symbol(Expr):
    name: str


@dataclass(frozen=True)
class IntConst(Expr):
    value: int


@dataclass(frozen=True)
class TupleConst(Expr):
    values: Tuple[str, ...]


@dataclass(frozen=True)
class App(Expr):
    op: str
    args: Tuple[Expr, ...]


def build_local_legality_fragment(obligation: LocalLegalityObligation) -> App:
    if obligation.kind == "tensor_component_group":
        return App(
            op="tensor_component_group_obligation",
            args=(
                Symbol(obligation.tensor_name),
                Symbol(obligation.current_value),
                TupleConst(tuple(obligation.allowed_values)),
                IntConst(len(obligation.binding_tensor_indices)),
                Symbol(obligation.status),
                Symbol(_classify_legality_detail(obligation.detail)),
            ),
        )

    op = obligation.kind.replace(".", "_") + "_obligation"
    anchor = obligation.tensor_name or obligation.stage_name or obligation.consumer_name or "!ANCHOR"
    return App(
        op=op,
        args=(
            Symbol(anchor),
            Symbol(obligation.current_value),
            TupleConst(tuple(obligation.allowed_values)),
            Symbol(obligation.status),
            Symbol(_classify_legality_detail(obligation.detail)),
        ),
    )


def build_local_legality_request_fragment(obligation: LocalLegalityObligation) -> App:
    

    if obligation.kind == "tensor_component_group":
        return App(
            op="tensor_component_group_obligation",
            args=(
                Symbol(obligation.tensor_name),
                Symbol(obligation.current_value),
                TupleConst(tuple(obligation.allowed_values)),
                IntConst(len(obligation.binding_tensor_indices)),
            ),
        )

    op = obligation.kind.replace(".", "_") + "_obligation"
    anchor = (
        obligation.tensor_name
        or obligation.stage_name
        or obligation.consumer_name
        or "!ANCHOR"
    )
    return App(
        op=op,
        args=(
            Symbol(anchor),
            Symbol(obligation.current_value),
            TupleConst(tuple(obligation.allowed_values)),
        ),
    )


def render_expr(expr: Expr) -> str:
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, IntConst):
        return str(expr.value)
    if isinstance(expr, TupleConst):
        return "[" + ",".join(expr.values) + "]"
    if isinstance(expr, App):
        return f"{expr.op}(" + ",".join(render_expr(arg) for arg in expr.args) + ")"
    raise TypeError(f"Unsupported Expr type: {type(expr)!r}")


def _classify_legality_detail(detail: str) -> str:
    lowered = detail.lower()
    if "locally consistent" in lowered or "use an allowed component value" in lowered:
        return "pass_group_consistent"
    if "not allowed" in lowered:
        return "fail_disallowed_component"
    if "inconsistent" in lowered:
        return "fail_inconsistent_group"
    if "missing" in lowered:
        return "fail_missing_binding_entry"
    if "duplicate" in lowered:
        return "fail_duplicate_ranks"
    if "do not cover" in lowered or "does not match" in lowered:
        return "fail_rank_coverage"
    if "syntactically well-formed" in lowered or "covers" in lowered:
        return "pass_structural"
    return "other"
