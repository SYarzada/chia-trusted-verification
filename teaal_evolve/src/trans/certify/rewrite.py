

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from .fragments import App, Expr, IntConst, Symbol, TupleConst, render_expr


RewriteRule = Callable[[Expr], Optional[Expr]]


def rewrite_forget_tensor_identity(expr: Expr) -> Optional[Expr]:
    

    if not isinstance(expr, App):
        return None
    if expr.op != "tensor_component_group_obligation":
        return None

    if len(expr.args) not in {4, 6}:
        return None
    tensor, component, allowed, arity, *result_fields = expr.args
    placeholder = Symbol("!TENSOR")

    if tensor == placeholder:
        return None

    return App(
        op=expr.op,
        args=(placeholder, component, allowed, arity, *result_fields),
    )


def rewrite_uppercase_component(expr: Expr) -> Optional[Expr]:
    

    if not isinstance(expr, App):
        return None
    if expr.op != "tensor_component_group_obligation":
        return None

    if len(expr.args) not in {4, 6}:
        return None
    tensor, component, allowed, arity, *result_fields = expr.args
    if not isinstance(component, Symbol):
        return None

    normalized = component.name.upper()
    if normalized == component.name:
        return None

    return App(
        op=expr.op,
        args=(tensor, Symbol(normalized), allowed, arity, *result_fields),
    )


def rewrite_sort_allowed_values(expr: Expr) -> Optional[Expr]:
    

    if not isinstance(expr, App):
        return None
    if expr.op != "tensor_component_group_obligation":
        return None

    if len(expr.args) not in {4, 6}:
        return None
    tensor, component, allowed, arity, *result_fields = expr.args
    if not isinstance(allowed, TupleConst):
        return None

    sorted_values = tuple(sorted(allowed.values))
    if sorted_values == allowed.values:
        return None

    return App(
        op=expr.op,
        args=(tensor, component, TupleConst(sorted_values), arity, *result_fields),
    )


DEFAULT_REWRITE_RULES: List[RewriteRule] = [
    rewrite_forget_tensor_identity,
    rewrite_uppercase_component,
    rewrite_sort_allowed_values,
]


class MiniEGraph:
    

    def __init__(self, rules: Optional[List[RewriteRule]] = None) -> None:
        self.rules = list(rules or DEFAULT_REWRITE_RULES)
        self.term_to_id: Dict[Expr, int] = {}
        self.id_to_term: Dict[int, Expr] = {}
        self.parent: List[int] = []

    def add(self, expr: Expr) -> int:
        if expr in self.term_to_id:
            return self.term_to_id[expr]

        expr_id = len(self.parent)
        self.term_to_id[expr] = expr_id
        self.id_to_term[expr_id] = expr
        self.parent.append(expr_id)
        return expr_id

    def find(self, expr_id: int) -> int:
        if self.parent[expr_id] != expr_id:
            self.parent[expr_id] = self.find(self.parent[expr_id])
        return self.parent[expr_id]

    def union_exprs(self, left: Expr, right: Expr) -> None:
        left_id = self.add(left)
        right_id = self.add(right)

        root_left = self.find(left_id)
        root_right = self.find(right_id)

        if root_left != root_right:
            self.parent[root_right] = root_left

    def saturate(self, root: Expr) -> Set[Expr]:
        

        queue: List[Expr] = [root]
        seen: Set[Expr] = {root}
        self.add(root)

        while queue:
            current = queue.pop(0)

            for rule in self.rules:
                rewritten = rule(current)
                if rewritten is None:
                    continue

                self.union_exprs(current, rewritten)

                if rewritten not in seen:
                    seen.add(rewritten)
                    queue.append(rewritten)

        return seen

    def normalize(self, root: Expr) -> Expr:
        

        closure = self.saturate(root)
        root_rep = self.find(self.term_to_id[root])

        eq_class_terms = [
            term
            for term in closure
            if self.find(self.term_to_id[term]) == root_rep
        ]

        return min(eq_class_terms, key=render_expr)


def normalize_expr(
    expr: Expr,
    rules: Optional[List[RewriteRule]] = None,
) -> Expr:
    

    return MiniEGraph(rules=rules).normalize(expr)