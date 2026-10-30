

from sympy import Basic, Symbol                
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from src.common import *
from src.ir.common.access.const_term import ConstantTerm
from src.ir.common.access.term import Term
from src.ir.common.access.var_term import VariableTerm
from src.ir.common.rank import Rank
from src.ir.solver import *


R = TypeVar("R")


class RankAccess(Base, Generic[R]):
    

    def __init__(self, expr: List[Term[R]]):
        self.expr = expr

    def gen(self) -> str:
        
        return " + ".join(term.gen() for term in self.expr)

    def get_expr(self) -> List[Term[R]]:
        
        return self.expr

    @staticmethod
    def merge(accesses: List["RankAccess[R]"],
              get_rank_expression: Callable[[R],
                                            Optional[Tuple[Basic,
                                                           Dict[Symbol,
                                                                R]]]],
              location: str) -> Tuple["RankAccess[R]",
                                      "RankAccess[R]"]:
        
                                                        
        ranges = []
        shapes = []
        rank_var_shapes = {}
        for access in accesses:
            start = 0
            end = 0
            for term in access.get_expr():
                                                                             
                       
                if isinstance(term, ConstantTerm):
                    start += term.get()
                    end += term.get()
                    continue

                assert isinstance(term, VariableTerm)

                shapes.append(term.get().get_shape())

                                                                      
                                         
                sympy_info = get_rank_expression(term.get())
                if sympy_info:
                    start += sympy_info[0]
                    end += sympy_info[0]

                    for rank_var, rank in sympy_info[1].items():
                        rank_var_shapes[str(rank_var)] = rank.get_shape()

                                                      
                else:
                    end += term.get().get_shape() - 1

            ranges.append((start, end))

                                                         
        start_i, end_i = RangeSolver.get_range(
            ranges, shapes, rank_var_shapes, location)

                                                                     
        final_range = []
        for access in [accesses[start_i], accesses[end_i]]:
            terms: List[Term] = []
            for term in access.get_expr():
                if isinstance(term, ConstantTerm):
                    terms.append(term)
                    continue

                assert isinstance(term, VariableTerm)

                sympy_info = get_rank_expression(term.get())
                if sympy_info:
                    terms.append(ConstantTerm(sympy_info[0]))

                else:
                    terms.append(term)

            final_range.append(RankAccess(terms))

        return (final_range[0], final_range[1])

    def to_sympy(self) -> Basic:
        
        sympy_expr = 0
        for term in self.get_expr():
            if isinstance(term, ConstantTerm):
                sympy_expr += term.get()
                continue

            assert isinstance(term, VariableTerm)

            sympy_expr += Symbol(term.get().get_name().lower())

        return sympy_expr

    def key(self) -> Tuple[Any, ...]:
        return self.expr,
