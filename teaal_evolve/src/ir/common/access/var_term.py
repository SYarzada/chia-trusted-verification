
from sympy import Basic                
from typing import Any, Tuple, TypeVar

from src.ir.common.access.term import Term
from src.ir.common.rank import Rank


R = TypeVar("R")


class VariableTerm(Term[R]):
    

    def __init__(self, rank: R):
        self.rank = rank

    def gen(self) -> str:
        
        return self.rank.get_name().lower()

    def get(self) -> R:
        
        return self.rank

    def key(self) -> Tuple[Any, ...]:
        
        return self.rank,
