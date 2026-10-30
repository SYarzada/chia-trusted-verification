

from sympy import Basic                
from typing import Any, Tuple

from src.ir.cond.cond import Condition


class EqualCondition(Condition):
    

    def __init__(self, lhs: Basic, rhs: Basic):
        self.lhs = lhs
        self.rhs = rhs

    def get_lhs(self) -> Basic:
        
        return self.lhs

    def get_rhs(self) -> Basic:
        
        return self.rhs

    def key(self) -> Tuple[Any, ...]:
        
        return self.lhs, self.rhs
