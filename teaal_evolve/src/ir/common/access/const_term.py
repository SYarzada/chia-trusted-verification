
from sympy import Basic                
from typing import Any, Tuple

from src.ir.common.access.term import Term


class ConstantTerm(Term):
    

    def __init__(self, term: Basic):
        self.term = term

    def gen(self) -> str:
        
        return str(self.term)

    def get(self) -> Basic:
        
        return self.term

    def key(self) -> Tuple[Any, ...]:
        
        return self.term,
