

from sympy import Basic                
from typing import Any, Tuple

from src.ir.common import *


class DeclaredRank(Rank):
    

    def __init__(self, name: str, shape: Basic):
        self.name = name
        self.shape = shape

    def get_name(self) -> str:
        
        return self.name

    def get_shape(self) -> Basic:
        
        return self.shape

    def to_str_repr(self) -> StringRank:
        
        return StringRank(self.name)

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.shape

    def __hash__(self) -> int:
        
        return hash(("DeclaredRank", self.name))
