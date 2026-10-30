
from sympy import Basic                
from typing import Any, Tuple

from src.ir.common.rank import *


class StringRank(Rank):
    

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        
        return self.name

    def get_shape(self) -> Basic:
        
        raise NotImplementedError("StringRanks do not have shapes")

    def to_str_repr(self) -> "StringRank":
        
        return StringRank(self.name)

    def key(self) -> Tuple[Any, ...]:
        return self.name,
