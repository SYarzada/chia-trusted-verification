

from typing import Any, List, Tuple

from src.ir.common import *
from src.ir.einsum.rank import DeclaredRank


class DeclaredTensor(Tensor):
    

    def __init__(self, name: str, ranks: List[DeclaredRank]):
        self.name = name
        self.ranks = ranks
        self.is_cascade_input = False
        self.is_cascade_output = False

    def get_is_cascade_input(self) -> bool:
        
        return self.is_cascade_input

    def get_is_cascade_output(self) -> bool:
        
        return self.is_cascade_output

    def get_name(self) -> str:
        
        return self.name

    def get_ranks(self) -> List[DeclaredRank]:
        
        return self.ranks

    def set_is_cascade_input(self, is_cascade_input: bool):
        
        self.is_cascade_input = is_cascade_input

    def set_is_cascade_output(self, is_cascade_output: bool):
        
        self.is_cascade_output = is_cascade_output

    def to_str_repr(self) -> StringTensor:
        
        ranks = [rank.to_str_repr() for rank in self.ranks]

        return StringTensor(self.name, ranks)

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.ranks

    def __hash__(self) -> int:
        
        return hash(("DeclaredTensor", self.name))

