
from typing import Any, List, Tuple

from src.ir.common.string.rank import StringRank
from src.ir.common.tensor import *


class StringTensor(Tensor):
    

    def __init__(self, name: str, ranks: List[StringRank]):
        self.name = name
        self.ranks = ranks

    def get_name(self) -> str:
        
        return self.name

    def get_ranks(self) -> List[StringRank]:
        
        return self.ranks

    def to_str_repr(self) -> "StringTensor":
        
        return StringTensor(self.name, self.ranks)

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.ranks
