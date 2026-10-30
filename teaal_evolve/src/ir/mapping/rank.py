

from sympy import Basic                
from typing import Any, Tuple

from src.common import *
from src.ir.common import *
from src.ir.partitioning import *


class MappedRank(Rank):
    

    def __init__(self, rank: PartitionedRank, is_spatial: bool):
        self.rank = rank
        self.is_spatial = is_spatial
        self.is_sequential = False

    def check_concrete(self):
        
        pass

    def check_template(self):
        
        shape = self.get_shape()
        if self.is_spatial and not isinstance(shape, int):
            location = self.rank.get_einsum() + " space"
            message = "spatial ranks in a template must have concrete shapes"
            raise IllegalSymbolicShapeError(shape, location, message)

    def get_einsum(self) -> str:
        
        return self.rank.get_einsum()

    def get_is_sequential(self) -> bool:
        
        return self.is_sequential

    def get_is_spatial(self) -> bool:
        
        return self.is_spatial

    def get_name(self) -> str:
        
        return self.rank.get_name()

    def get_rank(self) -> PartitionedRank:
        
        return self.rank

    def get_scoped_name(self) -> str:
        
        return self.get_einsum() + "." + self.get_name()

    def get_shape(self) -> Basic:
        
        return self.rank.get_shape()

    def set_is_sequential(self, is_sequential: bool):
        
        self.is_sequential = is_sequential

    def to_str_repr(self) -> StringRank:
        
        return StringRank(self.get_name())

    def key(self) -> Tuple[Any, ...]:
        return self.rank, self.is_spatial

    def __hash__(self) -> int:
        
        return hash(("MappedEinsum", self.get_einsum(), self.get_name()))
