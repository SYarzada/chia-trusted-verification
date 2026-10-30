

from sympy import Basic                
from typing import Any, List, Tuple

from src.ir.common import *
from src.ir.einsum import *


class PartitionedRank(Rank):
    

    def __init__(
            self,
            einsum: str,
            name: str,
            abs_shapes: Tuple[Basic, ...],
            rel_shapes: Tuple[Basic, ...],
            roots: Tuple[DeclaredRank, ...]):
        self.einsum = einsum
        self.name = name
        self.abs_shapes = abs_shapes
        self.rel_shapes = rel_shapes
        self.roots = roots

    def get_abs_shapes(self) -> Tuple[Basic, ...]:
        
        return self.abs_shapes

    def get_einsum(self) -> str:
        
        return self.einsum

    def get_name(self) -> str:
        
        return self.name

    def get_rel_shapes(self) -> Tuple[Basic, ...]:
        
        return self.rel_shapes

    def get_roots(self) -> Tuple[DeclaredRank, ...]:
        
        return self.roots

    def get_shape(self) -> Basic:
        
        shape = 1
        for rel_shape in self.rel_shapes:
            shape *= rel_shape
        return shape

    @staticmethod
    def scoped_ranks(part_ranks: List["PartitionedRank"]) -> List[str]:
        
        if not part_ranks:
            return []

        prefix = part_ranks[0].get_einsum() + "."
        return [prefix + part_rank.get_name() for part_rank in part_ranks]

    def to_str_repr(self) -> StringRank:
        
        return StringRank(self.name)

    def key(self) -> Tuple[Any, ...]:
        return self.einsum, self.name, self.abs_shapes, self.rel_shapes, self.roots

    def __hash__(self) -> int:
        
        return hash(("PartitionedRank", self.einsum, self.name))
