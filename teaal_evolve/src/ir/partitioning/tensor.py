

from typing import Any, Dict, List, Set, Tuple

from src.ir.common import *
from src.ir.einsum import *
from src.ir.partitioning.partitioning import Partitioning
from src.ir.partitioning.rank import PartitionedRank


class PartitionedTensor(Tensor):
    

    def __init__(
            self,
            decl_tensor: DeclaredTensor,
            partitioning: Partitioning
    ):
        self.decl_tensor = decl_tensor
        self.partitioning = partitioning
        self.einsum = self.partitioning.get_einsum().get_name()
        self.ranks = self.partitioning.apply_to_ranks(
            self.decl_tensor.get_ranks())

    def get_name(self) -> str:
        
        return self.decl_tensor.get_name()

    def get_decl_tensor(self) -> DeclaredTensor:
        
        return self.decl_tensor

    def get_einsum(self) -> str:
        
        return self.einsum

    def get_partitioning(self) -> Partitioning:
        
        return self.partitioning

    def get_ranks(self) -> List[PartitionedRank]:
        
        return self.ranks

    def to_str_repr(self) -> StringTensor:
        
        ranks = [rank.to_str_repr() for rank in self.ranks]

        return StringTensor(self.get_name(), ranks)

    def key(self) -> Tuple[Any, ...]:
        return self.decl_tensor, self.ranks

    def __hash__(self) -> int:
        
        return hash(("PartitionedTensor", self.einsum, self.get_name()))
