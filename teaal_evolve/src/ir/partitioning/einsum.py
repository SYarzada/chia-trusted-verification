

from lark.tree import Tree
from sympy import Symbol                
from typing import Any, cast, Dict, List, Set, Tuple, Union

from src.common import *
from src.ir.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.partitioning.rank import PartitionedRank
from src.ir.partitioning.tensor import PartitionedTensor
from src.ir.partitioning.tensor_access import PartitionedTensorAccess
from src.ir.partitioning.partitioning import Partitioning
from src.parse import *


class PartitionedEinsum(Einsum):
    

    def __init__(self,
                 decl_einsum: DeclaredEinsum,
                 mapping_parser: MappingParser):

        self.decl_einsum = decl_einsum
        self.conds: List[Condition] = []

        self.partitioning = Partitioning(self.decl_einsum, mapping_parser)
        self.conds = self.partitioning.get_conds().copy()

        tensor_accesses = []
        for decl_tensor_access in self.decl_einsum.get_reuse_spec().get_tensor_accesses():

            part_tensor_access = PartitionedTensorAccess(
                decl_tensor_access, self.partitioning)
            tensor_accesses.append(part_tensor_access)

            for cond in part_tensor_access.get_conds():
                if cond not in self.conds:
                    self.conds.append(cond)

        decl_iter_ranks = [
            decl_rank for decl_rank in decl_einsum.get_reuse_spec().get_iter_ranks().values()]
        iter_ranks = {part_rank.get_name(
        ): part_rank for part_rank in self.partitioning.apply_to_ranks(decl_iter_ranks)}

        reuse_spec: \
            ReuseSpecification[PartitionedRank,
                               PartitionedTensor,
                               PartitionedTensorAccess] \
            = self.decl_einsum.get_reuse_spec().update_spec(iter_ranks, tensor_accesses)

        super().__init__(
            self.decl_einsum.get_name(),
            reuse_spec,
            self.decl_einsum.get_op_spec())

    def get_decl_einsum(self) -> DeclaredEinsum:
        
        return self.decl_einsum

    def get_conds(self) -> List[Condition]:
        
        return self.conds

    def get_partitioning(self) -> Partitioning:
        
        return self.partitioning

    def key(self) -> Tuple[Any, ...]:
        return super().key() + (self.decl_einsum,)

    def __hash__(self) -> int:
        
        return hash(("PartitionedEinsum", self.get_reuse_spec().get_einsum()))
