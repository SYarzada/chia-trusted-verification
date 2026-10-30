

from sympy import Basic, Symbol                
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ir.binding.common.tensor import BoundTensor
from src.ir.common import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.partitioning import *
from src.parse import *


class BoundTensorAccess(TensorAccess[PartitionedRank, BoundTensor]):
    

    def __init__(self,
                 elaborated_einsum: ElaboratedEinsum,
                 map_einsum: MappedEinsum,
                 part_tensor_access: PartitionedTensorAccess,
                 elaborated_format: ElaboratedFormat,
                 binding_parser: BindingParser,
                 memories: Dict[str,
                                MemoryComponent]):
        self.part_tensor_access = part_tensor_access

        einsum = map_einsum.get_name()
        tensor = self.part_tensor_access.get_tensor().get_name()
        format_tensor = elaborated_format.get_tensor(einsum, tensor)

        self.bound_tensor = BoundTensor(
            elaborated_einsum,
            map_einsum,
            format_tensor,
            binding_parser,
            memories)

    def get_accesses(self) -> Dict[PartitionedRank,
                                   RankAccess[PartitionedRank]]:
        
        return self.part_tensor_access.get_accesses()

    def get_einsum(self) -> str:
        
        return self.part_tensor_access.get_einsum()

    def get_linearized_accesses(
        self,
        get_rank_expression: Callable[[PartitionedRank],
                                      Optional[Tuple[Basic,
                                                     Dict[Symbol, PartitionedRank]]]],
    ) -> Dict[str, Basic]:
        
        accesses = {}
        for mem_rank, part_ranks in self.get_tensor().get_rank_binding().items():
            multiplier = 1
            access = 0
            for part_rank in reversed(part_ranks):
                access_expr = 0
                for term in self.get_accesses()[part_rank].get_expr():
                    if isinstance(term, ConstantTerm):
                        access_expr += term.get()
                        continue

                    assert isinstance(term, VariableTerm)

                    rank_expr = get_rank_expression(term.get())
                    assert rank_expr is not None

                    access_expr += rank_expr[0]

                access += access_expr * multiplier
                multiplier *= part_rank.get_shape()

            accesses[mem_rank] = access

        return accesses

    def get_part_tensor_access(self) -> PartitionedTensorAccess:
        
        return self.part_tensor_access

    def get_tensor(self) -> BoundTensor:
        
        return self.bound_tensor

    @staticmethod
    def merge_accesses(
            tensor_accesses: List["BoundTensorAccess"],
            get_rank_expression:
        Callable[[PartitionedRank],
                 Optional[Tuple[Basic, Dict[Symbol,
                                            PartitionedRank]]]],
            location: str
        ) -> Dict[PartitionedRank, Tuple[RankAccess[PartitionedRank],
                                         RankAccess[PartitionedRank]]]:
        
                                                                            
                    
        bound_tensor0 = tensor_accesses[0].get_tensor()
        per_tensor_ranks = [bound_tensor0.get_format_tensor().get_ranks()]
        for tensor_access1 in tensor_accesses[1:]:
            bound_tensor1 = tensor_access1.get_tensor()

            per_tensor_ranks.append(
                bound_tensor1.get_format_tensor().get_ranks())
            ranks_match = {rank0: rank1 for rank0, rank1 in zip(
                per_tensor_ranks[0], per_tensor_ranks[-1])}
            bound_tensor0.match_layout(
                bound_tensor1, ranks_match, "concrete", "concrete", True)

                                                          
        ranges = {}
        for ranks in zip(*per_tensor_ranks):
            accesses = [tensor_accesses[i].get_accesses()[rank]
                        for i, rank in enumerate(ranks)]
            ranges[ranks[0]] = (RankAccess.merge(
                accesses, get_rank_expression, ranks[0].get_name() + " rank of the " + location))

        return ranges

    def key(self) -> Tuple[Any, ...]:
        return self.part_tensor_access, self.bound_tensor
