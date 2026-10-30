

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.common import *
from src.ir.binding.common.tensor import BoundTensor
from src.ir.binding.common.tensor_access import BoundTensorAccess
from src.ir.common import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.partitioning import *
from src.parse import *


class BoundEinsum(Einsum):
    

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            elaborated_mapping: ElaboratedMapping,
            map_einsum: MappedEinsum,
            elaborated_format: ElaboratedFormat,
            binding_parser: BindingParser,
            memories: Dict[str, MemoryComponent]):
        self.map_einsum = map_einsum
        einsum = self.map_einsum.get_name()
        location = einsum + " engine binding"

        self.bound_ranks = []
        for rank in binding_parser.get_engine_ranks(einsum):
            self.bound_ranks.append(
                self.map_einsum.maybe_get_rank(
                    rank, location))

        loop_order = self.map_einsum.get_loop_order()

                                                                         
        if self.bound_ranks and loop_order[-len(
                self.bound_ranks):] != self.bound_ranks:
            raise RankOrderMismatchError(einsum + " loop order", location)

        tensor_accesses = []
        for part_tensor_access in self.map_einsum.get_reuse_spec().get_tensor_accesses():
            tensor_accesses.append(
                BoundTensorAccess(
                    elaborated_einsum,
                    self.map_einsum,
                    part_tensor_access,
                    elaborated_format,
                    binding_parser,
                    memories))

        reuse_spec: ReuseSpecification[PartitionedRank, BoundTensor, BoundTensorAccess] = self.map_einsum.get_reuse_spec(
        ).update_spec(self.map_einsum.get_reuse_spec().get_iter_ranks(), tensor_accesses)

                                                                           
        part_ranks = {map_rank.get_rank() for map_rank in self.bound_ranks}
        for bound_tensor_access in tensor_accesses:
            for rank_access in bound_tensor_access.get_accesses().values():
                free = []
                for term in rank_access.get_expr():
                    if isinstance(
                            term, VariableTerm) and term.get() in part_ranks:
                        free.append(term.get())

                if len(free) > 1:
                    rank_vars = [part_rank.get_name().lower()
                                 for part_rank in free]
                    tensor = bound_tensor_access.get_tensor().get_name()
                    raise MultipleUnconcretizedRankVariablesError(
                        rank_vars, tensor, einsum)

        self.engine = binding_parser.get_engine(einsum)
        self.instruction = binding_parser.get_instruction(einsum)

        super().__init__(einsum, reuse_spec, self.map_einsum.get_op_spec())

    def check_concrete(self, requires_constructor: bool):
        
        self.map_einsum.check_concrete()
        for i, bound_tensor in enumerate(self.reuse_spec.get_tensors()):
            bound_tensor.check_concrete(i == 0 and requires_constructor)

    def check_template(self):
        
        self.map_einsum.check_template()
        for bound_tensor in self.reuse_spec.get_tensors():
            bound_tensor.check_template()

    def get_bound_ranks(self) -> List[MappedRank]:
        
        return self.bound_ranks

    def get_contracted_ranks(self) -> List[MappedRank]:
        
        return self.map_einsum.get_contracted_ranks()

    def get_engine(self) -> str:
        
        return self.engine

    def get_instruction(self) -> str:
        
        return self.instruction

    def get_loop_order(self) -> List[MappedRank]:
        
        return self.map_einsum.get_loop_order()

    def get_map_einsum(self) -> MappedEinsum:
        
        return self.map_einsum

    def get_unbound_ranks(self) -> List[MappedRank]:
        
        return self.map_einsum.get_loop_order()[:-len(self.bound_ranks)]

    def maybe_get_rank(self, rank: str, location: str) -> MappedRank:
        
        return self.map_einsum.maybe_get_rank(rank, location)

    def match_template_form(
            self, other: "BoundEinsum") -> List[Tuple[Optional[str], str, str]]:
        
        assert self.get_engine() == other.get_engine()
        assert self.get_instruction() == other.get_instruction()

        reuse_templates = self.get_reuse_spec().match_template_form(
            other.get_reuse_spec(), self.engine, self.instruction)
        op_templates = self.get_op_spec().match_template(
            other.get_op_spec(), self.engine, self.instruction)

                                                                     

        return reuse_templates + op_templates

    def key(self) -> Tuple[Any, ...]:
        return self.map_einsum, self.reuse_spec, self.bound_ranks

    def __hash__(self) -> int:
        
        return hash(("BoundEinsum", self.get_reuse_spec().get_einsum()))
