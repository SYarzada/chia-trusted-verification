

from typing import Any, List, Set, Tuple

from src.common import *
from src.ir.common import *
from src.ir.mapping.rank import MappedRank
from src.ir.partitioning import *
from src.parse import *


class MappedEinsum(Einsum):
    

    def __init__(self,
                 part_einsum: PartitionedEinsum,
                 mapping_parser: MappingParser):
        self.part_einsum = part_einsum
        einsum = self.part_einsum.get_name()

        iter_ranks = self.part_einsum.get_reuse_spec().get_iter_ranks()

        loop_order = mapping_parser.get_loop_orders()[einsum]
        ErrorUtils.check_ranks(
            set(iter_ranks.keys()),
            loop_order,
            einsum + " loop order")

        spacetime = mapping_parser.get_spacetime()
        space = spacetime[einsum].space
        time = spacetime[einsum].time

        ErrorUtils.check_ranks(
            set(iter_ranks.keys()),
            space + time,
            einsum + " spacetime")

        space_order = [rank for rank in loop_order if rank in space]
        if space_order != space:
            raise RankOrderMismatchError(
                einsum + " loop order", einsum + " space")

        time_order = [rank for rank in loop_order if rank in time]
        if time_order != time:
            raise RankOrderMismatchError(
                einsum + " loop order", einsum + " time")

        self.ranks = {
            rank: MappedRank(iter_ranks[rank], rank in space)
            for rank in loop_order}
        self.loop_order = [self.ranks[rank] for rank in loop_order]

        super().__init__(
            einsum,
            self.part_einsum.get_reuse_spec(),
            self.part_einsum.get_op_spec())

    def check_concrete(self):
        
        for rank in self.loop_order:
            rank.check_concrete()

    def check_template(self):
        
        for rank in self.loop_order:
            rank.check_template()

    def get_contracted_ranks(self) -> List[MappedRank]:
        
        contracted_ranks = {self.ranks[part_rank.get_name(
        )] for part_rank in self.part_einsum.get_reuse_spec().get_contracted_ranks()}

        return [rank for rank in self.loop_order if rank in contracted_ranks]

    def get_loop_order(self) -> List[MappedRank]:
        
        return self.loop_order

    def get_part_einsum(self) -> PartitionedEinsum:
        
        return self.part_einsum

    def maybe_get_rank(self, rank: str, location: str) -> MappedRank:
        
                                                                
        if rank not in self.ranks.keys():
            raise UnknownRankError(self.get_name() + "." + rank, location)

        return self.ranks[rank]

    def key(self) -> Tuple[Any, ...]:
        return self.part_einsum, self.loop_order

    def __hash__(self) -> int:
        
        return hash(("MappedEinsum", self.get_reuse_spec().get_einsum()))
