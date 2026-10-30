
from typing import cast, Dict, List, Optional, TypeVar

from src.common import *
from src.ir.binding.common.einsum import BoundEinsum
from src.ir.binding.common.rank import BoundRank
from src.ir.binding.common.stmt import InstructionStatement, LoopStatement, Statement
from src.ir.common import *
from src.ir.mapping import *
from src.parse import *


R = TypeVar("R")


class BoundFusion(Fusion[BoundEinsum, BoundRank]):
    def __init__(self,
                 map_fusion: MappedFusion,
                 bound_einsums: Dict[str, Tuple[Tuple[str, ...], int, BoundEinsum]],
                 binding_parser: BindingParser):
        
        self.map_fusion = map_fusion

        self.fused_ranks_dict = {}
        self.bound_ranks = {}
        fused_ranks: Dict[Tuple[BoundEinsum, ...], Dict[str, BoundRank]] = {}
        for map_group in self.map_fusion.get_groups():
            bound_group = tuple(bound_einsums[map_einsum.get_name()][2]
                                for map_einsum in map_group)

            fused_ranks[bound_group] = {}
            all_group_ranks = self.map_fusion.get_all_fused_ranks(map_group)
            for name, group_ranks in all_group_ranks.items():
                bound_rank = BoundRank(
                    name,
                    group_ranks,
                    map_fusion,
                    bound_einsums,
                    binding_parser)
                fused_ranks[bound_group][name] = bound_rank
                self.bound_ranks[name] = bound_rank

                for map_rank in group_ranks:
                    self.fused_ranks_dict[map_rank] = bound_rank

            for bound_einsum in bound_group:
                for map_rank in bound_einsum.get_loop_order():
                                                                 
                    if map_rank in self.fused_ranks_dict.keys():
                        continue

                    name = bound_einsum.get_name() + "." + map_rank.get_name()
                    bound_rank = BoundRank(
                        name, (map_rank,), map_fusion, bound_einsums, binding_parser)
                    fused_ranks[bound_group][name] = bound_rank
                    self.bound_ranks[name] = bound_rank
                    self.fused_ranks_dict[map_rank] = bound_rank

        ordered_einsums = [bound_einsum for _, _,
                           bound_einsum in bound_einsums.values()]
        ordered_einsums.sort(key=lambda e: bound_einsums[e.get_name()][1])

        stmts = []
        for bound_einsum in ordered_einsums:
            if bound_einsum.get_loop_order():
                map_rank = bound_einsum.get_loop_order()[0]
                stmt: Statement = LoopStatement(
                    self.fused_ranks_dict[map_rank].get_name())

                if stmt not in stmts:
                    stmts.append(stmt)

        ordered_stmts = self.__expand_loop_stmts(stmts, bound_einsums)
        self.positions = {}
        i = 0
        for stmt in ordered_stmts:
            assert isinstance(stmt, InstructionStatement)

            for einsum in stmt.get():
                self.positions[einsum] = i
                i += 1

        super().__init__(fused_ranks)

    def get_fused_ranks(self, rank: R) -> BoundRank:
        
        assert isinstance(rank, MappedRank)

        return self.fused_ranks_dict[rank]

    def get_position(self, einsum: str) -> int:
        
        return self.positions[einsum]

    def __expand_loop_stmts(
        self,
        given_stmts: List[Statement],
        bound_einsums: Dict[str, Tuple[Tuple[str, ...], int, BoundEinsum]],
    ) -> List[Statement]:
        
        stmts: List[Statement] = []
        has_loop = False
        for stmt in given_stmts:
            if isinstance(stmt, InstructionStatement):
                stmts.append(stmt)
                continue

            has_loop = True
            assert isinstance(stmt, LoopStatement)

            bound_rank = self.bound_ranks[stmt.get()]
            stmts.extend(bound_rank.get_statements())

        if has_loop:
            return self.__expand_loop_stmts(stmts, bound_einsums)
        else:
            return stmts
