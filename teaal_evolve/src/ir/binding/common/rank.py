
from sympy import Basic                
from typing import Any, Dict, Tuple

from src.common import *
from src.ir.binding.common.einsum import BoundEinsum
from src.ir.binding.common.stmt import InstructionStatement, LoopStatement, Statement
from src.ir.common import *
from src.ir.mapping import *
from src.parse import *


class BoundRank(Rank):
    

    def __init__(
            self,
            name: str,
            map_ranks: Tuple[MappedRank, ...],
            map_fusion: MappedFusion,
            bound_einsums: Dict[str, Tuple[Tuple[str, ...], int, BoundEinsum]],
            binding_parser: BindingParser):
        
        self.name = name
        self.map_ranks = map_ranks

                                                    
        shapes = {map_rank.get_shape() for map_rank in self.map_ranks}
        assert len(shapes) == 1
        self.shape = shapes.pop()

        self.is_sequential = any(map_rank.get_is_sequential()
                                 for map_rank in self.map_ranks)

        fused_ranks = map_fusion.get_all_fused_ranks_from_str(
            self.map_ranks[0].get_einsum())
        fused_rank_names = {val: key for key, val in fused_ranks.items()}

                              
        self.stmts: Dict[Statement, Tuple[int, int]] = {}
        stmt: Statement
        for map_rank in self.map_ranks:
            einsums, pos, bound_einsum = bound_einsums[map_rank.get_einsum()]
            loop_order = bound_einsum.get_loop_order()

            i = loop_order.index(map_rank)
            if i == len(loop_order) - 1 or \
                    loop_order[i + 1] in bound_einsum.get_bound_ranks():
                stmt = InstructionStatement(einsums)

            else:
                next_loop = loop_order[i + 1]
                ranks = map_fusion.get_fused_ranks(next_loop)
                if ranks in fused_rank_names.keys():
                    stmt = LoopStatement(fused_rank_names[ranks])
                else:
                    assert len(ranks) == 1
                    stmt = LoopStatement(
                        next_loop.get_einsum() + "." + next_loop.get_name())

            if stmt not in self.stmts:
                self.stmts[stmt] = (pos, 0)
            else:
                self.stmts[stmt] = (min(pos, self.stmts[stmt][0]), 0)

        pipelining = binding_parser.get_pipelining()
        if self.name in pipelining.keys():
            self.is_sequential = True

            pipelined_stmts = []
            for i, (tree, offset) in enumerate(pipelining[self.name]):
                if tree.data == "name":
                    stmt = LoopStatement(str(tree.children[0]))
                elif tree.data == "rank":
                    stmt = LoopStatement(".".join(str(child)
                                         for child in tree.children))
                elif tree.data == "instruction":
                    stmt = InstructionStatement(
                        tuple(str(child) for child in tree.children[0].children))
                else:
                                                                              
                           
                    raise NotImplementedError                    

                pipelined_stmts.append((stmt, i, offset))

            corr = {stmt.display() for stmt in self.stmts}
            given = [info[0].display() for info in pipelined_stmts]
            location = self.name + " loop pipelining"
            ErrorUtils.check_statements(corr, given, location)

                                                          

            for (stmt, pos, offset) in pipelined_stmts:
                self.stmts[stmt] = (pos, offset)

    def get_is_sequential(self) -> bool:
        
        return self.is_sequential

    def get_name(self) -> str:
        
        return self.name

    def get_max_offset(self) -> int:
        
        return max(info[1] for info in self.stmts.values())

    def get_offset(self, statement: Statement) -> int:
        
        return self.stmts[statement][1]

    def get_pos(self, statement: Statement) -> int:
        
        return self.stmts[statement][0]

    def get_ranks(self) -> Tuple[MappedRank, ...]:
        
        return self.map_ranks

    def get_shape(self) -> Basic:
        
        return self.shape

    def get_statements(self) -> List[Statement]:
        
        stmts = list(self.stmts.keys())
        stmts.sort(key=lambda s: self.stmts[s][0])

        return stmts

    def to_str_repr(self) -> StringRank:
        
        return StringRank(self.name)

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.map_ranks,

    def __hash__(self) -> int:
        
        return hash(("BoundRank", self.name))
