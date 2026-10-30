

from typing import Any, Tuple

from src.ir.common import *
from src.ir.binding.common import *
from src.ir.binding.concrete import *
from src.ir.mapping import *


class BranchBodyNode(Node):
    

    def __init__(self, bound_einsum: BoundEinsum):
        self.bound_einsum = bound_einsum

    def get_einsum(self) -> BoundEinsum:
        
        return self.bound_einsum

    def key(self) -> Tuple[Any, ...]:
        
        return (self.bound_einsum,)

    def __str__(self):
        
        return "(BranchBodyNode, " + self.bound_einsum.get_name() + ")"


class BranchEndLoopNode(Node):
    

    def __init__(self, map_rank: MappedRank):
        self.map_rank = map_rank

    def get_rank(self) -> MappedRank:
        
        return self.map_rank

    def key(self) -> Tuple[Any, ...]:
        
        return self.map_rank,

    def __str__(self):
        
        return "(BranchEndLoopNode, (" + self.map_rank.get_einsum() + \
            ", " + self.map_rank.get_name() + "))"


class BranchLoopNode(Node):
    

    def __init__(self, map_rank: MappedRank):
        self.map_rank = map_rank

    def get_rank(self) -> MappedRank:
        
        return self.map_rank

    def key(self) -> Tuple[Any, ...]:
        
        return self.map_rank,

    def __str__(self):
        
        return "(BranchLoopNode, (" + self.map_rank.get_einsum() + \
            ", " + self.map_rank.get_name() + "))"


class BodyNode(Node):
    

    def __init__(self, instruction: Instruction):
        self.instruction = instruction

    def get_instruction(self) -> Instruction:
        
        return self.instruction

    def key(self) -> Tuple[Any, ...]:
        
        return (self.instruction,)

    def __str__(self):
        
        return "(BodyNode, " + self.instruction.get_name() + ")"


class EndLoopNode(Node):
    

    def __init__(self, bound_rank: BoundRank):
        self.rank = bound_rank

    def get_rank(self) -> BoundRank:
        
        return self.rank

    def key(self) -> Tuple[Any, ...]:
        
        return self.rank,

    def __str__(self):
        
        return "(EndLoopNode, [" + ", ".join("(" + rank.get_einsum() + ", " + \
                 rank.get_name() + ")" for rank in self.rank.get_ranks()) + "])"


class LoopNode(Node):
    

    def __init__(self, bound_rank: BoundRank):
        self.rank = bound_rank

    def get_rank(self) -> BoundRank:
        
        return self.rank

    def key(self) -> Tuple[Any, ...]:
        
        return self.rank,

    def __str__(self):
        
        return "(LoopNode, [" + ", ".join("(" + rank.get_einsum() + ", " +
                                          rank.get_name() + ")" for rank in self.rank.get_ranks()) + "])"


class StartNode(Node):
    

    def __init__(self):
        pass

    def key(self) -> Tuple[Any, ...]:
        
        return ()

    def __str__(self):
        
        return "(StartNode,)"
