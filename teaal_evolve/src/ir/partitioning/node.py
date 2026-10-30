

from lark.tree import Tree
from typing import Any, Tuple

from src.ir.common import *


class DirectiveNode(Node):
    

    def __init__(self, directive: Tree,
                 src: Tuple[str, ...], dst: Tuple[str, ...]):
        self.directive = directive
        self.src = src
        self.dst = dst

    def get_directive(self) -> Tree:
        
        return self.directive

    def get_dst(self) -> Tuple[str, ...]:
        
        return self.dst

    def get_src(self) -> Tuple[str, ...]:
        
        return self.src

    def key(self) -> Tuple[Any, ...]:
        
        return self.directive, self.src, self.dst


class RankNode(Node):
    

    def __init__(self, rank: str):
        self.rank = rank

    def get_rank(self) -> str:
        
        return self.rank

    def key(self) -> Tuple[Any, ...]:
        
        return (self.rank,)
