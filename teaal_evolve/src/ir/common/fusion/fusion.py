
import abc
from typing import Dict, Generic, List, Set, Tuple, TypeVar

from src.common import *
from src.ir.common.rank import Rank
from src.ir.common.einsum import Einsum


E = TypeVar("E")
FusedRankInfo = TypeVar("FusedRankInfo")
R = TypeVar("R")


class Fusion(Base, Generic[E, FusedRankInfo], metaclass=abc.ABCMeta):
    

    def __init__(
            self, fused_ranks: Dict[Tuple[E, ...], Dict[str, FusedRankInfo]]):
        self.fused_ranks = fused_ranks

        self.groups: Dict[E, Tuple[E, ...]] = {}
        for group in self.fused_ranks.keys():
            for einsum in group:
                self.groups[einsum] = group

    def get_group(self, einsum: E) -> Tuple[E, ...]:
        
        return self.groups[einsum]

    def get_groups(self) -> List[Tuple[E, ...]]:
        
        return list(self.fused_ranks.keys())

    def get_all_fused_ranks(
            self, group: Tuple[E, ...]) -> Dict[str, FusedRankInfo]:
        
        return self.fused_ranks[group]

    def get_all_fused_ranks_from_str(
            self, einsum_name: str) -> Dict[str, FusedRankInfo]:
        
        for einsum, group in self.groups.items():
            if einsum.get_name() == einsum_name:
                return self.fused_ranks[group]

                                                                 
        assert False                    

    @abc.abstractmethod
    def get_fused_ranks(self, rank: R) -> FusedRankInfo:
        
        raise NotImplementedError                    

    def key(self) -> Tuple[Any, ...]:
        return self.fused_ranks,
