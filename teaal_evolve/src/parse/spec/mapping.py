

from lark.tree import Tree
from typing import Any, cast, Dict, List, Optional, Tuple

from src.parse.lark import *
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase
from src.common import *


class Mapping(Base):
    

    def __init__(self, model: MappingModel):
        self.partitioning: Dict[str, Dict[Tuple[str, ...], List[Tree]]]
        if model.partitioning is None:
            self.partitioning = {}
        else:
            self.partitioning = {}
            for einsum, parts in model.partitioning.items():
                self.partitioning[einsum] = {}
                for root, directs in parts.items():
                    tuple_root = tuple(
                        str(token) for token in PartitioningRank.parse(root).children)

                    self.partitioning[einsum][tuple_root] = [
                        PartitioningDirective.parse(direct) for direct in directs]

        self.loop_orders = model.loop_order
        self.spacetime = model.spacetime

                                    
        self.fusion: Optional[Dict[Tuple[str, ...],
                                   List[Tuple[Optional[str], List[Tuple[str, str]]]]]] = None
        if model.fusion:
            self.fusion = {}
            for fusion_mapping in model.fusion:
                fused_ranks = []
                for ranks_spec in fusion_mapping.ranks:
                    if isinstance(ranks_spec, list):
                        loop_name = None
                        unparsed_ranks = ranks_spec

                    else:
                                                    

                        loop_name = next(iter(ranks_spec))
                        unparsed_ranks = ranks_spec[loop_name]

                    parsed_ranks = [
                        ScopedRank.parse(rank) for rank in unparsed_ranks]
                    fused_ranks.append((loop_name, parsed_ranks))

                self.fusion[tuple(fusion_mapping.einsums)] = \
                    [(loop_name, [(str(scoped_rank.children[0]), str(scoped_rank.children[1]))
                                  for scoped_rank in scoped_ranks])
                     for loop_name, scoped_ranks in fused_ranks]

    @classmethod
    def from_dict(cls, dict_: dict) -> "Mapping":
        
        return Mapping(MappingModel.model_validate(dict_["mapping"]))

    @classmethod
    def from_file(cls, filename: str) -> "Mapping":
        
        return Mapping.from_dict(ParseBase.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Mapping":
        
        return Mapping.from_dict(ParseBase.parse_str(string))

    def get_fusion(self) -> Optional[Dict[Tuple[str, ...],
                                          List[Tuple[Optional[str], List[Tuple[str, str]]]]]]:
        
        return self.fusion

    def get_loop_orders(self) -> Dict[str, List[str]]:
        
        return self.loop_orders

    def get_partitioning(self) -> Dict[str, Dict[Tuple[str, ...], List[Tree]]]:
        
        return self.partitioning

    def get_spacetime(self) -> Dict[str, SpacetimeModel]:
        
        return self.spacetime

    def key(self) -> Tuple[Any, ...]:
        
        return self.partitioning, self.loop_orders, self.spacetime, self.fusion
