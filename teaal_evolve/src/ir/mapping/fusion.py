
from typing import Any, Dict, List, Set, Tuple, TypeVar

from src.common import *
from src.ir.common import *
from src.ir.mapping.einsum import MappedEinsum
from src.ir.mapping.rank import MappedRank
from src.ir.memory import *
from src.ir.partitioning import *
from src.parse import *


R = TypeVar("R")


class MappedFusion(Fusion[MappedEinsum, Tuple[MappedRank, ...]]):
    

    def __init__(self,
                 map_einsum_list: List[MappedEinsum],
                 mapping_parser: MappingParser):
        map_einsums = {}
        self.sorted_tensors: Dict[str,
                                  Dict[MappedEinsum, PartitionedTensor]] = {}
        for map_einsum in map_einsum_list:
            map_einsums[map_einsum.get_name()] = map_einsum

            for part_tensor in map_einsum.get_reuse_spec().get_tensors():
                tensor = part_tensor.get_name()

                if part_tensor.get_name() not in self.sorted_tensors:
                    self.sorted_tensors[tensor] = {}

                self.sorted_tensors[tensor][map_einsum] = part_tensor

        fusion_spec = mapping_parser.get_fusion()
        self.prev_fused: Set[MappedRank] = set()
        fused_ranks: Dict[Tuple[MappedEinsum, ...],
                          Dict[str, Tuple[MappedRank, ...]]] = {}
        self.fused_ranks_dict: Dict[MappedRank, Tuple[MappedRank, ...]] = {}
        if fusion_spec:
            given_einsums = [einsum for group in fusion_spec.keys()
                             for einsum in group]
                                                                                  
                     
            ErrorUtils.check_einsums(
                set(map_einsums.keys()), given_einsums, "fusion")

                                                                         
                                  
            for str_group, rank_spec in fusion_spec.items():
                location = "[" + ", ".join(str(einsum)
                                           for einsum in str_group) + "] fusion"

                group = tuple(map_einsums[einsum] for einsum in str_group)
                fused_ranks[group] = {}
                for parsed_ranks_info in rank_spec:

                    map_ranks = []
                    ranks_str = "[" + ", ".join(".".join(info)
                                                for info in parsed_ranks_info[1]) + "]"
                    for einsum, rank in parsed_ranks_info[1]:
                        rank_str = einsum + "." + rank

                                                             
                        if einsum not in map_einsums.keys():
                            raise UnknownEinsumError(einsum, location)

                        map_einsum = map_einsums[einsum]

                                                                
                        if map_einsum not in group:
                            raise UnknownRankError(rank_str, location)

                        map_rank = map_einsum.maybe_get_rank(rank, location)

                                                                
                        if map_rank in self.prev_fused:
                            raise DuplicateRankError(rank_str, location)

                        self.prev_fused.add(map_rank)
                        map_ranks.append(map_rank)

                                                                              
                              
                    is_spatials = [map_rank.get_is_spatial()
                                   for map_rank in map_ranks]
                    if not all(is_spatials) and any(is_spatials):
                        raise SpacetimeMismatchError(ranks_str, location)

                                                                       
                    shapes = [map_rank.get_shape()
                              for map_rank in map_ranks]
                                                                              
                                            
                    ErrorUtils.check_one_shape(shapes, ranks_str, location)

                    if parsed_ranks_info[0]:
                        fused_name = parsed_ranks_info[0]
                    else:
                        fused_name = "-".join(".".join(info)
                                              for info in parsed_ranks_info[1])

                    fused_ranks[group][fused_name] = tuple(map_ranks)

                    for map_rank in map_ranks:
                        self.fused_ranks_dict[map_rank] = tuple(map_ranks)

                                       
        else:
            fused_ranks = {(map_einsum,): {}
                           for map_einsum in map_einsum_list}

                                                                            
                                                       
        for all_fused_ranks in fused_ranks.values():
            for rank_group in all_fused_ranks.values():
                matched_ranks: Dict[str,
                                    List[Tuple[VariableTerm, List[Term]]]] = {}
                for map_rank in rank_group:
                    map_einsum = map_einsums[map_rank.get_einsum()]

                    var_term = VariableTerm(map_rank.get_rank())
                    for tensor_access in map_einsum.get_reuse_spec().get_tensor_accesses():
                        tensor = tensor_access.get_tensor().get_name()
                        if tensor not in matched_ranks:
                            matched_ranks[tensor] = []

                        for tensor_rank, access in tensor_access.get_accesses().items():
                            if var_term in access.get_expr():
                                matched_ranks[tensor].append(
                                    (var_term, access.get_expr()))

                for tensor in matched_ranks.keys():
                    if len(matched_ranks[tensor]) < 2:
                        continue

                    base_var_term, base_expr = matched_ranks[tensor][0]
                    for comp_var_term, comp_expr in matched_ranks[tensor][1:]:
                        if len(base_expr) != len(comp_expr):
                            for map_rank in rank_group:
                                map_rank.set_is_sequential(True)
                            break

                        for base_term, comp_term in zip(base_expr, comp_expr):
                            if base_term == comp_term:
                                continue

                            if base_term == base_var_term and comp_term == comp_var_term:
                                continue

                            for map_rank in rank_group:
                                map_rank.set_is_sequential(True)
                            break

                    if rank_group[0].get_is_sequential():
                        break

        super().__init__(fused_ranks)

    def get_fused_ranks(self, rank: R) -> Tuple[MappedRank, ...]:
        
                                            
        assert isinstance(rank, MappedRank)

        if rank in self.fused_ranks_dict.keys():
            return self.fused_ranks_dict[rank]
        else:
            return (rank,)
