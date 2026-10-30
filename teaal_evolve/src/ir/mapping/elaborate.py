
from sympy import Symbol                
from typing import Dict, List, Set

from src.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.partitioning import *
from src.ir.mapping.einsum import MappedEinsum
from src.ir.mapping.fusion import MappedFusion
from src.parse import *


class ElaboratedMapping:

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            mapping_parser: MappingParser):

        self.elaborated_einsum = elaborated_einsum
        self.conds: List[Condition] = []

        part_einsums = {}
        for decl_einsum in self.elaborated_einsum.get_einsums():
            einsum = decl_einsum.get_name()
            part_einsums[einsum] = PartitionedEinsum(
                decl_einsum, mapping_parser)

            for cond in part_einsums[einsum].get_conds():
                if cond not in self.conds:
                    self.conds.append(cond)

        extra_einsums = set(
            mapping_parser.get_partitioning().keys()).difference(
            part_einsums.keys())
                                                                             
        if extra_einsums:
            raise UnknownEinsumError(extra_einsums.pop(), "partitioning")

        loop_orders = mapping_parser.get_loop_orders()
        spacetime = mapping_parser.get_spacetime()

        corr_einsums = set(part_einsums.keys())
        given_lo_einsums = list(loop_orders.keys())
        ErrorUtils.check_einsums(corr_einsums, given_lo_einsums, "loop order")

        given_st_einsums = list(spacetime.keys())
        ErrorUtils.check_einsums(corr_einsums, given_st_einsums, "spacetime")

        self.einsums = []
        self.einsums_dict = {}
        for decl_einsum in elaborated_einsum.get_einsums():
            einsum = decl_einsum.get_name()
            self.einsums_dict[einsum] = MappedEinsum(
                part_einsums[einsum], mapping_parser)
            self.einsums.append(self.einsums_dict[einsum])

        self.fusion = MappedFusion(self.einsums, mapping_parser)

    def check_concrete(self):
        
        for einsum in self.einsums:
            einsum.check_concrete()

    def check_template(self):
        
        for einsum in self.einsums:
            einsum.check_template()

    def get_conds(self) -> List[Condition]:
        
        return self.conds

    def get_einsums(self) -> List[MappedEinsum]:
        
        return self.einsums

    def get_fusion(self) -> MappedFusion:
        
        return self.fusion

    def get_symbols(self) -> List[Symbol]:
        
        all_symbols = [self.elaborated_einsum.get_symbols()] + \
            [einsum.get_part_einsum().get_partitioning().get_symbols()
             for einsum in self.einsums]

        symbols = []
        for symbol_list in all_symbols:
            for symbol in symbol_list:
                if symbol not in symbols:
                    symbols.append(symbol)

        return symbols

    def maybe_get_einsum(self, einsum: str, location: str) -> MappedEinsum:
        
        if einsum not in self.einsums_dict.keys():
            raise UnknownEinsumError(einsum, location)

        return self.einsums_dict[einsum]
