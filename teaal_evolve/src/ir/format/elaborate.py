

from typing import Dict, List, Set

from src.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.format.tensor import FormattedTensor
from src.ir.mapping import *
from src.ir.partitioning import *
from src.parse import *


class ElaboratedFormat:

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            elaborated_mapping: ElaboratedMapping,
            format_parser: FormatParser):

        self.conds: List[Condition] = []

                                                       
        ErrorUtils.check_tensors(
            set(elaborated_einsum.get_tensors().keys()),
            list(format_parser.get_tensors()),
            "format")

        self.tensors: Dict[str, Dict[str, FormattedTensor]] = {}
        part_einsums = {}
        for map_einsum in elaborated_mapping.get_einsums():
            einsum = map_einsum.get_name()
            part_einsums[einsum] = map_einsum.get_part_einsum()

            for part_tensor in map_einsum.get_reuse_spec().get_tensors():
                tensor = part_tensor.get_name()

                if tensor not in self.tensors:
                    self.tensors[tensor] = {}

                                                                        
                rank_orders = format_parser.get_rank_orders(tensor)
                if einsum not in rank_orders.keys():
                    raise MissingEinsumError(einsum, tensor + " format")

                self.tensors[tensor][einsum] = FormattedTensor(
                    part_tensor, format_parser)

        for tensor in format_parser.get_tensors():
            ErrorUtils.check_einsums(set(self.tensors[tensor].keys()), list(
                format_parser.get_rank_orders(tensor).keys()), tensor + " format")

                                                                            
                               
        for tensor, tensor_dict in self.tensors.items():
            format_tensors = list(tensor_dict.values())

            base_tensor = format_tensors[0]
            for format_tensor in format_tensors[1:]:
                base_tensor.match(format_tensor)

    def check_concrete(self):
        
        for tensor_dict in self.tensors.values():
            for format_tensor in tensor_dict.values():
                format_tensor.check_concrete()

    def check_template(self):
        
        for tensor_dict in self.tensors.values():
            for format_tensor in tensor_dict.values():
                format_tensor.check_template()

    def get_tensor(self, einsum: str, tensor: str) -> FormattedTensor:
        
        return self.tensors[tensor][einsum]
