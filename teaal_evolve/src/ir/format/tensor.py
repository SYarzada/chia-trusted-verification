

from typing import Any, List, Optional, Tuple

from src.common import *
from src.ir.common import *
from src.ir.einsum import *
from src.ir.solver import *
from src.ir.partitioning import *
from src.parse import *


class FormattedTensor(Tensor):
    

    def __init__(
            self,
            part_tensor: PartitionedTensor,
            format_parser: FormatParser):

        self.part_tensor = part_tensor
        einsum = self.part_tensor.get_einsum()
        tensor = self.part_tensor.get_name()

        rank_order = format_parser.get_rank_orders(tensor)[einsum]

        location = self.get_name() + \
            " format (when computing the " + einsum + " Einsum)"

        part_ranks = {
            part_rank.get_name(): part_rank for part_rank in self.part_tensor.get_ranks()}
        ErrorUtils.check_ranks(
            set(part_ranks.keys()), rank_order, location)

        format_ranks = [part_ranks[rank] for rank in rank_order]
        self.ranks = format_ranks

        self.bit_width = format_parser.get_bit_width(tensor)
        self.bit_widths = format_parser.get_bit_widths(tensor)
        self.data_type = format_parser.get_data_type(tensor)
        self.data_types = format_parser.get_data_types(tensor)

    def check_concrete(self):
        
                                               
        if self.bit_width is None:
            raise MissingFieldError("bit-width", self.get_name() + " format")

                                                          
        if self.bit_widths is not None:
            raise IllegalFieldError(
                "bit-widths", self.get_name() + " format")

                                               
        if self.data_type is None:
            raise MissingFieldError("data-type", self.get_name() + " format")

                                                          
        if self.data_types is not None:
            raise IllegalFieldError(
                "data-types", self.get_name() + " format")

    def check_template(self):
        
                                                      
        if self.bit_widths is None and self.bit_width is None:
            raise MissingFieldError("bit-widths", self.get_name() + " format")

                                           
        if self.bit_widths is not None and self.bit_width is not None:
            raise IllegalFieldError("bit-width", self.get_name() + " format")

                                                                            
                    
        if self.bit_widths is None:
            self.bit_widths = [self.bit_width]
            self.bit_width = None

                                                      
        if self.data_types is None and self.data_type is None:
            raise MissingFieldError("data-types", self.get_name() + " format")

                                           
        if self.data_types is not None and self.data_type is not None:
            raise IllegalFieldError("data-type", self.get_name() + " format")

                                                                            
                    
        if self.data_types is None:
            self.data_types = [self.data_type]
            self.data_type = None

    def get_bit_width(self) -> Optional[int]:
        
        return self.bit_width

    def get_bit_widths(self) -> Optional[List[int]]:
        
        return self.bit_widths

    def get_data_type(self) -> Optional[str]:
        
        return self.data_type

    def get_data_types(self) -> Optional[List[str]]:
        
        return self.data_types

    def get_decl_tensor(self) -> DeclaredTensor:
        
        return self.part_tensor.get_decl_tensor()

    def get_einsum(self) -> str:
        
        return self.part_tensor.get_einsum()

    def get_name(self) -> str:
        
        return self.part_tensor.get_name()

    def get_part_tensor(self) -> PartitionedTensor:
        
        return self.part_tensor

    def get_ranks(self) -> List[PartitionedRank]:
        
        return self.ranks

    def match(self, other: "FormattedTensor"):
        
        assert self.get_name() == other.get_name()

        spart = self.part_tensor.get_partitioning()
        sranks = PartitionedRank.scoped_ranks(
            [rank for rank in self.ranks if rank.get_shape() != 1])

        opart = other.part_tensor.get_partitioning()
        oranks = PartitionedRank.scoped_ranks(
            [rank for rank in other.ranks if rank.get_shape() != 1])

        equalities = spart.get_solver_equalities(opart)

                                                                         
        if not OrderSolver.check_eq(equalities, sranks, oranks):
            prefix = self.get_name() + " format (when computing Einsum "
            raise RankOrderMismatchError(
                prefix + self.get_einsum() + ")",
                prefix + other.get_einsum() + ")")

    def to_str_repr(self) -> StringTensor:
        
        ranks = [rank.to_str_repr() for rank in self.ranks]

        return StringTensor(self.get_name(), ranks)

    def key(self) -> Tuple[Any, ...]:
        return self.part_tensor, self.ranks

    def __hash__(self) -> int:
        
        return hash(("FormattedTensor", self.get_einsum(), self.get_name()))
