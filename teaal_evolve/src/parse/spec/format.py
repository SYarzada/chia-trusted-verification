

from lark.tree import Tree
from typing import Any, cast, Dict, List, Optional, Set, Tuple

from src.common import *
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase


class Format(Base):
    

    def __init__(self, model: FormatModel):
        if model.format_:
            self.spec = model.format_
        else:
            self.spec = {}

    @classmethod
    def from_dict(cls, dict_: dict) -> "Format":
        
        return Format(FormatModel.model_validate({"format": dict_["format"]}))

    @classmethod
    def from_file(cls, filename: str) -> "Format":
        
        return Format.from_dict(ParseBase.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Format":
        
        return Format.from_dict(ParseBase.parse_str(string))

    def get_bit_width(self, tensor: str) -> Optional[int]:
        
        return self.spec[tensor].bit_width

    def get_bit_widths(self, tensor: str) -> Optional[List[int]]:
        
        return self.spec[tensor].bit_widths

    def get_data_type(self, tensor: str) -> Optional[str]:
        
        return self.spec[tensor].data_type

    def get_data_types(self, tensor: str) -> Optional[List[str]]:
        
        return self.spec[tensor].data_types

    def get_rank_orders(self, tensor: str) -> Dict[str, List[str]]:
        
        return self.spec[tensor].rank_order

    def get_tensors(self) -> Set[str]:
        
        return set(self.spec.keys())

    def key(self) -> Tuple[Any, ...]:
        
        return self.spec,
