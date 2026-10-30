

from lark.tree import Tree
from typing import Any, cast, Dict, List, Optional, Tuple

from src.common import *
from src.parse.lark.equation import Equation
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase


class Einsum(Base):
    

    def __init__(self, model: EinsumModel):
        self.declaration = model.declaration
        self.expressions: List[Tuple[Optional[str], Tree]] = []
        for expr in model.expressions:
            if isinstance(expr, str):
                self.expressions.append((None, Equation.parse(expr)))

            elif isinstance(expr, dict):
                                                                          
                            
                name = list(expr)[0]
                self.expressions.append((name, Equation.parse(expr[name])))

        self.shape = model.shape

    @classmethod
    def from_dict(cls, dict_: dict) -> "Einsum":
        
        return Einsum(EinsumModel.model_validate(dict_["einsum"]))

    @classmethod
    def from_file(cls, filename: str) -> "Einsum":
        
        return Einsum.from_dict(ParseBase.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Einsum":
        
        return Einsum.from_dict(ParseBase.parse_str(string))

    def get_declaration(self) -> Dict[str, List[str]]:
        
        return self.declaration

    def get_expressions(self) -> List[Tuple[Optional[str], Tree]]:
        
        return self.expressions

    def get_shapes(self) -> Dict[str, int]:
        
        return self.shape

    def key(self) -> Tuple[Any, ...]:
        
        return self.declaration, self.expressions, self.shape
