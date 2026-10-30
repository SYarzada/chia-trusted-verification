
from lark import Token
from typing import Any, Dict, List, Tuple

from src.ir.gen.base import BaseGeneration


class TemplateGeneration(BaseGeneration):

    def __init__(self, tokens: List[Token]):
        self.template = [str(token) for token in tokens]

        self.concrete = [True for _ in self.template]
        for i, field in enumerate(self.template):
            if field == "tensor":
                self.concrete[i + 1] = False
            if field == "einsum":
                self.concrete[i + 1] = False

    def get(self) -> List[str]:
        
        return self.template

    def update(self, type_: str, old: str, new: str):
        
        for i, is_concrete in enumerate(self.concrete):
            if not is_concrete and self.template[i - 1] == type_ and \
                    self.template[i] == old:
                self.template[i] = new
                self.concrete[i] = True

    def key(self) -> Tuple[Any, ...]:
        return self.template,

    def __str__(self) -> str:
        return "<" + ".".join(self.template) + ">"
