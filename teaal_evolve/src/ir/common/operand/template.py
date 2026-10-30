
from typing import Any, Tuple

from src.ir.common.operand.operand import Operand


class TemplateOperand(Operand):
    

    def __init__(self, template: str):
        self.template = template

    def gen(self) -> str:
        
        return "<" + self.template + ">"

    def get(self) -> str:
        
        return self.template

    def get_type(self) -> str:
        
        return "template"

    def key(self) -> Tuple[Any, ...]:
        return self.template,
