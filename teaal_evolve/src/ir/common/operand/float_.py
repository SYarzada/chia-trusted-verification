
from typing import Any, Tuple

from src.ir.common.operand.operand import Operand


class FloatOperand(Operand):
    

    def __init__(self, val: float):
        self.val = val

    def gen(self) -> str:
        
        return str(self.val)

    def get(self) -> float:
        
        return self.val

    def get_type(self) -> str:
        
        return "constant"

    def key(self) -> Tuple[Any, ...]:
        return self.val,
