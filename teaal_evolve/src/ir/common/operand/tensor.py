
from typing import Any, Tuple, TypeVar

from src.ir.common.access import *
from src.ir.common.operand.operand import Operand


T = TypeVar("T")


class TensorOperand(Operand[T]):
    

    def __init__(self, tensor_access: T):
        self.tensor_access = tensor_access

    def gen(self) -> str:
        
        return self.tensor_access.gen()

    def get(self) -> T:
        
        return self.tensor_access

    def get_type(self) -> str:
        
        return "tensor"

    def key(self) -> Tuple[Any, ...]:
        return self.tensor_access,
