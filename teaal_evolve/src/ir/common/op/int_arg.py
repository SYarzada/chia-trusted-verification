

from typing import Any, List, Tuple

from src.ir.common.op.concrete_arg import ConcreteArgument


class IntegerArgument(ConcreteArgument):
    

    def __init__(self, int_: int):
        self.int = int_

    def gen(self) -> str:
        
        return str(self.int)

    def get(self) -> int:
        
        return self.int

    def key(self) -> Tuple[Any, ...]:
        return self.int,
