

from typing import Any, List, Tuple

from src.ir.common.op.arg import Argument
from src.ir.common.op.op import Operator


class CurriedOperator(Operator):
    

    def __init__(self, name: str, args: List[Argument]):
        super().__init__(name)
        self.args = args

    def gen(self) -> str:
        
        return self.name + "[" + ", ".join(arg.gen()
                                           for arg in self.args) + "]"

    def get_args(self) -> List[Argument]:
        
        return self.args

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.args
