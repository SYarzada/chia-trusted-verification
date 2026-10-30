

from typing import Any, List, Tuple

from src.ir.common.op.arg import Argument


class TemplateArgument(Argument):
    

    def __init__(self, str_: str):
        self.str = str_

    def gen(self) -> str:
        
        return "<" + self.str + ">"

    def get(self) -> str:
        
        return self.str

    def key(self) -> Tuple[Any, ...]:
        return self.str,
