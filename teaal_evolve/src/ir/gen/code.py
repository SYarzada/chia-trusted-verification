
from typing import Any, Tuple

from src.ir.gen.base import BaseGeneration


class CodeGeneration(BaseGeneration):

    def __init__(self, code: str):
        self.code = code

    def get(self) -> str:
        
        return self.code

    def key(self) -> Tuple[Any, ...]:
        return self.code,

    def __str__(self) -> str:
        return self.code
