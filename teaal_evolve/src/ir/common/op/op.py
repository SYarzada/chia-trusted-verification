

from typing import Any, Tuple

from src.common import *


class Operator(Base):
    

    def __init__(self, name: str):
        self.name = name

    def gen(self) -> str:
        
        return self.name

    def get_name(self) -> str:
        
        return self.name

    def key(self) -> Tuple[Any, ...]:
        return self.name,
