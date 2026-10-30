
import abc
from typing import Any, Tuple

from src.common import *


class Statement(Base, metaclass=abc.ABCMeta):
    

    def get(self) -> Any:
        
        raise NotImplementedError                    

    def display(self) -> str:
        
        raise NotImplementedError                    


class InstructionStatement(Statement):
    

    def __init__(self, einsums: Tuple[str, ...]):
        self.einsums = einsums

    def get(self) -> Tuple[str, ...]:
        
        return self.einsums

    def display(self) -> str:
        
        if len(self.einsums) == 1:
            return self.einsums[0] + ".instruction"
        else:
            return "(" + ", ".join(self.einsums) + ").instruction"

    def key(self) -> Tuple[Any, ...]:
        return self.einsums,


class LoopStatement(Statement):
    

    def __init__(self, name: str):
        self.name = name

    def get(self) -> str:
        
        return self.name

    def display(self) -> str:
        
        return self.name

    def key(self) -> Tuple[Any, ...]:
        return self.name,
