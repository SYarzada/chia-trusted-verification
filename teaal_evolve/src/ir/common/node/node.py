

import abc

from typing import Any, Iterable

from src.common import *


class Node(Base, metaclass=abc.ABCMeta):
    

    def __hash__(self) -> int:
        
        return hash(repr(self))

    def __repr__(self) -> str:
        
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"
