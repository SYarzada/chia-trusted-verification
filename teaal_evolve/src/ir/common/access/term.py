

import abc

from typing import Any, Generic, TypeVar

from src.common import *
from src.ir.common import *


R = TypeVar("R")


class Term(Base, Generic[R], metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def get(self) -> Any:
        
        raise NotImplementedError                    
