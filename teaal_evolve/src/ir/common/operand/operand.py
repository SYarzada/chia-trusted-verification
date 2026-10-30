
import abc
from typing import Any, Generic, TypeVar

from src.common import *
from src.ir.common.access import *


T = TypeVar("T")


class Operand(Base, Generic[T], metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def get(self) -> Any:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def get_type(self) -> str:
        
        raise NotImplementedError                    
