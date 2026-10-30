
import abc
from sympy import Basic                
from typing import Dict, Generic, TypeVar

from src.common import *
from src.ir.common.access.rank import RankAccess
from src.ir.common.rank import *
from src.ir.common.tensor import *


R = TypeVar("R")
T = TypeVar("T")


class TensorAccess(Base, Generic[R, T], metaclass=abc.ABCMeta):
    

    def gen(self) -> str:
        
        accesses = [self.get_accesses()[rank].gen()
                    for rank in self.get_tensor().get_ranks()]

        return self.get_tensor().get_name() + "[" + ", ".join(accesses) + "]"

    @abc.abstractmethod
    def get_accesses(self) -> Dict[R, RankAccess[R]]:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def get_tensor(self) -> T:
        
        raise NotImplementedError                    
