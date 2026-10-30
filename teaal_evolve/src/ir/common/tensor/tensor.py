

import abc
from typing import Generic, TypeVar

from src.common import *
from src.ir.common.rank import *


R = TypeVar("R")


class Tensor(Base, Generic[R], metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def get_name(self) -> str:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def get_ranks(self) -> List[R]:
        
        raise NotImplementedError                    

    @abc.abstractmethod
    def to_str_repr(self) -> Any:
        pass
