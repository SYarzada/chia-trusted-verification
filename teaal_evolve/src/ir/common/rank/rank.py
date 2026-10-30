

import abc

from src.common import *


class Rank(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_shape(self) -> Basic:
        pass

    @abc.abstractmethod
    def to_str_repr(self) -> Any:
        pass
