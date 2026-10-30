

import abc

from src.common import *


class Component(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def get_name(self) -> str:
        pass
