

import abc

from typing import Any, Tuple

from src.common import *


class Argument(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    
