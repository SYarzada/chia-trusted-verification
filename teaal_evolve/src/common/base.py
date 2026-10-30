

import abc

from typing import Any, Tuple


class Base(metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def key(self) -> Tuple[Any, ...]:
        
        raise NotImplementedError                    

    def __eq__(self, other: object) -> bool:
        
        if isinstance(other, type(self)):
            return self.key() == other.key()
        return False

    def __hash__(self) -> int:
        
        return hash(repr(self))

    def __repr__(self) -> str:
        
        return "(" + type(self).__name__ + ", " + ", ".join(str(field)
                                                            for field in self.key()) + ")"

    def __str__(self) -> str:
        
        return repr(self)
