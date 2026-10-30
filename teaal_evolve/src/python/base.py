

import abc

from typing import Any, Dict, Tuple

from src.common.base import Base as GlobalBase


class Base(GlobalBase):
    def __hash__(self) -> int:
        
        return hash(repr(self))

    def key(self) -> Tuple[Any, ...]:
        
        return tuple(vars(self).values())

    def get_attrs(self) -> Dict[str, Any]:
        
        return vars(self)

    def __repr__(self) -> str:
        
        attrs = self.get_attrs()
        strs = [key + "=" + attrs[key] if isinstance(attrs[key], str)
                else key + "=" + repr(attrs[key])
                for key in sorted(attrs)]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"


class Argument(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    


class Assignable(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    


class Expression(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    


class Index(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    


class Operator(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self) -> str:
        
        raise NotImplementedError                    


class Payload(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self, parens: bool) -> str:
        
        raise NotImplementedError                    


class Statement(Base, metaclass=abc.ABCMeta):
    

    @abc.abstractmethod
    def gen(self, depth: int) -> str:
        
        raise NotImplementedError                    
