

from src.python.base import Assignable, Expression


class AAccess(Assignable):
    

    def __init__(self, obj: Expression, ind: Expression) -> None:
        self.obj = obj
        self.ind = ind

    def gen(self) -> str:
        
        return self.obj.gen() + "[" + self.ind.gen() + "]"


class AField(Assignable):
    

    def __init__(self, obj: str, field: str):
        self.obj = obj
        self.field = field

    def gen(self) -> str:
        
        return self.obj + "." + self.field


class AVar(Assignable):
    

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        
        return self.name
