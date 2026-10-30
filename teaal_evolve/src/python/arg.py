

from src.python.base import Argument, Expression


class AJust(Argument):
    

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        
        return self.expr.gen()


class AParam(Argument):
    

    def __init__(self, name: str, expr: Expression) -> None:
        self.name = name
        self.expr = expr

    def gen(self) -> str:
        
        return self.name + "=" + self.expr.gen()
