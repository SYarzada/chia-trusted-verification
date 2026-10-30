

from typing import Optional

from src.python.base import Expression, Index


class IColon(Index):
    

    def __init__(
            self,
            start: Optional[Expression],
            end: Optional[Expression]) -> None:
        self.start = start
        self.end = end

    def gen(self) -> str:
        
        if self.start is None:
            start = ""
        else:
            start = self.start.gen()

        if self.end is None:
            end = ""
        else:
            end = self.end.gen()

        return start + ":" + end


class IJust(Index):
    

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        
        return self.expr.gen()
