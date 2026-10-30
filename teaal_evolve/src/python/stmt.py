

from typing import List, Optional, Tuple

from src.python.base import Assignable, Expression, Operator, Payload, Statement
from src.python.expr import EVar


class SAssign(Statement):
    

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        
        return "    " * depth + self.assn.gen() + " = " + self.expr.gen()


class SBlock(Statement):
    

    def __init__(self, stmts: List[Statement]) -> None:
        self.stmts = stmts

    def gen(self, depth: int) -> str:
        
        return "\n".join([s.gen(depth) for s in self.stmts])

    def add(self, stmt: Statement) -> None:
        
        if isinstance(stmt, SBlock):
            self.stmts.extend(stmt.stmts)
        else:
            self.stmts.append(stmt)

    def __len__(self) -> int:
        
        return len(self.stmts)


class SCode(Statement):
    

    def __init__(self, code: str) -> None:
        self.code = code

    def gen(self, depth: int) -> str:
        
        return "    " * depth + self.code


class SComment(Statement):
    

    def __init__(self, comment: str) -> None:
        self.comment = comment

    def gen(self, depth: int) -> str:
        
        return "    " * depth + "# " + self.comment


class SExpr(Statement):
    

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        
        return "    " * depth + self.expr.gen()


class SFor(Statement):
    

    def __init__(
            self,
            payload: Payload,
            expr: Expression,
            stmt: Statement) -> None:
        self.payload = payload
        self.expr = expr
        self.stmt = stmt

    def gen(self, depth: int) -> str:
        
        return "    " * depth + "for " + \
            self.payload.gen(False) + " in " + self.expr.gen() + ":\n" + self.stmt.gen(depth + 1)


class SFunc(Statement):
    

    def __init__(self, name: str, args: List[EVar], body: Statement) -> None:
        self.name = name
        self.args = args
        self.body = body

    def gen(self, depth: int) -> str:
        
        args = ", ".join([arg.gen() for arg in self.args])
        header = "def " + self.name + "(" + args + "):\n"
        return "    " * depth + header + self.body.gen(depth + 1)


class SIAssign(Statement):
    

    def __init__(
            self,
            assn: Assignable,
            op: Operator,
            expr: Expression) -> None:
        self.assn = assn
        self.op = op
        self.expr = expr

    def gen(self, depth: int) -> str:
        
        return "    " * depth + self.assn.gen() + " " + self.op.gen() + \
            "= " + self.expr.gen()


class SIf(Statement):
    

    def __init__(self,
                 if_: Tuple[Expression,
                            Statement],
                 elifs: List[Tuple[Expression,
                                   Statement]],
                 else_: Optional[Statement]) -> None:
        self.if_ = if_
        self.elifs = elifs
        self.else_ = else_

    def gen(self, depth: int) -> str:
        
        out = "    " * depth
        out += "if " + self.if_[0].gen() + ":\n" + self.if_[1].gen(depth + 1)

        for cond, stmt in self.elifs:
            out += "\n" + "    " * depth
            out += "elif " + cond.gen() + ":\n" + stmt.gen(depth + 1)

        if self.else_ is not None:
            out += "\n" + "    " * depth
            out += "else:\n" + self.else_.gen(depth + 1)

        return out


class SPass(Statement):
    

    def __init__(self) -> None:
        pass

    def gen(self, depth: int) -> str:
        
        return "    " * depth + "pass"


class SReturn(Statement):
    

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        
        return "    " * depth + "return " + self.expr.gen()
