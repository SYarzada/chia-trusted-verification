

from typing import Dict, Sequence

from src.python.base import Argument, Expression, Index, Operator


class EAccess(Expression):
    

    def __init__(self, obj: Expression, inds: Sequence[Index]) -> None:
        self.obj = obj
        self.inds = inds

    def gen(self) -> str:
        
        return self.obj.gen() + "[" + ", ".join(ind.gen()
                                                for ind in self.inds) + "]"


class EBinOp(Expression):
    

    def __init__(
            self,
            expr1: Expression,
            op: Operator,
            expr2: Expression) -> None:
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2

    def gen(self) -> str:
        
        return self.expr1.gen() + " " + self.op.gen() + " " + self.expr2.gen()


class EBool(Expression):
    

    def __init__(self, bool_: bool) -> None:
        self.bool = bool_

    def gen(self) -> str:
        
        return str(self.bool)


class ECode(Expression):
    

    def __init__(self, code: str) -> None:
        self.code = code

    def gen(self) -> str:
        
        return self.code


class EComp(Expression):
    

    def __init__(self, elem: Expression, var: str, iter_: Expression) -> None:
        self.elem = elem
        self.var = var
        self.iter = iter_

    def gen(self) -> str:
        
        return "[" + self.elem.gen() + " for " + self.var + \
            " in " + self.iter.gen() + "]"


class EDict(Expression):
    

    def __init__(self, dict_: Dict[Expression, Expression]):
        self.dict = dict_

    def gen(self) -> str:
        
        items = []
        for key, val in self.dict.items():
            items.append(key.gen() + ": " + val.gen())
        return "{" + ", ".join(items) + "}"


class EField(Expression):
    

    def __init__(self, obj: str, field: str):
        self.obj = obj
        self.field = field

    def gen(self) -> str:
        
        return self.obj + "." + self.field


class EFloat(Expression):
    

    def __init__(self, float_: float) -> None:
        self.float = float_

    def gen(self) -> str:
        
        if self.float == float("inf"):
            return "float(\"inf\")"
        elif self.float == -float("inf"):
            return "-float(\"inf\")"
        else:
            return str(self.float)


class EFunc(Expression):
    

    def __init__(self, name: str, args: Sequence[Argument]) -> None:
        self.name = name
        self.args = args

    def gen(self) -> str:
        
        return self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EInt(Expression):
    

    def __init__(self, int_: int) -> None:
        self.int = int_

    def gen(self) -> str:
        
        return str(self.int)


class ELambda(Expression):
    

    def __init__(self, args: Sequence[str], body: Expression) -> None:
        self.args = args
        self.body = body

    def gen(self) -> str:
        
        return "lambda " + ", ".join(self.args) + ": " + self.body.gen()


class EList(Expression):
    

    def __init__(self, list_: Sequence[Expression]) -> None:
        self.list = list_

    def gen(self) -> str:
        
        return "[" + ", ".join([e.gen() for e in self.list]) + "]"


class EMethod(Expression):
    

    def __init__(self, obj: Expression, name: str,
                 args: Sequence[Argument]) -> None:
        self.obj = obj
        self.name = name
        self.args = args

    def gen(self) -> str:
        
        return self.obj.gen() + "." + self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EParens(Expression):
    

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        
        return "(" + self.expr.gen() + ")"


class EString(Expression):
    

    def __init__(self, string: str) -> None:
        self.string = string

    def gen(self) -> str:
        
        return "\"" + self.string + "\""


class ETuple(Expression):
    

    def __init__(self, elems: Sequence[Expression]) -> None:
        self.elems = elems

    def gen(self) -> str:
        
                                                                        
        if len(self.elems) == 1:
            return "(" + self.elems[0].gen() + ",)"

        else:
            return "(" + ", ".join([elem.gen() for elem in self.elems]) + ")"


class EVar(Expression):
    

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        
        return self.name
