

from src.python.base import Operator


class OAdd(Operator):
    

    def gen(self) -> str:
        
        return "+"


class OAnd(Operator):
    

    def gen(self) -> str:
        
        return "&"


class ODiv(Operator):
    

    def gen(self) -> str:
        
        return "/"


class OEqEq(Operator):
    

    def gen(self) -> str:
        
        return "=="


class OFDiv(Operator):
    

    def gen(self) -> str:
        
        return "//"


class OGt(Operator):
    

    def gen(self) -> str:
        
        return ">"


class OGtEq(Operator):
    

    def gen(self) -> str:
        
        return ">="


class OIn(Operator):
    

    def gen(self) -> str:
        
        return "in"


class OLAnd(Operator):
    

    def gen(self) -> str:
        
        return "and"


class OLOr(Operator):
    

    def gen(self) -> str:
        
        return "or"


class OLt(Operator):
    

    def gen(self) -> str:
        
        return "<"


class OLtEq(Operator):
    

    def gen(self) -> str:
        
        return "<="


class OLtLt(Operator):
    

    def gen(self) -> str:
        
        return "<<"


class OMod(Operator):
    

    def gen(self) -> str:
        
        return "%"


class OMul(Operator):
    

    def gen(self) -> str:
        
        return "*"


class ONotIn(Operator):
    

    def gen(self) -> str:
        
        return "not in"


class OOr(Operator):
    

    def gen(self) -> str:
        
        return "|"


class OSub(Operator):
    

    def gen(self) -> str:
        
        return "-"
