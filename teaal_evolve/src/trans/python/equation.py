

from sympy import Add, Basic, floor, Integer, Mod, Mul, Pow, Rational, Symbol                
from typing import Type

from src.common import *
from src.ir import *
from src.python import *


class Equation:
    @staticmethod
    def trans(sexpr: Basic) -> Expression:
        expr = Equation.__trans(sexpr)

        if isinstance(expr, EParens):
            return expr.expr

        else:
            return expr

    @staticmethod
    def __trans(sexpr: Basic) -> Expression:
        if isinstance(sexpr, Add):
            return Equation.__combine(sexpr, OAdd)

                                 
        elif isinstance(sexpr, floor) and isinstance(sexpr.args[0], Mul):
            a, b = Utils.split(sexpr)
            return EParens(
                EBinOp(
                    Equation.__trans(a),
                    OFDiv(),
                    Equation.__trans(b)))

                                 
        elif isinstance(sexpr, floor) and isinstance(sexpr.args[0], Symbol):
            return EVar(str(sexpr.args[0]))

        elif isinstance(sexpr, Integer) or isinstance(sexpr, int):
            return EInt(int(sexpr))

        elif isinstance(sexpr, Mod):
            return EParens(EBinOp(
                Equation.__trans(
                    sexpr.args[0]), OMod(), Equation.__trans(
                    sexpr.args[1])))

        elif isinstance(sexpr, Mul):
            return Equation.__combine(sexpr, OMul)

        elif isinstance(sexpr, Rational):
            return EParens(EBinOp(EInt(sexpr.p), ODiv(), EInt(sexpr.q)))

        elif isinstance(sexpr, Symbol):
            return EVar(str(sexpr))

        raise NotImplementedError(
            "Unknown expression: " + str(sexpr) + " with type " + str(type(sexpr)))

    @staticmethod
    def __combine(sexpr: Basic, op: Type[Operator]) -> Expression:
        
        hexprs = [Equation.__trans(arg) for arg in sexpr.args]
        bexpr = EBinOp(hexprs[-2], op(), hexprs[-1])

        for hexpr in reversed(hexprs[:-2]):
            bexpr = EBinOp(hexpr, op(), bexpr)

        return EParens(bexpr)
