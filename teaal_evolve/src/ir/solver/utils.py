

from functools import reduce

from sympy import Add, Basic, floor, Integer, Mod, Mul, Rational, Symbol                
import z3                

from src.common import *


class SolverUtils:
    @staticmethod
    def mk_seq(elems, sort):
        match len(elems):
            case 0:
                return z3.Empty(sort)
            case 1:
                return z3.Unit(elems[0])
            case _:
                return z3.Concat([z3.Unit(elem) for elem in elems])

    @staticmethod
    def sympy_to_z3(sexpr: Basic) -> z3.ArithSortRef:
        if isinstance(sexpr, floor):
            return SolverUtils.sympy_to_z3(sexpr.args[0])

        divide = False
        if isinstance(sexpr, Integer) or isinstance(sexpr, int):
            return z3.IntSort().cast(int(sexpr))

        elif isinstance(sexpr, Symbol):
            return z3.Int(str(sexpr))

                           
        elif isinstance(sexpr, Mul):
            sargs = Utils.split(sexpr)
            if sargs[1] == 1:
                sargs = sargs[0].args
            else:
                divide = True

        elif isinstance(sexpr, Rational):
            sargs = (sexpr.p, sexpr.q)

        else:
            sargs = sexpr.args

        args = [SolverUtils.sympy_to_z3(arg) for arg in sargs]
        if isinstance(sexpr, Add):
            return reduce(lambda x, y: x + y, args)

        elif isinstance(sexpr, Mod):
            return args[0] % args[1]

        elif isinstance(sexpr, Mul) and divide:
            return args[0] / args[1]

        elif isinstance(sexpr, Mul):
            return reduce(lambda x, y: x * y, args)

        elif isinstance(sexpr, Rational):
            return args[0] / args[1]

                              
        elif isinstance(sexpr.args[0], Symbol) and sexpr.args[1] == 1:
            return args[0]

        raise NotImplementedError(
            "Unknown expression: " + str(sexpr) + " with type " + str(type(sexpr)))
