

from sympy import Basic, floor, Mul, Pow, Rational, Symbol                
from typing import Tuple


class Utils:
    @staticmethod
    def nth_as_string(x: int) -> str:
        
        if x % 10 == 1 and x % 100 != 11:
            return str(x) + "st"
        elif x % 10 == 2 and x % 100 != 12:
            return str(x) + "nd"
        elif x % 10 == 3 and x % 100 != 13:
            return str(x) + "rd"
        else:
            return str(x) + "th"

    @staticmethod
    def less_than(a: Basic, b: Basic) -> bool:
        
        if isinstance(a, int) and isinstance(b, int):
            return a < b

        if a == b:
            return False

        anum, aden = Utils.split(a)
        bnum, bden = Utils.split(b)

        if anum == bnum and aden == 1:
            return False

        elif anum == bnum and bden == 1:
            return True

        raise ValueError(
            "Cannot compare expressions " + str(a) + " and " + str(b))

    @staticmethod
    def split(sexpr: Basic) -> Tuple[Basic, Basic]:
        
        sexpr = Utils.strip_floor(sexpr)

        if isinstance(sexpr, Mul):
            args = sexpr.args
        else:
            args = [sexpr]

        nums = []
        denoms = []
        for arg in args:
            if isinstance(arg, Pow) and arg.args[1] < 0:
                denoms.append(Pow(arg.args[0], -1 * arg.args[1]))
            elif isinstance(arg, Rational):
                if arg.p != 1:
                    nums.append(arg.p)

                if arg.q != 1:
                    denoms.append(arg.q)
            else:
                nums.append(arg)

        if not nums:
            nums = [1]
        if not denoms:
            denoms = [1]

        return (Mul(*nums), Mul(*denoms))

    @staticmethod
    def strip_floor(sexpr: Basic) -> Basic:
        
        if isinstance(sexpr, floor):
            return sexpr.args[0]
        return sexpr
