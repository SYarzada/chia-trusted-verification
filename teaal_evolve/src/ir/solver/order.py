

from typing import List, Tuple
import z3                

from src.ir.solver.utils import SolverUtils


class OrderSolver:
    @staticmethod
    def check_eq(equations: List[Tuple[List[str], List[str]]],
                 goal1: List[str], goal2: List[str]) -> bool:
        
        Element = z3.DeclareSort("Element")
        ListSort = z3.SeqSort(Element)

        s = z3.Solver()

        def mk_const(name):
            return z3.Const(name, ListSort)

        def mk_seq(names):
            match len(names):
                case 0:
                    return z3.Empty(ListSort)
                case 1:
                    return mk_const(names[0])
                case _:
                    return z3.Concat([mk_const(name) for name in names])

        for lhs, rhs in equations:
            s.add(mk_seq(lhs) == mk_seq(rhs))

                                             
                                                
                                         
        result = s.check(mk_seq(goal1) != mk_seq(goal2))

        return result == z3.unsat
