

from sympy import Basic, Symbol                
from typing import Dict, List, Tuple
import z3                

from src.common import *
from src.ir.solver.utils import SolverUtils


class RangeSolver:
    @staticmethod
    def get_range(ranges: List[Tuple[Basic,
                                     Basic]],
                  rank_shapes: List[Basic],
                  rank_var_shapes: Dict[str, Basic],
                  location: str) -> Tuple[int, int]:
        
                          
        solver = z3.Solver()

                                       
        for shape in rank_shapes:
            z3_shape = SolverUtils.sympy_to_z3(shape)
            solver.add(z3_shape > 0)

                                  
        for rank_var, shape in rank_var_shapes.items():
            z3_rank_var = z3.Int(rank_var)

            solver.add(z3_rank_var >= 0)
            solver.add(z3_rank_var < SolverUtils.sympy_to_z3(shape))

                                                                        
        active = z3.Int("active")
        inactive = z3.Int("inactive")

        active_ranges = []
        inactive_ranges = []
        for start, end in ranges:
            z3_start = SolverUtils.sympy_to_z3(start)
            z3_end = SolverUtils.sympy_to_z3(end)
            active_ranges.append((z3_start <= active) & (active <= z3_end))
            inactive_ranges.append((inactive < z3_start) | (z3_end < inactive))

        solver.add(z3.Or(*active_ranges))
        solver.add(z3.And(*inactive_ranges))

                                         
        z3_range_start = None
        for i, (start, _) in enumerate(ranges):
            z3_start = SolverUtils.sympy_to_z3(start)
            result = solver.check(active < z3_start)

            if result == z3.unsat:
                z3_range_start = z3_start
                start_i = i
                break

                                                           
        if z3_range_start is None:
            raise CoordinateRangeMergeError(location)

                                         
        z3_range_end = None
        for i, (_, end) in enumerate(ranges):
            z3_end = SolverUtils.sympy_to_z3(end)
            result = solver.check(active > z3_end)

            if result == z3.unsat:
                z3_range_end = z3_end
                end_i = i
                break

                                                         
        if z3_range_end is None:
            raise CoordinateRangeMergeError(location)

                                                                        
                                  
        result = solver.check(
            (z3_range_start <= inactive) & (
                inactive <= z3_range_end))

                                                                           
                                           
        if result == z3.sat:
            raise CoordinateRangeMergeError(location)

        return (start_i, end_i)
