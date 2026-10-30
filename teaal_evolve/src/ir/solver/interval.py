
from sympy import Basic, Symbol                
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import z3                

from src.ir.solver.utils import SolverUtils


class IntervalSolver:
    def __init__(
            self,
            access_info: List[Tuple[str, int, Dict[str, Basic]]],
            iter_spaces: Dict[str, List[str]],
            tensor_roots: List[str],
            shapes: Dict[str, Dict[str, int]]):
        
        self.roots = tensor_roots

        self.solver = z3.Solver()

                                                       
        self.iter_rank_vars: Dict[str, List[z3.ArithSortRef]] = {}
        for einsum, iter_space in iter_spaces.items():
            self.iter_rank_vars[einsum] = []
            for rank in iter_space:
                rank_var = z3.Int(einsum + "_" + rank.lower())
                self.iter_rank_vars[einsum].append(rank_var)

                                                
                self.solver.assert_and_track(
                    0 <= rank_var, "-".join([einsum, rank, "iter", "0"]))
                                                                                   
                                                                       
                self.solver.assert_and_track(
                    rank_var <= shapes[einsum][rank] - 1, "-".join([einsum, rank, "iter", "shape"]))

                                        
        self.rank_vars: Dict[Tuple[str, int], Dict[str, z3.ArithSortRef]] = {}
        for einsum, pos, access_dict in access_info:
            self.rank_vars[(einsum, pos)] = {}

            for root in self.roots:
                                                                           
                                                                   
                access = access_dict[root]
                if not isinstance(access, int):
                    for free_symbol in access_dict[root].free_symbols:
                        str_symbol = str(free_symbol)
                        new_val: Basic
                        if str_symbol.isupper():
                            new_val = shapes[einsum][str_symbol]
                        else:
                            new_val = Symbol(einsum + "_" + str_symbol)

                        access = access.subs(free_symbol, new_val)

                                                         
                root_var = z3.Int(einsum + "_" + root.lower())
                                                                              
                self.solver.assert_and_track(root_var == SolverUtils.sympy_to_z3(
                    access), "-".join([einsum, root, str(pos), "expr"]))

                                                
                self.solver.assert_and_track(
                    0 <= root_var, "-".join([einsum, root, "tensor", str(pos), "0"]))
                                                                       
                self.solver.assert_and_track(
                    root_var <= shapes[einsum][root] - 1, "-".join([einsum, root, "tensor", str(pos), "shape"]))

                self.rank_vars[(einsum, pos)][root] = root_var

    def intersect(self,
                  einsum0: str,
                  tensor_pos0: int,
                  point0: Tuple[Optional[int], ...],
                  einsum1: str,
                  tensor_pos1: int,
                  point1: Tuple[Optional[int], ...]) -> bool:
        
                                                  
        conds = []
        for root in self.roots:
            conds.append(
                self.rank_vars[(einsum0, tensor_pos0)][root] ==
                self.rank_vars[(einsum1, tensor_pos1)][root])

        for einsum, point in zip([einsum0, einsum1], [point0, point1]):
            for coord, rank_var in zip(point, self.iter_rank_vars[einsum]):
                if coord is not None:
                    conds.append(rank_var == coord)

        result = self.solver.check(z3.And(*conds))

                                
                                             
               
                                        

        return result == z3.sat
