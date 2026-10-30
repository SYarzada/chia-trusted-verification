

import math

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import z3                

from src.ir.solver.utils import SolverUtils


class ComputeRoofSolver:
    @staticmethod
    def find_roof(costs: Dict[Tuple[Tuple[str, ...], str, str], float], rtol: float) \
            -> Tuple[float, Set[str], Dict[Tuple[str, ...], Set[Tuple[str, str]]]]:
        
        solver = z3.Optimize()
        engine_costs = ComputeRoofSolver.__make_solver(solver, costs)

                                                       
        slowest = z3.Real("slowest")
        solver.minimize(slowest)

        result = solver.check()
        assert result == z3.sat
        model = solver.model()

                                                     
        best_slowest = model.evaluate(slowest).py_value()

        solver = z3.Optimize()
        engine_costs = ComputeRoofSolver.__make_solver(solver, costs)

                                                                             
                                                          
        solver.assert_and_track(
            slowest <= best_slowest * (1 + rtol), "best_slowest")

        total_cost = 0
        for engine_cost in engine_costs.values():
            total_cost += engine_cost

        solver.minimize(total_cost)
        result = solver.check()
        assert result == z3.sat
        model = solver.model()

                              
        bounding_engines = set()
        for engine in engine_costs.keys():
            if math.isclose(model.evaluate(engine_costs[engine]).py_value(),
                            model.evaluate(slowest).py_value()):
                bounding_engines.add(engine)

                                                             
        best_total_cost = model.evaluate(total_cost).py_value()

        solver = z3.Solver()
        engine_costs = ComputeRoofSolver.__make_solver(solver, costs)

                                               
                                                          
        solver.assert_and_track(
            slowest <= best_slowest * (1 + rtol), "best_slowest")
                                                   
        solver.assert_and_track(
            total_cost <= best_total_cost * 1.0001,
            "best_total_cost")

                                            
        instructions_used: Dict[Tuple[str, ...], Set[Tuple[str, str]]] = {}
        for (einsums, engine, instruction) in costs.keys():
            name = "".join(einsums) + engine + instruction
            result = solver.check(z3.Bool(name))

            if result == z3.sat:
                if einsums not in instructions_used.keys():
                    instructions_used[einsums] = set()

                instructions_used[einsums].add((engine, instruction))

        return model.evaluate(slowest).py_value(
        ), bounding_engines, instructions_used

    @staticmethod
    def __make_solver(solver: z3.Solver, costs: Dict[Tuple[Tuple[str, ...], str, str], float]) \
            -> Dict[str, z3.ArithSortRef]:
        
        engine_costs: Dict[str, z3.ArithSortRef] = {}
        all_uses: Dict[str, List[z3.BoolSortRef]] = {}

                              
        for (einsums, engine, instruction), cost in costs.items():
            used = "".join(einsums) + engine + instruction

            used_bool = z3.Bool(used)
            used_int = z3.Int(used + "_int")

                                                              
            solver.assert_and_track(
                z3.Implies(
                    used_bool,
                    used_int == 1),
                used + "-True->1")
                                                                      
            solver.assert_and_track(
                z3.Implies(
                    z3.Not(used_bool),
                    used_int == 0),
                used + "-False->0")

            if engine not in engine_costs.keys():
                engine_costs[engine] = 0

            engine_costs[engine] += used_int * cost

                                                                
            for einsum in einsums:
                if einsum not in all_uses:
                    all_uses[einsum] = []

                all_uses[einsum].append(used_bool)

                                                 
        for einsum, uses in all_uses.items():
                                      
            solver.assert_and_track(z3.Or(*uses), einsum + "-at-least-once")
                                             
            solver.assert_and_track(
                z3.AtMost(
                    *
                    uses,
                    1),
                einsum +
                "-at-most-once")

                        
        engines = list(engine_costs.keys())
        slowest = z3.Real("slowest")
        one_cost = []
        for i, engine in enumerate(engines):
            engine_cost = engine_costs[engine]
            one_cost.append(engine_cost == slowest)

            other_costs = []
            for j, other in enumerate(engines):
                if i == j:
                    continue

                other_costs.append(engine_cost >= engine_costs[other])

                                                                             
                         
                                                                           
            solver.assert_and_track(
                z3.Implies(
                    engine_cost == slowest,
                    z3.And(
                        *
                        other_costs)),
                engine +
                "-slowest_is_cost")

                                                         
                                      
        solver.assert_and_track(z3.Or(*one_cost), "slowest-is-cost")

        return engine_costs
