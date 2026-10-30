
from functools import reduce
import operator

from sympy import Symbol                
from typing import Dict, List, Set, Tuple

from src.common import *
from src.ir import *
from src.parse import *
from src.trans.model.binder import Binder
from src.trans.yaml import *


class ComputeRoofModel:
    
    @staticmethod
    def display_roof(
            elaborated_einsum: ElaboratedEinsum,
            elaborated_arch: ElaboratedArchitecture,
            roof: Tuple[float, Set[str], Dict[Tuple[str, ...], Set[Tuple[str, str]]]]) -> YAMLGenerator:
        
        cascade = []
        for decl_einsum in elaborated_einsum.get_einsums():
            split_einsums = decl_einsum.split_for_solver()
            for str_einsum in split_einsums:
                cascade.append(str_einsum.get_name())

        engines = list(elaborated_arch.get_engines().keys())
        bottlenecks = list(roof[1])
        bottlenecks.sort(key=lambda e: engines.index(e))

        match_keys = list(roof[2].keys())
        match_keys.sort(
            key=lambda key: tuple(
                cascade.index(einsum) for einsum in key))

        matching = []
        for einsums in match_keys:
            templates = []
            for component, instruction in roof[2][einsums]:
                templates.append(
                    {"component": component, "instruction": instruction})

            matching.append({"einsums": list(einsums), "templates": templates})

        spec = {
            "latency": roof[0],
            "bottlenecks": bottlenecks,
            "optimal-matching": matching}

        return YAMLGenerator(spec)

    @staticmethod
    def find_roof(
            elaborated_einsum: ElaboratedEinsum,
            elaborated_arch: ElaboratedArchitecture,
            shapes: Dict[str, int],
            matches: Dict[Tuple[Einsum, ...], List[Tuple[ElaboratedTemplate, Optional[float]]]]) -> \
            Tuple[float, Set[str], Dict[Tuple[str, ...], Set[Tuple[str, str]]]]:
        

                                      
        for rank_name, decl_rank in elaborated_einsum.get_ranks().items():
            if isinstance(decl_rank.get_shape(), int):
                shapes[rank_name] = decl_rank.get_shape()

                                                      
        costs = {}
        for ceinsums, templates in matches.items():
            einsum_names = tuple(einsum.get_name() for einsum in ceinsums)

            for ceinsum in ceinsums:
                for crank in ceinsum.get_reuse_spec().get_iter_ranks().keys():
                    if crank not in shapes.keys():
                        message = "all shapes must be concretized to calculate latency"
                        raise IllegalSymbolicShapeError(
                            crank, "given", message)

            for i, (template, _) in enumerate(templates):
                key = (
                    einsum_names,
                    template.get_component(),
                    template.get_name())

                cost_metrics = template.get_cost_metrics()

                                                                      
                              
                if not cost_metrics or CostMetricEnum.compute not in cost_metrics:
                    costs[key] = 0.0
                    templates[i] = (template, costs[key])
                    continue

                                          
                spatial_ranks: Dict[str, Dict[str, Optional[int]]] = {}
                iter_ranks: Dict[str, Dict[str, Optional[int]]] = {}
                teinsums = []
                for bound_einsum in template.get_instruction().get_einsums():
                    teinsums += bound_einsum.split_for_solver()

                    einsum = bound_einsum.get_name()
                    spatial_ranks[einsum] = {}
                    iter_ranks[einsum] = {}

                    for map_rank in bound_einsum.get_loop_order():
                        shape = map_rank.get_shape()
                        if isinstance(shape, int):
                            iter_ranks[einsum][map_rank.get_name()] = shape
                        else:
                            iter_ranks[einsum][map_rank.get_name()] = None

                        if map_rank.get_is_spatial():
                            spatial_ranks[einsum][map_rank.get_name()] = shape

                assert len(teinsums) == len(ceinsums)

                                                           
                parallelism = ComputeRoofModel.__get_per_instruction_max(
                    teinsums, list(ceinsums), spatial_ranks, shapes)
                per_instruction_points = ComputeRoofModel.__get_per_instruction_max(
                    teinsums, list(ceinsums), iter_ranks, shapes)

                                                
                per_instruction_cost = float(
                    template.get_instruction_overhead())
                num_instructions = None
                for ceinsum, teinsum in zip(ceinsums, teinsums):
                    op_init_latency = template.get_op_init_latency(
                        teinsum.get_name())

                                                                        
                    if op_init_latency is None:
                        continue

                    iter_points = 1
                    for rank in ceinsum.get_reuse_spec().get_iter_ranks():
                        iter_points *= shapes[rank]

                    per_instruction_cost += \
                        per_instruction_points[teinsum.get_name()] * \
                        op_init_latency / parallelism[teinsum.get_name()]

                    new_num_instructions = iter_points / \
                        per_instruction_points[teinsum.get_name()]
                    if num_instructions is None:
                        num_instructions = new_num_instructions

                                                                           
                                                                          
                                              
                    elif num_instructions != new_num_instructions:
                        per_instruction_cost = float("inf")

                assert num_instructions is not None

                min_cycles = template.get_instruction_init_latency()
                if min_cycles:
                    cost = max(
                        per_instruction_cost,
                        min_cycles) * num_instructions
                else:
                    cost = per_instruction_cost * num_instructions

                frequency = template.get_frequency()
                if frequency is None:
                    GlobalState.set_location_prefix(
                        template.get_component() +
                        "'s " +
                        template.get_name() +
                        " template")
                    raise MissingFieldError("frequency", "attributes")

                if cost != float("inf"):
                    costs[key] = cost / frequency
                    templates[i] = (template, costs[key])
                else:
                    templates[i] = (template, float("inf"))

        latency, bottlenecks, opt_matches = ComputeRoofSolver.find_roof(
            costs, 1e-4)

        return latency, bottlenecks, opt_matches

    @staticmethod
    def __get_per_instruction_max(
            teinsums: List[Einsum],
            ceinsums: List[Einsum],
            target_rank_info: Dict[str, Dict[str, Optional[int]]],
            shapes: Dict[str, int]) -> Dict[str, int]:
        
        target_ranks: Dict[str, Set[str]] = {}
        for teinsum, tshapes in target_rank_info.items():
            target_ranks[teinsum] = set()
            for trank in tshapes.keys():
                target_ranks[teinsum].add(trank)

        matches = ComputeRoofModel.__find_rank_matches(
            teinsums, ceinsums, target_ranks)

        instruction_max = {}
        for teinsum, teinsum_matches in matches.items():
            classes: Dict[Tuple[str, ...], List[str]] = {}
            for trank, cranks_set in teinsum_matches.items():
                sorted_cranks = list(cranks_set)
                sorted_cranks.sort()

                cranks = tuple(sorted_cranks)
                if cranks not in classes:
                    classes[cranks] = []
                classes[cranks].append(trank)

                                        
            available = 1
            for cranks, tranks in classes.items():
                cshape = reduce(operator.mul, map(lambda r: shapes[r], cranks))

                curr_tshapes = [target_rank_info[teinsum][trank]
                                for trank in tranks]
                if all(isinstance(opt_tshape, int)
                       for opt_tshape in curr_tshapes):
                    tshape = reduce(operator.mul, curr_tshapes)
                    assert isinstance(tshape, int)

                    available *= min(cshape, tshape)
                else:
                    available *= cshape

            instruction_max[teinsum] = available

        return instruction_max

    @staticmethod
    def __find_rank_matches(teinsums: List[Einsum],
                            ceinsums: List[Einsum],
                            target_ranks: Dict[str,
                                               Set[str]]) -> Dict[str,
                                                                  Dict[str,
                                                                       Set[str]]]:
        
                               
                                                                         
                                
        csolver_einsums = [ceinsum.get_reuse_spec().get_solver_repr()
                           for ceinsum in ceinsums]
        coutputs = [
            ceinsums[-1].get_reuse_spec().get_output().get().get_tensor().get_name()]
        tsolver_einsums = [teinsum.get_reuse_spec().get_solver_repr()
                           for teinsum in teinsums]
        toutputs = [
            teinsums[-1].get_reuse_spec().get_output().get().get_tensor().get_name()]

        solver = InstructionSolver(
            csolver_einsums, coutputs, tsolver_einsums, toutputs)

        matches: Dict[str, Dict[str, Set[str]]] = {}
        for teinsum in teinsums:
            if teinsum.get_name() not in target_ranks.keys():
                continue

            matches[teinsum.get_name()] = {}

            for trank in target_ranks[teinsum.get_name()]:
                                                          
                matches[teinsum.get_name()][trank] = solver.find_all_rank_matches(
                    trank)

        return matches
