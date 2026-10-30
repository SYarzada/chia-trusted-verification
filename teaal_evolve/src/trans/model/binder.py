
from sympy import Symbol                
from typing import Any, cast, Dict, List, Optional, Tuple

from src.ir import *
from src.trans.teaal import *
from src.trans.yaml import *


class Binder:
    
    @staticmethod
    def display_all_matches(
            elaborated_einsum: ElaboratedEinsum,
            elaborated_arch: ElaboratedArchitecture,
            matches: Dict[Tuple[Einsum, ...], List[Tuple[ElaboratedTemplate, Optional[float]]]]) -> YAMLGenerator:
        
        cascade = []
        for decl_einsum in elaborated_einsum.get_einsums():
            cascade.extend(decl_einsum.split_for_solver())

        sorted_keys = list(matches.keys())
        sorted_keys.sort(
            key=lambda t: tuple(
                cascade.index(einsum) for einsum in t))

        bindings = []
        for key in sorted_keys:
            templates: List[Dict[str, Any]] = []
            for template, opt_cost in matches[key]:
                templates.append(
                    {"component": template.get_component(), "instruction": template.get_name()})
                if opt_cost is not None:
                    templates[-1]["latency"] = opt_cost

            binding = {
                "einsums": [
                    einsum.get_name() for einsum in key],
                "templates": templates}

            bindings.append(binding)

        return YAMLGenerator({"matches": bindings})

    @staticmethod
    def find_all_matches(
            elaborated_einsum: ElaboratedEinsum,
            elaborated_arch: ElaboratedArchitecture) \
            -> Dict[Tuple[Einsum, ...], List[Tuple[ElaboratedTemplate, Optional[float]]]]:
        
        cascade = elaborated_einsum.get_einsums()
        outputs = elaborated_einsum.get_cascade_outputs()

        ccascade = []
        for decl_einsum in cascade:
            ccascade.extend(decl_einsum.split_for_solver())

                                      
        ceinsums_dict = {}
        for ceinsum in ccascade:
            ceinsums_dict[ceinsum.get_name()] = ceinsum

                                                               
        cs_cascade = Binder.__build_solver_cascade(ccascade)
        coutputs = [tensor.get_name() for tensor in outputs]

        matches: Dict[Tuple[Einsum, ...],
                      List[Tuple[ElaboratedTemplate, Optional[float]]]] = {}
        for engine_name, engine in elaborated_arch.get_engines().items():
            for template in engine.get_templates():
                instruction = template.get_instruction()

                                                                       
                tcascade = []
                for bound_einsum in instruction.get_einsums():
                    tcascade.extend(bound_einsum.split_for_solver())

                ts_cascade = Binder.__build_solver_cascade(tcascade)
                toutputs = template.get_output_names()

                solver = InstructionSolver(
                    cs_cascade, coutputs, ts_cascade, toutputs)

                                     
                for ceinsum in ccascade:
                    for teinsum in tcascade:
                        if ceinsum.is_form_match(
                                teinsum, engine_name, template.get_name()):
                            continue

                        solver.add_illegal_match(
                            [ceinsum.get_name()], [teinsum.get_name()])

                solver_matches = solver.find_all_einsum_matches()

                for match in solver_matches:
                    ematch = tuple([ceinsums_dict[einsum_name]
                                   for einsum_name in match])

                    if ematch not in matches:
                        matches[ematch] = []

                    matches[ematch].append((template, None))

        return matches

    @staticmethod
    def __build_solver_cascade(cascade: List[Einsum]) -> List[SolverEinsum]:
        
        return [einsum.get_reuse_spec().get_solver_repr()
                for einsum in cascade]
