
from itertools import permutations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import z3                

from src.ir.solver.utils import SolverUtils

"""
Cascades should be of the form:
List[Tuple[einsum_name, List[Tuple[tensor_name, List[List[rank_term]]]]]]
"""
SolverEinsum = Tuple[str, List[Tuple[str, List[List[Union[str, int]]]]]]


class InstructionSolver:
    solver_num = 0

                                                                             
                         
    def __init__(
            self,
            concrete_cascade: List[SolverEinsum],
            concrete_outputs: List[str],
            template_cascade: List[SolverEinsum],
            template_outputs: List[str]):
        
        self.solver = z3.Solver()

        self.num = str(InstructionSolver.__get_num())

        self.concrete_info = self.__add_cascade(
            "concrete" + self.num, concrete_cascade, concrete_outputs)
        self.template_info = self.__add_cascade(
            "template" + self.num, template_cascade, template_outputs)

        self.__build_solver()

    def add_illegal_match(
            self,
            concrete_einsums: List[str],
            template_einsums: List[str]):
        
        assert len(concrete_einsums) == len(template_einsums)

        ceinsum_dict = self.concrete_info["einsum_dict"]
        teinsum_dict = self.template_info["einsum_dict"]
        einsum_match_fn = self.match_info["einsum_match_fn"]

        illegal_match = []
        for ceinsum, teinsum in zip(concrete_einsums, template_einsums):
            illegal_match.append(
                einsum_match_fn(teinsum_dict[teinsum]) ==
                ceinsum_dict[ceinsum])

        self.solver.add(z3.Not(z3.And(*illegal_match)))

    def find_all_einsum_matches(self) -> Set[Tuple[str, ...]]:
        
        matches = set()
        result = self.solver.check()

        einsum_match_fn = self.match_info["einsum_match_fn"]

        teinsums = [str(teinsum) for teinsum in self.template_info["einsums"]]

        while result == z3.sat:
            model = self.solver.model()

            ceinsums = []
            for teinsum in self.template_info["einsums"]:
                ceinsums.append(
                    str(model.evaluate(einsum_match_fn(teinsum))))

            matches.add(tuple(ceinsums))

            self.add_illegal_match(ceinsums, teinsums)

            result = self.solver.check()

        return matches

    def find_all_rank_matches(self, trank: str) -> Set[str]:
        
        matches = set()
        result = self.solver.check()

        rank_match_fn = self.match_info["rank_match_fn"]
        trank_dict = self.template_info["rank_dict"]
        crank_dict = self.concrete_info["rank_dict"]

        matched_checks = []

        while result == z3.sat:
            model = self.solver.model()

            crank = str(model.evaluate(rank_match_fn(trank_dict[trank])))
            matches.add(crank)

            matched_checks.append(
                rank_match_fn(trank_dict[trank]) != crank_dict[crank])

            result = self.solver.check(z3.And(*matched_checks))

        return matches

    def match_template(
            self, concrete_einsums: List[str]) -> bool:
        
        cinfo = self.concrete_info
        tinfo = self.template_info
        minfo = self.match_info

                                                            
        assert len(concrete_einsums) == len(tinfo["einsums"])

                                                             
        checks = []
        einsum_match_fn = minfo["einsum_match_fn"]
        for ceinsum_name, teinsum in zip(concrete_einsums, tinfo["einsums"]):
            ceinsum = cinfo["einsum_dict"][ceinsum_name]
            checks.append(einsum_match_fn(teinsum) == ceinsum)

        result = self.solver.check(z3.And(*checks))

        return result == z3.sat

    def __add_cascade(self,
                      cascade_name: str,
                      cascade: List[SolverEinsum],
                      outputs: List[str]) -> Dict[str, Any]:
        
                                
        einsum_names = [einsum[0] for einsum in cascade]
        Einsum, einsums = z3.EnumSort(cascade_name + "_Einsum", einsum_names)

                                               
        tensor_names = []
        rank_names = []
        for einsum in cascade:
            for tensor in einsum[1]:
                if tensor[0] not in tensor_names:
                    tensor_names.append(tensor[0])

                for rank_var_expr in tensor[1]:
                    for term in rank_var_expr:
                        if isinstance(term, str) and term.islower() and \
                                term not in rank_names:
                            rank_names.append(term.upper())

        if not rank_names:
            rank_names = ["None"]

                                          
        Tensor, tensors = z3.EnumSort(cascade_name + "_Tensor", tensor_names)
        Rank, ranks = z3.EnumSort(cascade_name + "_Rank", rank_names)
        Ranks = z3.SeqSort(Rank)
        Tensors = z3.SeqSort(Tensor)

                                     
        einsum_dict = {
            einsum_name: einsum for einsum_name,
            einsum in zip(
                einsum_names,
                einsums)}
        tensor_dict = {
            tensor_name: tensor for tensor_name,
            tensor in zip(
                tensor_names,
                tensors)}
        rank_dict = {
            rank_name: rank for rank_name,
            rank in zip(
                rank_names,
                ranks)}

                                                         
        RanksInput, mkRI, RI_accessors = z3.TupleSort(
            cascade_name + "_RanksInput", [Einsum, z3.IntSort(), z3.IntSort()])
        access_fn = z3.Function(cascade_name + "_ranks", RanksInput, Ranks)
        const_fn = z3.Function(
            cascade_name + "_const",
            RanksInput,
            z3.IntSort())
        tensors_fn = z3.Function(cascade_name + "_tensors", Einsum, Tensors)

                                                    
        tensors_fn_py = []
        ranks_fn_py: List[List[Set[z3.DatatypeSortRef]]] = []
        access_fn_py: List[List[List[List[z3.DatatypeSortRef]]]] = []
        inputs = tensor_names.copy()
        for einsum_input in cascade:
            einsum = einsum_dict[einsum_input[0]]

            einsum_tensors = []
            ranks_fn_py.append([])
            access_fn_py.append([])
            for i, tensor_input in enumerate(einsum_input[1]):
                                                        
                if i == 0 and tensor_input[0] in inputs:
                    inputs.remove(tensor_input[0])

                tensor = tensor_dict[tensor_input[0]]
                einsum_tensors.append(tensor)

                access_fn_py[-1].append([])
                tensor_ranks = set()
                for j, rank_var_expr in enumerate(tensor_input[1]):
                    expr_ranks = []
                    const = 0
                    for term in rank_var_expr:
                                                              
                        if isinstance(term, int):
                            const += term
                            continue

                                                                            
                                    
                        if term.isupper():
                            const += z3.Int(term)
                            continue

                        expr_ranks.append(rank_dict[term.upper()])
                        tensor_ranks.add(rank_dict[term.upper()])

                    ranks_input = mkRI(einsum, i, j)

                                                   
                    self.solver.add(
                        access_fn(ranks_input) ==
                        SolverUtils.mk_seq(expr_ranks, Ranks))

                                                             
                    self.solver.add(
                        const_fn(ranks_input) == const)

                    access_fn_py[-1][-1].append(expr_ranks)

                ranks_fn_py[-1].append(tensor_ranks)

            self.solver.add(
                tensors_fn(einsum) == SolverUtils.mk_seq(
                    einsum_tensors, Tensors))
            tensors_fn_py.append(einsum_tensors)

        z3_outputs = [tensor_dict[tensor_name] for tensor_name in outputs]
        intermediates = [tensor_dict[tensor_name]
                         for tensor_name in tensor_names if tensor_name not in outputs and tensor_name not in inputs]

        info = {}
        info["Einsum"] = Einsum
        info["einsums"] = einsums
        info["einsum_dict"] = einsum_dict
        info["Tensor"] = Tensor
        info["Tensors"] = Tensors
        info["tensors"] = tensors
        info["tensor"] = tensor_dict
        info["outputs"] = z3_outputs
        info["intermediates"] = intermediates
        info["Rank"] = Rank
        info["Ranks"] = Ranks
        info["ranks"] = ranks
        info["rank_dict"] = rank_dict
        info["RanksInput"] = RanksInput
        info["mkRI"] = mkRI
        info["RI_accessors"] = RI_accessors
        info["access_fn"] = access_fn
        info["const_fn"] = const_fn
        info["tensors_fn"] = tensors_fn
        info["ranks_fn_py"] = ranks_fn_py
        info["access_fn_py"] = access_fn_py
        info["tensors_fn_py"] = tensors_fn_py

        return info

    def __build_solver(self):
        
        cinfo = self.concrete_info
        tinfo = self.template_info

                                                                          
                                
        rank_match_fn = z3.Function(
            "rank_match" + self.num,
            tinfo["Rank"],
            cinfo["Rank"])
        access_match_fn = z3.Function(
            "access_match" + self.num,
            tinfo["RanksInput"],
            cinfo["RanksInput"])
        tensor_match_fn = z3.Function(
            "tensor_match" + self.num,
            tinfo["Tensor"],
            cinfo["Tensor"])
        einsum_match_fn = z3.Function(
            "einsum_match" + self.num,
            tinfo["Einsum"],
            cinfo["Einsum"])

                                                                  
        ConstInput, mkCI, CI_accessors = z3.TupleSort(
            "ConstInput" + self.num,
            [tinfo["Tensor"], z3.IntSort()])
        offset_fn = z3.Function(
            "offset" + self.num, ConstInput, z3.IntSort())

                                                                          
        mismatch_output = InstructionSolver.__mismatch_all(
            tensor_match_fn, tinfo["intermediates"], cinfo["outputs"])
        self.solver.add(mismatch_output)

        for ci, ceinsum in enumerate(cinfo["einsums"]):
                                                                          
                                                           
            not_einsum_match = [einsum_match_fn(teinsum) != ceinsum
                                for teinsum in tinfo["einsums"]]

            mismatch_other = InstructionSolver.__mismatch_all(
                tensor_match_fn, tinfo["intermediates"], cinfo["tensors_fn_py"][ci])
            self.solver.add(z3.Implies(
                z3.And(*not_einsum_match), mismatch_other))

            for ti, teinsum in enumerate(tinfo["einsums"]):
                                                                 
                einsum_match = einsum_match_fn(teinsum) == ceinsum

                ttensors = tinfo["tensors_fn"](teinsum)
                ctensors = cinfo["tensors_fn"](ceinsum)
                tensor_matches = [
                    z3.Length(ttensors) == z3.Length(ctensors)]

                tc_tensors_py = (
                    tinfo["tensors_fn_py"][ti],
                    cinfo["tensors_fn_py"][ci])

                for ttensor, ctensor in zip(*tc_tensors_py):
                    tensor_matches.append(
                        tensor_match_fn(ttensor) == ctensor)

                self.solver.add(
                    z3.Implies(einsum_match,
                               z3.And(*tensor_matches)))

                                                                      
                       
                for tci, (ttensor, ctensor) in enumerate(
                        zip(*tc_tensors_py)):
                    tensor_match = tensor_match_fn(ttensor) == ctensor

                    for tai, tranks in enumerate(
                            tinfo["access_fn_py"][ti][tci]):
                        taccess = tinfo["mkRI"](teinsum, tci, tai)

                        access_matches = []
                        for cai, cranks in enumerate(
                                cinfo["access_fn_py"][ci][tci]):
                            caccess = cinfo["mkRI"](ceinsum, tci, cai)

                            access_match = access_match_fn(
                                taccess) == caccess
                            access_matches.append(access_match)

                                                                       
                                                                          
                            self.solver.add(
                                InstructionSolver.__match_one_to_one(
                                    access_match,
                                    rank_match_fn,
                                    tranks,
                                    cranks))

                                                                       
                                                                          
                                    
                            cconst = cinfo["const_fn"](caccess)
                            tconst = tinfo["const_fn"](taccess)
                            const_input = mkCI(ttensor, tai)
                            self.solver.add(z3.Implies(
                                access_match,
                                cconst == offset_fn(const_input) + tconst))

                                                                         
                        self.solver.add(z3.Implies(
                            einsum_match,
                            z3.Implies(tensor_match, z3.Or(*access_matches))))

                                                                      
                    for trank in tinfo["ranks"]:
                        if trank in tinfo["ranks_fn_py"][ti][tci]:
                            continue

                        for crank in cinfo["ranks_fn_py"][ci][tci]:
                            self.solver.add(z3.Implies(
                                einsum_match,
                                z3.Implies(
                                    tensor_match,
                                    rank_match_fn(trank) != crank)))

        self.match_info = {}
        self.match_info["einsum_match_fn"] = einsum_match_fn
        self.match_info["tensor_match_fn"] = tensor_match_fn
        self.match_info["access_match_fn"] = access_match_fn
        self.match_info["rank_match_fn"] = rank_match_fn

    @classmethod
    def __get_num(cls) -> int:
        
        num = cls.solver_num

        cls.solver_num += 1

        return num

    @staticmethod
    def __mismatch_all(match_fn: z3.FuncDeclRef,
                       templates: List[z3.DatatypeSortRef],
                       concretes: List[z3.DatatypeSortRef]) -> z3.BoolRef:
        
        no_match = []
        for concrete in concretes:
            for template in templates:
                no_match.append(match_fn(template) != concrete)

        return z3.And(*no_match)

    @staticmethod
    def __match_one_to_one(parent_match_cond: z3.BoolRef,
                           child_match_fn: z3.FuncDeclRef,
                           templates: List[z3.DatatypeSortRef],
                           concretes: List[z3.DatatypeSortRef]) -> z3.BoolRef:
        
                                                                      
        if len(templates) != len(concretes):
            return z3.Not(parent_match_cond)

        perm_matches = []
        for perm in permutations(concretes, len(concretes)):
            elem_match = []
            for template, concrete in zip(templates, perm):
                elem_match.append(child_match_fn(template) == concrete)

            perm_matches.append(z3.And(elem_match))

                                     
        return z3.Implies(parent_match_cond, z3.Or(*perm_matches))
