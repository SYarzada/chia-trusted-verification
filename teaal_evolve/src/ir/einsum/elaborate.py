

from sympy import Symbol                
from typing import Dict, List, Optional, Set

from src.common import *
from src.ir.cond import *
from src.ir.einsum.einsum import DeclaredEinsum
from src.ir.einsum.rank import DeclaredRank
from src.ir.einsum.tensor import DeclaredTensor
from src.parse import *


class ElaboratedEinsum:

    def __init__(self, einsum_parser: EinsumParser,
                 options_parser: OptionsParser):
        self.ranks = {name: DeclaredRank(name, shape)
                      for name, shape in einsum_parser.get_shapes().items()}

        self.tensors = {}
        self.cascade_inputs = []
        for tensor_name, rank_names in einsum_parser.get_declaration().items():

            for rank_name in rank_names:
                if rank_name in self.ranks.keys():
                    continue

                self.ranks[rank_name] = DeclaredRank(
                    rank_name, Symbol(rank_name))

            tensor_ranks = [self.ranks[rank_name] for rank_name in rank_names]

            self.tensors[tensor_name] = DeclaredTensor(
                tensor_name, tensor_ranks)
            self.cascade_inputs.append(self.tensors[tensor_name])

        all_outputs = set()

        self.einsums = []
        tensors_used = set()
        names: Dict[str, int] = {}
        for i, (opt_name, expr) in enumerate(einsum_parser.get_expressions()):
            einsum = DeclaredEinsum(opt_name, expr, self.ranks, self.tensors)

            name = einsum.get_name()
            if name in names.keys():
                raise DuplicateEinsumNameError(name, names[name], i)
            names[name] = i

            output = einsum.get_reuse_spec().get_output_tensor()
            if output in self.cascade_inputs:
                self.cascade_inputs.remove(output)

                                                                           

            self.einsums.append(einsum)
            all_outputs.add(output)
            tensors_used.update(einsum.get_reuse_spec().get_tensors())

        if len(self.einsums) < 1:
            raise MissingEinsumError(None, "expressions")

                                                                         
        unused_tensors = list(
            set(self.tensors.values()).difference(tensors_used))
        if unused_tensors:
            raise UnusedTensorError(unused_tensors[0].get_name())

                                                                        
        fields = options_parser.get_fields()
                                                      
        if "cascade-outputs" not in fields.keys():
            raise MissingFieldError("cascade-outputs", "options")

        self.cascade_outputs = []
        for tensor in fields["cascade-outputs"]:
                                                             
            if tensor not in self.tensors.keys():
                raise UnknownTensorError(tensor, "cascade outputs")

                                                                  
            if self.tensors[tensor] not in all_outputs:
                raise IllegalOutputTensorError(tensor, "cascade outputs")

            self.cascade_outputs.append(self.tensors[tensor])

                                               
        for decl_tensor in self.cascade_inputs:
            decl_tensor.set_is_cascade_input(True)

        for decl_tensor in self.cascade_outputs:
            decl_tensor.set_is_cascade_output(True)

    def check_concrete(self):
        
        for einsum in self.einsums:
            einsum.check_concrete()

    def check_template(self):
        
        for einsum in self.einsums:
            einsum.check_template()

    def get_conds(self) -> List[Condition]:
        
        conds = []
        for einsum in self.einsums:
            for cond in einsum.get_conds():
                if cond not in conds:
                    conds.append(cond)

        return conds

    def get_cascade_inputs(self) -> List[DeclaredTensor]:
        
        return self.cascade_inputs

    def get_cascade_outputs(self) -> List[DeclaredTensor]:
        
        return self.cascade_outputs

    def get_einsums(self) -> List[DeclaredEinsum]:
        
        return self.einsums

    def get_ranks(self) -> Dict[str, DeclaredRank]:
        
        return self.ranks

    def get_symbols(self) -> List[Symbol]:
        
        return [rank.get_shape() for rank in self.ranks.values()
                if not isinstance(rank.get_shape(), int)]

    def get_tensors(self) -> Dict[str, DeclaredTensor]:
        
        return self.tensors
