

from typing import Any, Dict, List, Optional, Tuple

from src.common import *
from src.ir.binding.common.einsum import BoundEinsum
from src.ir.binding.common.fusion import BoundFusion
from src.ir.binding.common.instruction import CommonInstruction
from src.ir.binding.common.tensor import BoundTensor
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.format import *
from src.parse import *


class ElaboratedCommonBinding(Base):

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            elaborated_mapping: ElaboratedMapping,
            elaborated_format: ElaboratedFormat,
            binding_parser: BindingParser,
            memories: Dict[str, MemoryComponent]):

        self.conds: List[Condition] = []

        self.instructions = []
        map_einsums = elaborated_mapping.get_einsums()
        self.first_access: Dict[str, BoundTensor] = {}
        for engine_binding in binding_parser.get_engine_bindings():
            instruction = CommonInstruction(
                elaborated_einsum,
                elaborated_mapping,
                elaborated_format,
                binding_parser,
                engine_binding,
                memories)
            self.instructions.append(instruction)

        map_einsums = elaborated_mapping.get_einsums()
        self.instructions.sort(
            key=lambda inst: map_einsums.index(
                inst.get_einsums()[0].get_map_einsum()))

        corr_einsums = set(map_einsum.get_name() for map_einsum in map_einsums)
        given_einsums = [einsum for instruction in self.instructions
                         for einsum in instruction.get_einsum_names()]
        ErrorUtils.check_einsums(corr_einsums, given_einsums, "engine binding")

        bound_einsums = {}
        pos = 0
        for instruction in self.instructions:
            instruction_name = tuple(bound_einsum.get_name()
                                     for bound_einsum in instruction.get_einsums())
            engine = instruction.get_component()
            for bound_einsum in instruction.get_einsums():
                einsum = bound_einsum.get_name()
                bound_einsums[einsum] = (instruction_name, pos, bound_einsum)
                pos += 1

                for bound_tensor in bound_einsum.get_reuse_spec().get_tensors():
                    tensor = bound_tensor.get_name()

                                                                  
                    for memory, port in bound_tensor.get_memories():
                        memory.maybe_get_port(engine, port, einsum, tensor)

                                                         
                    if tensor not in self.first_access:
                        self.first_access[tensor] = bound_tensor
                        continue

                    self.first_access[tensor].match_with_reshape(bound_tensor)

                                                                            
                                   
                    num_unbound = len(
                        bound_tensor.get_rank_binding()["unbound"])
                    first_unbound = len(
                        self.first_access[tensor].get_rank_binding()["unbound"])
                    if num_unbound != first_unbound:
                        raise RankNumberMismatchError(
                            tensor, "unbound memory rank of the tensor binding", first_unbound, num_unbound)

        self.fusion = BoundFusion(
            elaborated_mapping.get_fusion(),
            bound_einsums,
            binding_parser)

    def check_concrete(self):
        
                                                                      
                                              
        outputs = set()
        for instruction in self.instructions:
            for bound_einsum in instruction.get_einsums():
                output = bound_einsum.get_reuse_spec().get_output_tensor().get_name()
                bound_einsum.check_concrete(output not in outputs)
                outputs.add(output)

    def check_template(self):
        
                                                                
        if len(self.instructions) > 1:
            raise MultipleEngineBindingError()

        for bound_einsum in self.instructions[0].get_einsums():
            bound_einsum.check_template()

    def get_instructions(self) -> List[CommonInstruction]:
        
        return self.instructions

    def get_first_access(self, tensor: str) -> BoundTensor:
        
        return self.first_access[tensor]

    def get_fusion(self) -> BoundFusion:
        
        return self.fusion

    def key(self) -> Tuple[Any, ...]:
        return self.instructions, self.fusion
