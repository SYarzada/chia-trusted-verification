
from typing import Any, Dict, List, Tuple

from src.common import *
from src.ir.arch import *
from src.ir.binding.common import *
from src.ir.binding.concrete.instruction import Instruction
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.parse import *


class ElaboratedBinding(Base):
    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            elaborated_mapping: ElaboratedMapping,
            elaborated_format: ElaboratedFormat,
            binding_parser: BindingParser,
            elaborated_arch: ElaboratedArchitecture):

        self.elaborated_mapping = elaborated_mapping
        self.common_binding = ElaboratedCommonBinding(
            elaborated_einsum,
            elaborated_mapping,
            elaborated_format,
            binding_parser,
            elaborated_arch.get_memories())
        self.common_binding.check_concrete()

        self.instructions = []
        self.positions = {}
        self.einsums = {}

        for i, common_instruction in enumerate(
                self.common_binding.get_instructions()):
            self.instructions.append(
                Instruction(common_instruction, elaborated_arch))

            for bound_einsum in common_instruction.get_einsums():
                self.positions[bound_einsum.get_name()] = i
                self.einsums[bound_einsum.get_name()] = bound_einsum

    def get_einsum(self, einsum: str) -> BoundEinsum:
        
        return self.einsums[einsum]

    def get_first_access(self, tensor: str) -> BoundTensor:
        
        return self.common_binding.get_first_access(tensor)

    def get_fusion(self) -> BoundFusion:
        
        return self.common_binding.get_fusion()

    def get_instructions(self) -> List[Instruction]:
        
        return self.instructions

    def get_position(self, einsum: str) -> int:
        
        return self.positions[einsum]

    def key(self) -> Tuple[Any, ...]:
        return self.instructions,
