

from typing import Any, Dict, List, Tuple

from src.common import *
from src.ir.binding.common.einsum import BoundEinsum
from src.ir.binding.common.tensor_access import BoundTensorAccess
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.solver import *
from src.parse import *


class CommonInstruction(Base):
    

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            elaborated_mapping: ElaboratedMapping,
            elaborated_format: ElaboratedFormat,
            binding_parser: BindingParser,
            engine_binding: EngineBindingParser,
            memories: Dict[str, MemoryComponent]):

        self.elaborated_einsum = elaborated_einsum

        self.einsums = []
        self.einsums_dict = {}
        assert engine_binding.einsums is not None
        for einsum in engine_binding.einsums:
            map_einsum = elaborated_mapping.maybe_get_einsum(
                einsum, "engine binding")
            bound_einsum = BoundEinsum(
                elaborated_einsum,
                elaborated_mapping,
                map_einsum,
                elaborated_format,
                binding_parser,
                memories)
            self.einsums.append(bound_einsum)
            self.einsums_dict[bound_einsum.get_name()] = bound_einsum

        self.component = engine_binding.component
        self.instruction = engine_binding.instruction

        fusion = elaborated_mapping.get_fusion()
        all_unbound_ranks = [bound_einsum.get_unbound_ranks()
                             for bound_einsum in self.einsums]
        num_unbound_ranks = [len(unbound_ranks)
                             for unbound_ranks in all_unbound_ranks]
        if len(set(num_unbound_ranks)) > 1:
            raise UnboundRankNumberMismatchError(
                self.get_name(), num_unbound_ranks)

                                                               
        for map_ranks in zip(*all_unbound_ranks):
            all_fused_ranks = [fusion.get_fused_ranks(
                map_rank) for map_rank in map_ranks]
            if len(set(all_fused_ranks)) > 1:
                ranks = [map_rank.get_scoped_name()
                         for map_rank in map_ranks]
                raise MissingFusionError(self.get_name(), ranks)

                                                                           
                                          
        for bound_einsum in self.einsums:
            for map_rank in bound_einsum.get_bound_ranks():
                fused_ranks = fusion.get_fused_ranks(map_rank)

                for fused_rank in fused_ranks:
                    if fused_rank.get_einsum() not in self.einsums_dict.keys():
                        ranks = [rank.get_scoped_name()
                                 for rank in [map_rank, fused_rank]]
                        location = "[" + ", ".join(
                            map_einsum.get_name() for map_einsum in fusion.get_group(
                                bound_einsum.get_map_einsum())) + "]"
                        reason = "the Einsums are bound to different instructions"
                        raise FusionRankMismatchError(ranks, location, reason)

    def get_component(self) -> str:
        
        return self.component

    def get_einsums(self) -> List[BoundEinsum]:
        
        return self.einsums

    def get_einsum_names(self) -> List[str]:
        
        return [einsum.get_name() for einsum in self.einsums]

    def get_instruction(self) -> str:
        
        return self.instruction

    def get_name(self) -> str:
        
        einsums = [bound_einsum.get_name() for bound_einsum in self.einsums]
        return "[" + ", ".join(einsums) + "]"

    def match_template(
            self, other: "CommonInstruction") -> List[Tuple[Optional[str], str, str]]:
        
        assert self.get_component() == other.get_component()
        assert self.get_instruction() == other.get_instruction()

                                                                      
        if len(self.get_einsums()) != len(other.get_einsums()):
            raise EinsumNumberMismatchError(
                self.get_name(),
                self.get_component(),
                self.get_instruction(),
                len(other.get_einsums()),
                len(self.get_einsums()))

                                              
        templates = []
        for ceinsum, teinsum in zip(self.get_einsums(), other.get_einsums()):
            templates.extend(ceinsum.match_template_form(teinsum))

                                                          
        ceinsums = self.get_einsum_names()

        cascades = []
        outputs = []
        for instruction in [ self, other]:
            cascade = []
            local_outputs = []
            for einsum in instruction.get_einsums():
                cascade.append(einsum.get_reuse_spec().get_solver_repr())
                local_outputs.append(einsum.get_reuse_spec().get_output_tensor().get_name())
            cascades.append(cascade)

            instr_outputs = []
            for output in instruction.elaborated_einsum.get_cascade_outputs():
                if output.get_name() in local_outputs:
                    instr_outputs.append(output.get_name())

            for decl_einsum in instruction.elaborated_einsum.get_einsums():
                if decl_einsum.get_name() in instruction.get_einsum_names():
                    continue

                for decl_tensor in decl_einsum.get_reuse_spec().get_tensors():
                    tensor = decl_tensor.get_name()
                    if tensor not in instr_outputs and tensor in local_outputs:
                        instr_outputs.append(tensor)

            outputs.append(instr_outputs)

        solver = InstructionSolver(
            cascades[0], outputs[0], cascades[1], outputs[1])

                                                   
        result = solver.match_template(ceinsums)
        if not result:
            raise TemplateMismatchError(
                ceinsums,
                "expressions or cascade outputs",
                self.component,
                self.instruction,
                None,
                None)

                               

        return templates

    def maybe_get_einsum(self, einsum: str, location: str) -> BoundEinsum:
        
        if einsum not in self.einsums_dict.keys():
            raise UnknownEinsumError(einsum, location)

        return self.einsums_dict[einsum]

    def key(self) -> Tuple[Any, ...]:
        return self.einsums, self.component, self.instruction

    def __hash__(self) -> int:
        
        return hash(("CommonInstruction", tuple(self.get_einsum_names())))
