
from typing import Any, Dict, List, Tuple

from src.common import *
from src.ir.arch import *
from src.ir.binding.common import *
from src.ir.gen import *
from src.ir.partitioning import *
from src.parse import *


class Instruction(Base):
    def __init__(self, common_instruction: CommonInstruction,
                 elaborated_arch: ElaboratedArchitecture):
        self.common_instruction = common_instruction
        location = self.get_name() + " engine binding"

        engine = self.common_instruction.get_component()
        instruction = self.common_instruction.get_instruction()

        self.template = elaborated_arch.maybe_get_template(
            engine, instruction, location)

        self.lines = self.template.get_line_gen()

        templates = self.common_instruction.match_template(
            self.template.get_instruction())

        for update_info in templates:
                                                                        
                                                     
            if update_info[0] == "tensor":
                continue

            for line in self.lines:
                line.update(*update_info)

    def get_cost_metrics(self) -> List[CostMetricEnum]:
        
        cost_metrics = self.template.get_cost_metrics()
        if cost_metrics is None:
            GlobalState.set_location_prefix(
                self.get_engine() +
                "'s " +
                self.get_instruction() +
                " template")
            raise MissingFieldError("cost-metrics", "attributes")

        return cost_metrics

    def get_einsums(self) -> List[BoundEinsum]:
        
        return self.common_instruction.get_einsums()

    def get_einsum_names(self) -> List[str]:
        
        return self.common_instruction.get_einsum_names()

    def get_engine(self) -> str:
        
        return self.common_instruction.get_component()

    def get_frequency(self) -> Optional[float]:
        
        return self.template.get_frequency()

    def get_line_gen(self) -> List[LineGeneration]:
        
        return self.lines

    def get_instruction(self) -> str:
        
        return self.common_instruction.get_instruction()

    def get_instruction_init_latency(self) -> Optional[int]:
        
        return self.template.get_instruction_init_latency()

    def get_instruction_latency(self) -> Optional[int]:
        
        return self.template.get_instruction_latency()

    def get_instruction_overhead(self) -> int:
        
        return self.template.get_instruction_overhead()

    def get_name(self) -> str:
        
        return self.common_instruction.get_name()

    def get_op_init_latency(self, einsum: BoundEinsum) -> Optional[int]:
        
        i = self.get_einsums().index(einsum)
        teinsum = self.template.get_instruction().get_einsums()[i].get_name()
        return self.template.get_op_init_latency(teinsum)

    def maybe_get_einsum(self, einsum: str, location: str) -> BoundEinsum:
        
        return self.common_instruction.maybe_get_einsum(einsum, location)

    def maybe_get_tensor_accesses(
            self,
            template_tensor: str,
            location: str) -> List[BoundTensorAccess]:
        
        tensor_access_locs = []
        for e, teinsum in enumerate(
                self.template.get_instruction().get_einsums()):
            for t, ttensor in enumerate(
                    teinsum.get_reuse_spec().get_tensors()):
                if ttensor.get_name() == template_tensor:
                    tensor_access_locs.append((e, t))

        if not tensor_access_locs:
            GlobalState.set_location_prefix(
                self.template.get_component() +
                "'s " +
                self.template.get_name() +
                " template")
            raise UnknownTensorError(template_tensor, "code generation")

        tensor_accesses = []
        for e, t in tensor_access_locs:
            ceinsum = self.common_instruction.get_einsums()[e]
            tensor_accesses.append(
                ceinsum.get_reuse_spec().get_tensor_accesses()[t])

        return tensor_accesses

    def key(self) -> Tuple[Any, ...]:
        return self.common_instruction, self.lines

    def __hash__(self) -> int:
        
        return hash(("Instruction", tuple(self.get_einsum_names())))
