
from typing import Any, Dict, List, Optional, Tuple

from src.common import *
from src.ir.binding.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.gen import *
from src.ir.mapping import *
from src.ir.memory import *
from src.parse import *


class ElaboratedTemplate(Base):
    

    def __init__(self,
                 component: str,
                 default_attributes: Optional[TemplateAttributesParser],
                 template_parser: TemplateParser,
                 memories: Dict[str,
                                MemoryComponent]):
        self.component = component
        self.name = template_parser.get_name()
        self.attributes = template_parser.get_attributes()
        self.default_attributes = default_attributes

        GlobalState.set_location_prefix(
            self.component + "'s " + self.name + " template")

        self.elaborated_einsum = ElaboratedEinsum(
            template_parser.get_einsum(), template_parser.get_options())
        self.elaborated_einsum.check_template()

        elaborated_mapping = ElaboratedMapping(
            self.elaborated_einsum, template_parser.get_mapping())
        elaborated_mapping.check_template()

        elaborated_format = ElaboratedFormat(
            self.elaborated_einsum,
            elaborated_mapping,
            template_parser.get_format())
        elaborated_format.check_template()

        elaborated_binding = ElaboratedCommonBinding(
            self.elaborated_einsum,
            elaborated_mapping,
            elaborated_format,
            template_parser.get_binding(),
            memories)
        elaborated_binding.check_template()

        self.instruction = elaborated_binding.get_instructions()[0]

        einsums = self.instruction.get_einsum_names()
        if self.instruction.get_component() != self.component:
            raise ComponentMismatchError(
                "component name",
                "engine binding",
                self.component,
                self.instruction.get_component())

        location = "[" + ", ".join(einsums) + "] engine binding"
        if self.instruction.get_instruction() != self.name:
            raise InstructionMismatchError(
                self.name,
                self.instruction.get_instruction(),
                location)

        self.lines = [LineGeneration(tree)
                      for tree in template_parser.get_code_gen()]

        GlobalState.clear_location_prefix()

    def get_component(self) -> str:
        
        return self.component

    def get_cost_metrics(self) -> Optional[List[CostMetricEnum]]:
        
        return self.__get_attribute("cost_metrics")

    def get_instruction(self) -> CommonInstruction:
        
        return self.instruction

    def get_frequency(self) -> Optional[float]:
        
        return self.__get_attribute("frequency")

    def get_line_gen(self) -> List[LineGeneration]:
        
        return [line.copy_blank() for line in self.lines]

    def get_instruction_init_latency(self) -> Optional[int]:
        
        return self.__get_attribute("instruction_init_latency")

    def get_instruction_latency(self) -> Optional[int]:
        
        return self.__get_attribute("instruction_latency")

    def get_instruction_overhead(self) -> int:
        
        opt_overhead = self.__get_attribute("instruction_overhead")
        if opt_overhead:
            return opt_overhead
        return 0

    def get_name(self) -> str:
        
        return self.name

    def get_op_init_latency(self, einsum: str) -> Optional[int]:
        
        einsums = {bound_einsum.get_name()
                   for bound_einsum in self.instruction.get_einsums()}
        if einsum not in einsums:
            return None

        op_init_latency = self.__get_attribute("op_init_latency")
        if op_init_latency is None or isinstance(op_init_latency, int):
            return op_init_latency

                                           

        if einsum not in op_init_latency.keys():
            return None

        return op_init_latency[einsum]

    def get_output_names(self) -> List[str]:
        
        return [output.get_name()
                for output in self.elaborated_einsum.get_cascade_outputs()]

    def __get_attribute(self, attribute: str) -> Any:
        
        has_attrs = self.attributes is not None
        has_default_attrs = self.default_attributes is not None

        if not has_attrs and not has_default_attrs:
            GlobalState.set_location_prefix(
                self.component + "'s " + self.name + " template")
            raise MissingFieldError("attributes", "given")

        if has_attrs and getattr(self.attributes, attribute) is not None:
            return getattr(self.attributes, attribute)

        if has_default_attrs:
            return getattr(self.default_attributes, attribute)

        return None

    def key(self) -> Tuple[Any, ...]:
        return self.component, self.name, self.attributes, self.instruction, self.lines
