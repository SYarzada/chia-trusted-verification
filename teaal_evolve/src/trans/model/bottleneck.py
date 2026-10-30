
from sympy import Basic, Symbol                
from typing import Any, Dict, Tuple

from src.common import *
from src.ir import *
from src.parse import *
from src.trans.model.instruction import InstructionModel
from src.trans.yaml import *


class BottleneckAnalysisModel:
    

    def __init__(self, elaborated_binding: ElaboratedBinding,
                 shapes: Dict[str, int]):
        self.binding = elaborated_binding
        self.shapes = {Symbol(name): value for name, value in shapes.items()}

    def display_latency(
            self, latencies: Tuple[float, Dict[Tuple[str, ...], Dict[str, float]]]) -> YAMLGenerator:
        
        spec: Dict[str, Any] = {"total-latency": latencies[0], "breakdown": []}
        for group_tuple, engine_latencies in latencies[1].items():
            group = "[" + ", ".join(group_tuple) + "]"
            spec["breakdown"].append(
                {"einsums": group, "latencies": engine_latencies})

        return YAMLGenerator(spec)

    def get_latency(self) -> Tuple[float,
                                   Dict[Tuple[str, ...], Dict[str, float]]]:
        

        latencies: Dict[Tuple[BoundEinsum, ...], Dict[str, Basic]] = {}
        for instruction in self.binding.get_instructions():
            cost_metrics = instruction.get_cost_metrics()
            if not cost_metrics:
                continue

                                                                       
                                                     
            bound_einsum = instruction.get_einsums()[0]
            group = self.binding.get_fusion().get_group(bound_einsum)
            if group not in latencies.keys():
                latencies[group] = {}

            num_calls = InstructionModel.get_num_instruction_calls(
                instruction, self.shapes)

            if CostMetricEnum.compute in cost_metrics:
                engine = instruction.get_engine()
                engine_latencies = InstructionModel.get_per_instr_compute_latency(
                    instruction, self.shapes)

                for engine, latency_info in engine_latencies.items():
                    if engine not in latencies[group].keys():
                        latencies[group][engine] = 0

                    latencies[group][engine] += latency_info[0] * num_calls

            if CostMetricEnum.memory in cost_metrics:
                memory_latencies = InstructionModel.get_per_instr_mem_latency(
                    instruction, self.shapes)
                for memory, access_latency in memory_latencies.items():
                    if memory not in latencies[group].keys():
                        latencies[group][memory] = 0

                    latencies[group][memory] += access_latency * num_calls

                                                                          
        total_latency = 0.0
        latency_breakdown: Dict[Tuple[str, ...], Dict[str, float]] = {}
        for group, group_latencies in latencies.items():
            group_name = tuple(bound_einsum.get_name()
                               for bound_einsum in group)
            latency_breakdown[group_name] = {}

            group_latency = 0.0
            for component, latency in group_latencies.items():
                free_symbols = getattr(latency, "free_symbols", set())
                if free_symbols:
                    shapes = ", ".join(sorted(str(symbol) for symbol in free_symbols))
                    message = "all shapes must be concretized to calculate latency"
                    raise IllegalSymbolicShapeError(shapes, "given", message)

                numeric_latency = float(latency)
                latency_breakdown[group_name][component] = numeric_latency
                if numeric_latency > group_latency:
                    group_latency = numeric_latency

            total_latency += group_latency

        return total_latency, latency_breakdown
