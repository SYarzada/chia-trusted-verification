

from typing import Dict

from src.common import *
from src.ir import *


class InstructionModel:
    
    @staticmethod
    def get_per_instr_compute_latency(
            instruction: Instruction, shapes: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
        
        frequency = instruction.get_frequency()
        if frequency is None:
            GlobalState.set_location_prefix(
                instruction.get_engine() +
                "'s " +
                instruction.get_instruction() +
                " template")
            raise MissingFieldError("frequency", "attributes")

        latencies = {}
        engine_latency = instruction.get_instruction_overhead() / frequency
        final_op_init_latency = 0.0
        for bound_einsum in instruction.get_einsums():
            sequential_ops = 1
            for map_rank in bound_einsum.get_bound_ranks():
                if not map_rank.get_is_spatial():
                    sequential_ops *= map_rank.get_shape()

            if not isinstance(sequential_ops, int):
                sequential_ops = sequential_ops.subs(shapes)

            op_init_latency = instruction.get_op_init_latency(bound_einsum)
            if op_init_latency is None:
                GlobalState.set_location_prefix(
                    instruction.get_engine() +
                    "'s " +
                    instruction.get_instruction() +
                    " template")
                raise MissingFieldError("op-init-latency", "attributes")

            engine_latency += sequential_ops * op_init_latency / frequency
            final_op_init_latency = op_init_latency / frequency

        min_cycles = instruction.get_instruction_init_latency()
        if min_cycles:
            engine_latency = max(engine_latency, min_cycles / frequency)

        data_latency = engine_latency
        instruction_latency = instruction.get_instruction_latency()
        if instruction_latency is not None:
            data_latency -= final_op_init_latency
            data_latency += instruction_latency / frequency

        engine = instruction.get_engine()
        latencies[engine] = (engine_latency, data_latency)

        for bound_einsum in instruction.get_einsums():
            for bound_tensor in bound_einsum.get_reuse_spec().get_tensors():
                memory = bound_tensor.get_memory()
                port_info = memory.maybe_get_port(
                    engine,
                    bound_tensor.get_port(),
                    bound_einsum.get_name(),
                    bound_tensor.get_name())

                if port_info and port_info.name not in latencies:
                    latencies[port_info.name] = (engine_latency, data_latency)

        return latencies

    @staticmethod
    def get_per_instr_mem_latency(
            instruction: Instruction, shapes: Dict[str, int]) -> Dict[str, float]:
        
        memories = []
        mem_latencies: Dict[str, float] = {}

        for bound_einsum in instruction.get_einsums():
            bound_ranks = bound_einsum.get_bound_ranks()
            bound_part_ranks = [map_rank.get_rank()
                                for map_rank in bound_ranks]

            for bound_tensor_access in bound_einsum.get_reuse_spec().get_tensor_accesses():
                memory_component = bound_tensor_access.get_tensor().get_memory()
                memory = memory_component.get_name()

                if memory not in memories:
                    memories.append(memory)

                bandwidth = memory_component.maybe_get_bandwidth(
                    instruction.get_engine(),
                    bound_tensor_access.get_tensor().get_port(),
                    bound_einsum.get_name(),
                    bound_tensor_access.get_tensor().get_name())

                if not bandwidth:
                    continue

                if memory not in mem_latencies.keys():
                    mem_latencies[memory] = 0

                values = 1
                for rank_access in bound_tensor_access.get_accesses().values():
                                                                      
                                              
                    access_shape = 1
                    for term in rank_access.get_expr():
                        if isinstance(term, VariableTerm) \
                                and term.get() in bound_part_ranks:
                            access_shape += term.get().get_shape() - 1
                    values *= access_shape

                bit_width = bound_tensor_access.get_tensor().get_bit_width()
                mem_latencies[memory] += values * bit_width / bandwidth

                                                                              
                                       
        if not mem_latencies:
            location = "attributes (for one or more of memories [" + ", ".join(
                memories) + "])"
            raise MissingFieldError("bandwidth", location)

        for memory, latency in mem_latencies.items():
            if not isinstance(latency, float):
                mem_latencies[memory] = latency.subs(shapes)

        total_latency = max(mem_latencies.values())
        port_latencies: Dict[str, float] = {}
        for bound_einsum in instruction.get_einsums():
            for bound_tensor in bound_einsum.get_reuse_spec().get_tensors():
                memory_component = bound_tensor.get_memory()
                ports = memory_component.get_ports(instruction.get_engine())
                if not ports:
                    continue

                                              
                assert len(ports) == 1
                port = ports[0]

                if port not in port_latencies:
                    port_latencies[port] = total_latency

        return port_latencies

    @staticmethod
    def get_num_instruction_calls(
            instruction: Instruction, shapes: Dict[str, int]) -> int:
        
                                                                         
                                                                             
        bound_einsum = instruction.get_einsums()[0]

        loop_order = bound_einsum.get_loop_order()
        bound_ranks = bound_einsum.get_bound_ranks()

        calls = 1
        for map_rank in loop_order[:len(loop_order) - len(bound_ranks)]:
            if not map_rank.get_is_spatial():
                calls *= map_rank.get_shape()

        if not isinstance(calls, int):
            calls = calls.subs(shapes)

        return calls
