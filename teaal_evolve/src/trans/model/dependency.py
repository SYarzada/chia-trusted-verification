
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common import *
from src.ir import *
from src.parse import *
from src.trans.model.instruction import InstructionModel

DependencyAwareTraces = Dict[str, List[Tuple[Tuple[str, ...],
                                                  Tuple[int, ...], float, float, float, str, str, Tuple[int, ...]]]]


class DependencyAwareModel:
    
    @staticmethod
    def display_traces(traces: DependencyAwareTraces) -> str:
        
        max_einsums = max(len(entry[0])
                          for trace in traces.values() for entry in trace)
        max_point = max(len(entry[1])
                        for trace in traces.values() for entry in trace)

        csv: List[Any] = [["Component"] +
                          ["Instruction Einsum " + str(i) for i in range(max_einsums)] +
                          ["Loop Index " + str(i) for i in range(max_point)] +
                          ["Start Time", "Engine End Time", "Data End Time", "Hazard Type", "Hazard Einsum"] +
                          ["Hazard Loop Index " + str(i) for i in range(max_point)]]

        for component, trace in traces.items():
            for entry in trace:
                line = [component] + \
                    [*entry[0]] + [None] * (max_einsums - len(entry[0])) + \
                    [*entry[1]] + [None] * (max_point - len(entry[1])) + \
                    [*entry[2:7]] + \
                    [*entry[7]] + [None] * (max_point - len(entry[7]))

                csv.append(line)

        return "\n".join(",".join(str(val) for val in line) for line in csv)

    @staticmethod
    def get_traces(program: ElaboratedProgram,
                   shapes: Dict[str, int]) -> DependencyAwareTraces:
        
        latencies: Dict[Tuple[str, ...], Dict[str, Tuple[float, float]]] = {}
        tensor_access_info: Dict[str, Tuple[List[str],
                                            List[Tuple[str, int, Basic]]]] = {}
        iter_spaces: Dict[str, List[str]] = {}
        tensor_shapes: Dict[str, Dict[str, Dict[str, int]]] = {}
        bound_loops: Dict[str, List[None]] = {}
        traces: DependencyAwareTraces = {}
        write_traces: Dict[str, List[Tuple[str, int]]] = {}
        for instruction in program.get_instructions():
            inst = tuple(instruction.get_einsum_names())

                                  
            latencies[inst] = {}
            if CostMetricEnum.compute in instruction.get_cost_metrics():
                latency_info = InstructionModel.get_per_instr_compute_latency(
                    instruction, shapes)
                for engine, engine_latencies in latency_info.items():
                    latencies[inst][engine] = engine_latencies

                                 
            if CostMetricEnum.memory in instruction.get_cost_metrics():
                memory_latencies = InstructionModel.get_per_instr_mem_latency(
                    instruction, shapes)
                for memory, mem_latency in memory_latencies.items():
                    latencies[inst][memory] = (mem_latency, mem_latency)

            for component in latencies[inst]:
                if component not in traces:
                    traces[component] = []

                                                              
            for bound_einsum in instruction.get_einsums():
                einsum = bound_einsum.get_name()
                bound_loops[einsum] = [None] * \
                    len(bound_einsum.get_bound_ranks())
                reuse_spec = bound_einsum.get_reuse_spec()

                iter_spaces[einsum] = []
                for map_rank in bound_einsum.get_loop_order():
                    iter_spaces[einsum].append(map_rank.get_name())

                for i, tensor_access in enumerate(
                        reuse_spec.get_tensor_accesses()):
                    tensor = tensor_access.get_tensor().get_name()

                    if tensor not in tensor_access_info:
                        roots = tensor_access.get_tensor().get_memory().get_ranks()
                        tensor_access_info[tensor] = (roots, [])
                        tensor_shapes[tensor] = {}

                                                    
                    tensor_shapes[tensor][einsum] = {}
                    ranks: Set[Rank] = reuse_spec.get_all_ranks()
                    decl_einsum = \
                        bound_einsum.get_map_einsum().get_part_einsum().get_decl_einsum()
                    ranks.update(decl_einsum.get_reuse_spec().get_all_ranks())
                    for rank in ranks:
                        shape = rank.get_shape()
                        if isinstance(shape, int):
                            tensor_shapes[tensor][einsum][rank.get_name()
                                                          ] = shape
                        else:
                            tensor_shapes[tensor][einsum][rank.get_name()] = shape.subs(
                                shapes)

                                                                         
                        if rank.get_name() not in iter_spaces[einsum]:
                            continue

                                                                
                        map_rank = bound_einsum.maybe_get_rank(
                            rank.get_name(), "dependency-aware model")
                        bound_rank = program.get_fusion().get_fused_ranks(map_rank)
                        tensor_shapes[tensor][einsum][rank.get_name(
                        )] += bound_rank.get_max_offset()

                                                                         
                    for mem_rank, part_ranks in tensor_access.get_tensor().get_rank_binding().items():

                        rank_shape = 1
                        for part_rank in part_ranks:
                            shape = part_rank.get_shape()
                            rank_shape *= shape

                        if not isinstance(rank_shape, int):
                            rank_shape = rank_shape.subs(shapes)

                        tensor_shapes[tensor][einsum][mem_rank] = rank_shape

                    loop_ranks = {
                        part_rank: rank.lower() for rank,
                        part_rank in reuse_spec.get_iter_ranks().items()}
                    get_rank_expression = program.get_rank_expression_fn(
                        loop_ranks)
                    tensor_access_info[tensor][1].append(
                        (einsum, i, tensor_access.get_linearized_accesses(get_rank_expression)))

                                                 
        tensors: Dict[str, Tuple[Dict[str, List[int]], IntervalSolver]] = {}
        for tensor, access_info in tensor_access_info.items():
            write_traces[tensor] = []

            uses: Dict[str, List[int]] = {}
            for einsum, i, _ in access_info[1]:
                if einsum not in uses:
                    uses[einsum] = []
                uses[einsum].append(i)

            used_iter_spaces = {
                einsum: iter_space for einsum,
                iter_space in iter_spaces.items() if einsum in uses.keys()}
            used_shapes = {
                einsum: einsum_shapes for einsum,
                einsum_shapes in tensor_shapes[tensor].items() if einsum in uses.keys()}
            tensors[tensor] = (
                uses,
                IntervalSolver(
                    access_info[1],
                    used_iter_spaces,
                    access_info[0],
                    used_shapes))

        einsum_shapes: Dict[str, Dict[str, int]] = {}
        for tensor, es_shapes in tensor_shapes.items():
            einsum_shapes = einsum_shapes | es_shapes

        _, traces, write_traces = DependencyAwareModel.__get_latency_helper(program, (), [], list(
            program.sort_nodes()), traces, write_traces, latencies, tensors, bound_loops, einsum_shapes)

        return traces

    @staticmethod
    def __get_latency_helper(
        program: ElaboratedProgram,
        point: Tuple[int, ...],
        call_stack: List[Node],
        remaining_nodes: List[Node],
        traces: DependencyAwareTraces,
        write_traces: Dict[str, List[Tuple[str, int]]],
        latencies: Dict[Tuple[str, ...], Dict[str, Tuple[float, float]]],
        tensors: Dict[str, Tuple[Dict[str, List[int]], IntervalSolver]],
        bound_loops: Dict[str, List[None]],
        shapes: Dict[str, Dict[str, int]]
    ) -> Tuple[List[Node], DependencyAwareTraces, Dict[str, List[Tuple[str, int]]]]:
        
        while remaining_nodes:
            node = remaining_nodes[0]

            if isinstance(node, BodyNode):
                execute_body = True
                for i, call_node in enumerate(call_stack[1:] + [node]):
                    info = program.get_offset(call_node)
                    if info is not None:
                        bound_rank, offset = info
                        upper_map_rank = bound_rank.get_ranks()[0]
                        bound_shape = shapes[upper_map_rank.get_einsum(
                        )][upper_map_rank.get_name()] - bound_rank.get_max_offset()

                        if point[i] < offset or point[i] >= bound_shape + offset:
                            execute_body = False
                            break

                if execute_body:
                    inst = tuple(node.get_instruction().get_einsum_names())

                                                                           
                                
                    best_start = 0.0
                    engine_latency = 0.0
                    data_latency = 0.0
                    hazard_type = "structural"
                    hazard_einsum = ""
                    hazard_point: Tuple[int, ...] = ()
                    for component in latencies[inst].keys():
                        if traces[component] and traces[component][-1][3] >= best_start:
                            hazard_einsum = traces[component][-1][0][-1]
                            hazard_point = traces[component][-1][1]
                            best_start = traces[component][-1][3]

                        engine_latency = max(
                            engine_latency, latencies[inst][component][0])
                        data_latency = max(
                            data_latency, latencies[inst][component][1])

                                     
                    read_fingers = {
                        component: len(trace) - 1
                        for component, trace in traces.items()}

                                             
                    write_fingers = {}
                    for bound_einsum in node.get_instruction().get_einsums():
                        for bound_tensor in bound_einsum.get_reuse_spec().get_tensors():
                            tensor = bound_tensor.get_name()
                            write_fingers[tensor] = len(
                                write_traces[tensor]) - 1

                    while any(i >= 0 for i in read_fingers.values()):
                                                         
                        read_end = 0.0
                        for component, i in read_fingers.items():
                            if i < 0:
                                continue

                            if traces[component][i][3] > read_end:
                                latest_read = component
                                read_end = traces[component][i][3]

                        write_end = 0.0
                        for tensor, i in write_fingers.items():
                            if i < 0:
                                continue

                            component, j = write_traces[tensor][i]
                            if traces[component][j][4] > write_end:
                                latest_write = tensor
                                write_end = traces[component][j][4]

                        if read_end <= best_start and write_end <= best_start:
                            break

                        check_prev_read = False
                        check_prev_write = False
                        if read_end >= write_end:
                            entry = traces[latest_read][read_fingers[latest_read]]
                            prev_end = entry[3]

                            read_fingers[latest_read] -= 1
                            check_prev_read = True

                        else:
                            component, j = write_traces[latest_write][write_fingers[latest_write]]
                            entry = traces[component][j]
                            prev_end = entry[4]

                            write_fingers[latest_write] -= 1
                            check_prev_write = True

                                                                       
                        for bound_einsum in node.instruction.get_einsums():
                            curr_einsum = bound_einsum.get_name()

                            for i, bound_tensor in enumerate(
                                    bound_einsum.get_reuse_spec().get_tensors()):
                                curr_tensor = bound_tensor.get_name()

                                for prev_einsum in entry[0]:
                                    if prev_einsum not in \
                                            tensors[curr_tensor][0].keys():
                                        continue

                                    curr_point: Tuple[Optional[int], ...] = point + \
                                        tuple(bound_loops[curr_einsum])
                                    prev_point: Tuple[Optional[int], ...] = entry[1] + \
                                        tuple(bound_loops[prev_einsum])

                                    for j in tensors[curr_tensor][0][prev_einsum]:
                                                                   
                                        if j > 0 and i > 0:
                                            break

                                        if tensors[curr_tensor][1].intersect(
                                                curr_einsum, i, curr_point, prev_einsum, j, prev_point):
                                            best_start = prev_end
                                            hazard_type = "data"
                                            hazard_einsum = prev_einsum
                                            hazard_point = entry[1]
                                            break

                                    if best_start == prev_end:
                                        break

                                if best_start == prev_end:
                                    break

                            if best_start == prev_end:
                                break

                        if best_start == prev_end:
                            break

                    engine_end = best_start + engine_latency
                    data_end = best_start + data_latency

                    for component in latencies[inst].keys():
                        traces[component].append(
                            (inst,
                             point,
                             best_start,
                             engine_end,
                             data_end,
                             hazard_type,
                             hazard_einsum,
                             hazard_point))

                    for bound_einsum in node.get_instruction().get_einsums():
                        output = bound_einsum.get_reuse_spec().get_output_tensor().get_name()
                        i = len(write_traces[output]) - 1
                        if i >= 0:
                            component, j = write_traces[output][i]
                            while traces[component][j][4] > data_end and i > 0:
                                i -= 1
                                component, j = write_traces[output][i]

                                                       
                        if latencies[inst]:
                            component = next(iter(latencies[inst].keys()))
                            if i < 0:
                                write_traces[output].insert(
                                    0, (component, len(traces[component]) - 1))
                            else:
                                                                                  
                                write_traces[output].insert(
                                    i + 2, (component, len(traces[component]) - 1))

            elif isinstance(node, EndLoopNode):
                return remaining_nodes, traces, write_traces

            elif isinstance(node, LoopNode):
                map_rank = node.get_rank().get_ranks()[0]
                shape = shapes[map_rank.get_einsum()][map_rank.get_name()]

                new_nodes = remaining_nodes
                for i in range(shape):
                    loop_result, traces, write_traces = DependencyAwareModel.__get_latency_helper(
                        program, point + (i,), call_stack + [node], remaining_nodes[1:], traces, write_traces, latencies, tensors, bound_loops, shapes)
                    if len(new_nodes) > len(loop_result):
                        new_nodes = loop_result

                remaining_nodes = new_nodes

            elif isinstance(node, StartNode):
                pass

            else:
                assert False                    

            remaining_nodes = remaining_nodes[1:]

        return [], traces, write_traces
