

import networkx as nx                
from sympy import Basic, Symbol                
from typing import Callable, Dict, List, Optional

from src.common import *
from src.ir.arch import *
from src.ir.binding.common import *
from src.ir.binding.concrete import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.ir.partitioning import *
from src.ir.program.loop_branch import LoopBranch
from src.ir.program.nest_node import *
from src.parse import *


class ElaboratedProgram:
    

    def __init__(
            self,
            einsum_parser: EinsumParser,
            mapping_parser: MappingParser,
            format_parser: FormatParser,
            binding_parser: BindingParser,
            arch_parser: ArchitectureParser,
            options_parser: OptionsParser):

                                     
        self.elaborated_einsum = ElaboratedEinsum(
            einsum_parser, options_parser)
        self.elaborated_einsum.check_concrete()

        self.elaborated_mapping = ElaboratedMapping(
            self.elaborated_einsum, mapping_parser)
        self.elaborated_mapping.check_concrete()

        self.elaborated_format = ElaboratedFormat(
            self.elaborated_einsum, self.elaborated_mapping, format_parser)
        self.elaborated_format.check_concrete()

        self.elaborated_arch = ElaboratedArchitecture(arch_parser)

        self.elaborated_binding = ElaboratedBinding(
            self.elaborated_einsum,
            self.elaborated_mapping,
            self.elaborated_format,
            binding_parser,
            self.elaborated_arch)

        self.__build_graph(options_parser)

    def __build_graph(self, options_parser: OptionsParser):
        
                                          
        self.graph = nx.DiGraph()
        self.loop_branches = {}
        fusion = self.elaborated_binding.get_fusion()

        for instruction in self.elaborated_binding.get_instructions():
            body_node = BodyNode(instruction)
            for bound_einsum in instruction.get_einsums():
                loop_branch = LoopBranch(self.elaborated_binding, bound_einsum)
                for edge in loop_branch.get_graph().edges():
                    new_edge: List[Node] = []
                    for node in edge:
                        if isinstance(node, BranchBodyNode):
                            new_edge.append(body_node)
                        elif isinstance(node, BranchEndLoopNode):
                            new_edge.append(
                                EndLoopNode(
                                    fusion.get_fused_ranks(
                                        node.get_rank())))
                        elif isinstance(node, BranchLoopNode):
                            new_edge.append(
                                LoopNode(
                                    fusion.get_fused_ranks(
                                        node.get_rank())))
                        else:
                            new_edge.append(node)
                    self.graph.add_edge(*new_edge)

                self.loop_branches[bound_einsum.get_name()] = loop_branch

                             
        fields = options_parser.get_fields()

                                                      
        if "kernel-name" not in fields.keys():
            raise MissingFieldError("kernel-name", "options")

        self.kernel_name = fields["kernel-name"]

    def get_cascade_inputs(self) -> List[DeclaredTensor]:
        
        return self.elaborated_einsum.get_cascade_inputs()

    def get_cascade_outputs(self) -> List[DeclaredTensor]:
        
        outputs = self.elaborated_einsum.get_cascade_outputs()

                                                                               
                                                                
        assert outputs is not None
        return outputs

    def get_declared_tensor(self, node: Node) -> List[BoundTensor]:
        
        node_info: List[Tuple[str, Node]]
        if isinstance(node, BodyNode):
            node_info = [(bound_einsum.get_name(), BranchBodyNode(bound_einsum))
                         for bound_einsum in node.get_instruction().get_einsums()]
        elif isinstance(node, LoopNode):
            node_info = []
            for map_rank in node.get_rank().get_ranks():
                node_info.append(
                    (map_rank.get_einsum(), BranchLoopNode(map_rank)))

        assert isinstance(node, BodyNode) or isinstance(node, LoopNode)

        declared_tensors = []
        for einsum, enode in node_info:
            opt_tensor = self.loop_branches[einsum].get_declared_tensor(enode)

            if opt_tensor:
                declared_tensors.append(opt_tensor)

        return declared_tensors

    def get_einsum(self, einsum: str) -> BoundEinsum:
        
        return self.elaborated_binding.get_einsum(einsum)

    def get_first_access(self, tensor: str) -> BoundTensor:
        
        return self.elaborated_binding.get_first_access(tensor)

    def get_fusion(self) -> BoundFusion:
        
        return self.elaborated_binding.get_fusion()

    def get_graph(self):
        
        return self.graph

    def get_offset(self, node: Node) -> Optional[Tuple[BoundRank, int]]:
        
        preds = list(self.graph.predecessors(node))

        if len(preds) != 1:
            assert isinstance(node, LoopNode)
            ranks = [map_rank.get_einsum() + "." + map_rank.get_name()
                     for map_rank in node.get_rank().get_ranks()]
            raise FusionRankMismatchError(
                ranks, "fusion", "not all ranks have the same parent")

        parent = preds[0]

                                                                            
        if isinstance(parent, StartNode):
            return None

        assert isinstance(parent, LoopNode)

        if parent.get_rank().get_max_offset() == 0:
            return None

        stmt = self.__node_to_stmt(node)
        return parent.get_rank(), parent.get_rank().get_offset(stmt)

    def get_instructions(self) -> List[Instruction]:
        
        return self.elaborated_binding.get_instructions()

    def get_kernel_name(self):
        
        return self.kernel_name

    def get_rank_expression_fn(
        self,
        loop_rank_vars: Dict[PartitionedRank, str]
    ) -> Callable[[PartitionedRank], Optional[Tuple[Basic, Dict[Symbol, PartitionedRank]]]]:
        
        def to_sympy(part_rank: PartitionedRank):
            einsum = part_rank.get_einsum()
            map_einsum = self.elaborated_mapping.maybe_get_einsum(
                einsum, einsum + " code generation")
            partitioning = map_einsum.get_part_einsum().get_partitioning()

            expr, symbols_dict = partitioning.get_rank_expression(part_rank)

                               
            assert len(symbols_dict) == 1
            loop_rank = list(symbols_dict.values())[0]

                                             
            map_rank = map_einsum.maybe_get_rank(
                loop_rank.get_name(), einsum + " code generation")
            bound_rank = self.elaborated_binding.get_fusion().get_fused_ranks(map_rank)
            loop_node = LoopNode(bound_rank)

                                                                         
                                                            
            offset = 0
            if loop_node in self.graph.nodes:
                succ: Node
                for node in self.graph.successors(loop_node):
                    if isinstance(node, BodyNode):
                        if einsum in node.get_instruction().get_einsum_names():
                            succ = node
                            break

                    else:
                        assert isinstance(node, LoopNode)

                        node_einsums = \
                            {node_map_rank.get_einsum()
                             for node_map_rank in node.get_rank().get_ranks()}

                        if map_rank.get_einsum() in node_einsums:
                            succ = node
                            break

                stmt = self.__node_to_stmt(succ)
                offset = bound_rank.get_offset(stmt)

            final_dict = {}
            for old_symbol, rank in symbols_dict.items():
                if rank in loop_rank_vars:
                    new_symbol = Symbol(loop_rank_vars[rank])

                    expr = expr.subs(old_symbol, new_symbol - offset)
                    final_dict[new_symbol] = rank

            if final_dict:
                return expr, final_dict
            return None

        return to_sympy

    def get_symbols(self) -> List[Symbol]:
        
        return self.elaborated_mapping.get_symbols()

    def sort_nodes(self) -> List[Node]:
        
        def sort_func(node):
            if isinstance(node, StartNode):
                return "a"

            get_pos = self.elaborated_binding.get_fusion().get_position

            if isinstance(node, BodyNode):
                pos = min(get_pos(einsum)
                          for einsum in node.get_instruction().get_einsum_names())

            elif isinstance(node, EndLoopNode):
                pos = min(get_pos(rank.get_einsum())
                          for rank in node.get_rank().get_ranks())

            elif isinstance(node, LoopNode):
                pos = min(get_pos(rank.get_einsum())
                          for rank in node.get_rank().get_ranks())

            else:
                raise NotImplementedError                    

            return chr(ord("b") + pos)

        return nx.lexicographical_topological_sort(self.graph, key=sort_func)

    def __node_to_stmt(self, node: Node) -> Statement:
        
        if isinstance(node, LoopNode):
            return LoopStatement(node.get_rank().get_name())
        elif isinstance(node, BodyNode):
            einsums = tuple(bound_einsum.get_name()
                            for bound_einsum in node.get_instruction().get_einsums())
            return InstructionStatement(einsums)

                                                   
        assert False                    
