

import networkx as nx                

from typing import Optional

from src.ir.binding.concrete import *
from src.ir.binding.common import *
from src.ir.program.nest_node import *


class LoopBranch:
    

    def __init__(
            self,
            elaborated_binding: ElaboratedBinding,
            bound_einsum: BoundEinsum):
        self.bound_einsum = bound_einsum

        self.graph = nx.DiGraph()

        hw_ranks = bound_einsum.get_bound_ranks()

                                              
        curr_node: Node = StartNode()
        end_node_stack = []
        for rank in bound_einsum.get_loop_order():
                                                                            
                                         
            if rank in hw_ranks[:1]:
                break

            loop_node = BranchLoopNode(rank)

            self.graph.add_edge(curr_node, loop_node)

            end_node_stack.append(BranchEndLoopNode(rank))

            curr_node = loop_node

        body_node = BranchBodyNode(self.bound_einsum)
        self.graph.add_edge(curr_node, body_node)
        curr_node = body_node

                                                    
        for node in reversed(end_node_stack):
            self.graph.add_edge(curr_node, node)
            curr_node = node

                            
        output = bound_einsum.get_reuse_spec().get_output_tensor()
        if not output.get_constructor():
            self.output_decl_node = None
        else:
            unbound_ranks = []
                                                        
            accesses = bound_einsum.get_reuse_spec().get_tensor_accesses()[
                0].get_accesses()
            for part_rank in output.get_rank_binding()["unbound"]:
                terms = accesses[part_rank].get_expr()
                var_terms = [
                    term.get() for term in terms if isinstance(
                        term, VariableTerm)]

                if not var_terms:
                    continue

                assert len(var_terms) == 1
                unbound_ranks.append(var_terms[0])

            nodes = list(self.graph.nodes())
            if not unbound_ranks:
                                                        
                self.output_decl_node = nodes[1]

            else:
                                                                    
                partitioning = bound_einsum.get_map_einsum().get_part_einsum().get_partitioning()
                part_rank = partitioning.get_iter_space_rank(unbound_ranks[-1])
                map_rank = bound_einsum.maybe_get_rank(
                    part_rank.get_name(), "loop branch")

                last_unbound_node = BranchLoopNode(map_rank)

                                                                  
                if last_unbound_node in nodes:
                    last_unbound_loop = nodes.index(last_unbound_node)

                    fusion = elaborated_binding.get_fusion()
                    bound_rank = fusion.get_fused_ranks(map_rank)

                                                                              
                          
                    if bound_rank.get_max_offset() == 0:
                        self.output_decl_node = nodes[last_unbound_loop + 1]

                                                                             
                    else:
                        self.output_decl_node = last_unbound_node

                                                                 
                else:
                    self.output_decl_node = None

    def get_graph(self) -> nx.DiGraph:
        
        return self.graph

    def get_declared_tensor(self, node: Node) -> Optional[BoundTensor]:
        
        if node == self.output_decl_node:
            return self.bound_einsum.get_reuse_spec().get_output_tensor()
        return None

    def __eq__(self, other) -> bool:
        
        if not isinstance(other, LoopBranch):
            return False

        return nx.is_isomorphic(self.get_graph(), other.get_graph())
