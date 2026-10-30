
from lark import Token
import networkx as nx                
from sympy import Basic, Symbol                
from typing import Any, Dict, List, Set, Tuple

from src.common import *
from src.ir.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.partitioning.node import DirectiveNode, RankNode
from src.ir.partitioning.rank import PartitionedRank
from src.parse import *


class Partitioning(Base):
    

    def __init__(self, decl_einsum: DeclaredEinsum,
                 mapping_parser: MappingParser):
        self.decl_einsum = decl_einsum
        self.conds: List[Condition] = []

        einsum = self.decl_einsum.get_name()

        self.part_ranks: Dict[str, PartitionedRank] = {}
        self.graph = nx.DiGraph()
        self.roots: List[str] = []
        self.ordered_ranks = []
        unpartitionable = set()
        for tensor_access in self.decl_einsum.get_reuse_spec().get_tensor_accesses():
            for coord_rank, rank_access in tensor_access.get_accesses().items():
                rank_var_expr = rank_access.get_expr()

                                                                       
                if len(rank_var_expr) > 1 or isinstance(
                        rank_var_expr[0], ConstantTerm):
                    unpartitionable.add(coord_rank.get_name())
                    for term in rank_var_expr:
                        if isinstance(term, VariableTerm):
                            unpartitionable.add(term.get().get_name())

                expr_ranks = [
                    term.get() for term in rank_var_expr if isinstance(
                        term, VariableTerm)]
                for decl_rank in [coord_rank] + expr_ranks:
                    if decl_rank.get_name() not in self.roots:
                        self.roots.append(decl_rank.get_name())

                    rank = decl_rank.get_name()
                    if rank in self.part_ranks:
                        continue

                    self.ordered_ranks.append(rank)
                    self.part_ranks[rank] = PartitionedRank(
                        einsum, rank, (decl_rank.get_shape(),), (decl_rank.get_shape(),), (decl_rank,))
                    self.graph.add_node(RankNode(rank))

        parsed_parts = mapping_parser.get_partitioning()
        self.symbols: List[Symbol] = []
        self.part_runs = {}
        self.partitioned: Set[str] = set()
        if einsum not in parsed_parts:
            return

                                      
        location = einsum + " partitioning"
        for roots, trees in parsed_parts[einsum].items():
                                               
            if not trees:
                continue

                                                                        
                           
            if len(roots) == 1:
                root = roots[0]

                                                                               
                if root in unpartitionable:
                    raise IllegalPartitioningError(roots, location)

                src_node = RankNode(root)
                part_run = []
                for i, tree in enumerate(trees):
                    if tree.data != "uniform_shape":
                        raise IllegalPartitioningDirectiveError(
                            tree.data, roots, location)

                    upper_name = root + str(len(trees) - i)
                    lower_name = root + str(len(trees) - i - 1)
                    if i < len(trees) - 1:
                        lower_name += "I"

                    dir_node = DirectiveNode(
                        tree, (src_node.get_rank(),), (upper_name, lower_name))

                    self.graph.add_edge(src_node, dir_node)
                    self.graph.add_edge(dir_node, RankNode(upper_name))
                    src_node = RankNode(lower_name)
                    self.graph.add_edge(dir_node, src_node)

                    part_run.append(upper_name)
                    self.ordered_ranks.append(upper_name)
                    self.ordered_ranks.append(lower_name)

                if len(trees) > 0:
                    part_run.append(src_node.get_rank())
                    self.part_runs[root] = part_run

                                                        
            else:
                if len(trees) > 1:
                    raise IllegalPartitioningError(roots, location)

                tree = trees[0]
                if tree.data != "flatten":
                    raise IllegalPartitioningDirectiveError(
                        tree.data, roots, location)

                dir_node = DirectiveNode(tree, roots, ("".join(roots),))
                for root in roots:
                    self.graph.add_edge(RankNode(root), dir_node)

                self.graph.add_edge(dir_node, RankNode("".join(roots)))
                self.ordered_ranks.append("".join(roots))

                                
        for node in nx.topological_sort(self.graph):
            if isinstance(node, RankNode):
                if node.get_rank() not in self.part_ranks.keys():
                    raise UnknownRankError(node.get_rank(), location)
                continue

            assert isinstance(node, DirectiveNode)

            for rank in node.get_src():
                if rank in self.partitioned:
                    raise MultiplePartitioningError(rank, location)
                self.partitioned.add(rank)

            directive = node.get_directive()
            if directive.data == "uniform_shape":
                                                                              
                            
                src_name = node.get_src()[0]

                                                                        
                                  
                shape_token = node.get_directive().children[0].children[0]
                assert isinstance(shape_token, Token)
                if shape_token.type == "NUMBER":
                    lower_shape = int(shape_token)
                else:
                    lower_shape = Symbol(str(shape_token))

                if isinstance(lower_shape,
                              Symbol) and lower_shape not in self.symbols:
                    self.symbols.append(lower_shape)

                                                                     
                                                                         
                                                                     
                src_shapes = self.part_ranks[src_name].get_abs_shapes()
                if len(src_shapes) > 1:
                    raise IllegalPartitioningError((src_name,), location)

                src_shape = self.part_ranks[src_name].get_abs_shapes()[0]
                root_ranks = self.part_ranks[src_name].get_roots()
                root = root_ranks[0].get_name()
                upper_name, lower_name = node.get_dst()

                                                          
                mod = src_shape % lower_shape
                if isinstance(mod, int) and mod != 0:
                    raise ImperfectFactorizationError(
                        src_shape, lower_shape, einsum, root)
                else:
                    self.conds.append(EqualCondition(mod, 0))

                upper_shape = src_shape // lower_shape

                lower_rank = PartitionedRank(
                    einsum, lower_name, (lower_shape,), (lower_shape,), root_ranks)
                upper_rank = PartitionedRank(
                    einsum, upper_name, (src_shape,), (upper_shape,), root_ranks)

                self.part_ranks[lower_name] = lower_rank
                self.part_ranks[upper_name] = upper_rank

            elif directive.data == "flatten":
                src_ranks = [self.part_ranks[rank] for rank in node.get_src()]

                root_ranks = tuple(
                    decl_rank for src_rank in src_ranks for decl_rank in src_rank.get_roots())
                abs_shapes = tuple(
                    shape for src_rank in src_ranks for shape in src_rank.get_abs_shapes())
                rel_shapes = tuple(
                    shape for src_rank in src_ranks for shape in src_rank.get_rel_shapes())

                flat_name = node.get_dst()[0]
                flat_rank = PartitionedRank(
                    einsum, flat_name, abs_shapes, rel_shapes, root_ranks)

                self.part_ranks[flat_name] = flat_rank

            else:
                                                                           
                            
                raise NotImplementedError                    

    def apply_to_accesses(self,
                          decl_tensor_access: DeclaredTensorAccess) -> Dict[PartitionedRank,
                                                                            RankAccess[PartitionedRank]]:
        
        einsum = self.decl_einsum.get_name()
        tensor = decl_tensor_access.get_tensor().get_name()
        part_accesses = {}

        tgraph = nx.DiGraph()
        agraph = nx.DiGraph()
        labels = {}
        for i, (decl_trank, decl_rank_access) in enumerate(
                decl_tensor_access.get_accesses().items()):
                                                               
                                                                         
            terms = decl_rank_access.get_expr()
            if len(terms) > 1 or not isinstance(terms[0], VariableTerm):
                part_access: List[Term] = []
                for term in decl_rank_access.get_expr():
                    if isinstance(term, ConstantTerm):
                        part_access.append(term)
                        continue

                    assert isinstance(term, VariableTerm)
                    part_access.append(VariableTerm(
                        self.part_ranks[term.get().get_name()]))

                part_accesses[self.part_ranks[decl_trank.get_name()]] = RankAccess(
                    part_access)
                continue

                                                 
            tgraph.add_node(decl_trank.get_name(), label=chr(ord("A") + i))
            agraph.add_node(terms[0].get().get_name(), label=chr(ord("A") + i))

            labels[decl_trank.get_name()] = chr(ord("A") + i)
            labels[terms[0].get().get_name()] = chr(ord("A") + i)

                                                                            
                      
        for graph in [tgraph, agraph]:
            active = set(graph.nodes())
            for node in nx.topological_sort(self.graph):
                if not isinstance(node, DirectiveNode):
                    continue

                                                        
                if all(rank in active for rank in node.get_src()):
                    active.difference_update(node.get_src())
                    active.update(node.get_dst())

                    part_node = "part" + "".join(node.get_src())
                    part_label = "".join(labels[src] for src in node.get_src())
                    graph.add_node(part_node, label=part_label)

                    for src in node.get_src():
                        graph.add_edge(src, part_node)

                    for i, dst in enumerate(node.get_dst()):
                        labels[dst] = part_label + chr(ord("a") + i)
                        graph.add_node(dst, label=labels[dst])
                        graph.add_edge(part_node, dst)
        if tgraph.number_of_nodes() == 0 and agraph.number_of_nodes() == 0:
            return part_accesses

        access_map = nx.vf2pp_isomorphism(tgraph, agraph, node_label="label")
        if access_map is None:
            raise PartitioningMismatchError(einsum, tensor)

        for trank in self.ordered_ranks:
                                   
            if not tgraph.has_node(trank):
                continue

                                                
            if list(tgraph.successors(trank)):
                continue

            arank = access_map[trank]

            part_trank = self.part_ranks[trank]
            part_arank = self.part_ranks[arank]

                                                      
            if len(part_trank.get_roots()) != 1 or part_trank.get_roots()[
                    0].get_shape() != part_trank.get_shape():
                between = "ranks " + trank + " and " + arank
                location = einsum + " partitioning"
                opt_cond = ConditionUtils.shape_equal_or_cond(
                    part_trank.get_shape(), part_arank.get_shape(), between, location)

                if opt_cond and opt_cond not in self.conds:
                    self.conds.append(opt_cond)

            part_accesses[part_trank] = RankAccess(
                [VariableTerm(part_arank)])

        return part_accesses

    def apply_to_ranks(
            self,
            decl_ranks: List[DeclaredRank]) -> List[PartitionedRank]:
        
        active = set(decl_rank.get_name() for decl_rank in decl_ranks)
        for node in nx.topological_sort(self.graph):
            if not isinstance(node, DirectiveNode):
                continue

                                                    
            if all(rank in active for rank in node.get_src()):
                active.difference_update(node.get_src())
                active.update(node.get_dst())

                               

        part_ranks = [self.part_ranks[rank] for rank in active]
        part_ranks.sort(key=lambda r: r.get_name())

        return part_ranks

    def get_conds(self) -> List[Condition]:
        
        return self.conds

    def get_einsum(self) -> DeclaredEinsum:
        
        return self.decl_einsum

    def get_iter_space_rank(
            self,
            part_rank: PartitionedRank) -> PartitionedRank:
        
        curr_node = RankNode(part_rank.get_name())
        succs = list(self.graph.successors(curr_node))
        while succs:
            assert len(succs) == 1
            curr_node = succs[0]
            succs = list(self.graph.successors(curr_node))

        return self.part_ranks[curr_node.get_rank()]

    def get_rank_expression(
            self, part_rank: PartitionedRank) -> Tuple[Basic, Dict[Symbol, PartitionedRank]]:
        
        curr_node = RankNode(part_rank.get_name())
        children = list(self.graph.successors(curr_node))

        if not children:
            rank_symbol = Symbol(part_rank.get_name().lower())
            return rank_symbol, {rank_symbol: part_rank}

        flattened_ranks = [part_rank.get_name()]
        while children:
            assert len(children) == 1
            prev_node = curr_node
            curr_node = children[0]

            if isinstance(curr_node, DirectiveNode):
                assert curr_node.get_directive().data == "flatten"
                assert isinstance(prev_node, RankNode)

                srcs = list(curr_node.get_src())

                i = srcs.index(prev_node.get_rank())

                flattened_ranks = srcs[:i] + flattened_ranks + srcs[i + 1:]

            children = list(self.graph.successors(curr_node))

        assert isinstance(curr_node, RankNode)

        shapes = [self.part_ranks[rank].get_shape()
                  for rank in flattened_ranks]

        i = flattened_ranks.index(part_rank.get_name())

        rank_symbol = Symbol(str(curr_node.get_rank()).lower())
        expr = rank_symbol

                                     
        div = 1
        for shape in shapes[i + 1:]:
            div *= shape

        if div != 1:
            expr = expr // div

                                  
        if i > 0:
            expr = expr % shapes[i]

        return expr, {rank_symbol: self.part_ranks[curr_node.get_rank()]}

    def get_solver_equalities(
            self, other: "Partitioning") -> List[Tuple[List[str], List[str]]]:
        
        equalities = []
        for partitioning in [self, other]:
            prefix = partitioning.get_einsum().get_name() + "."
            for node in partitioning.graph.nodes:
                if isinstance(node, DirectiveNode):
                    equalities.append(([prefix +
                                        src for src in node.get_src()], [prefix +
                                                                         dst for dst in node.get_dst()]))

        for rank in self.roots:
            if rank not in other.roots:
                continue

            sprefix = self.get_einsum().get_name() + "."
            oprefix = other.get_einsum().get_name() + "."

                                                                        
            equalities.append(([sprefix + rank], [oprefix + rank]))

        for rank in self.part_runs.keys():
            if rank not in other.part_runs:
                continue

            srank = self.part_ranks[rank]
            orank = other.part_ranks[rank]

                                                                       
            assert len(srank.get_abs_shapes()) == 1
            assert len(orank.get_abs_shapes()) == 1

            srun = [self.part_ranks[run_rank]
                    for run_rank in self.part_runs[rank]]
            orun = [other.part_ranks[run_rank]
                    for run_rank in other.part_runs[rank]]
            new_equalities = Partitioning.__equality_helper(
                srun, orun, 0, 0, 1, 1)

            for equality in new_equalities:
                equalities.append(
                    (PartitionedRank.scoped_ranks(equality[0]),
                     PartitionedRank.scoped_ranks(equality[1])))

                                                                           
        for partitioning in [self, other]:
            prefix = partitioning.get_einsum().get_name() + "."
            for rank, part_rank in partitioning.part_ranks.items():
                if part_rank.get_shape() != 1:
                    continue

                for str_equality in equalities:
                    for term in str_equality:
                        if prefix + rank in term:
                            term.remove(prefix + rank)

                                         
        while ([], []) in equalities:
            equalities.remove(([], []))

        return equalities

    def get_symbols(self) -> List[Symbol]:
        
        return self.symbols

    @staticmethod
    def __equality_helper(sranks: List[PartitionedRank],
                          oranks: List[PartitionedRank],
                          sstart: int,
                          ostart: int,
                          s: int,
                          o: int) -> List[Tuple[List[PartitionedRank],
                                                List[PartitionedRank]]]:
        
                                             
        if s >= len(sranks) or o >= len(oranks):
            if sstart > 0 or ostart > 0:
                return [(sranks[sstart:], oranks[ostart:])]
            else:
                return []

                                                                   
        assert len(sranks[s].get_abs_shapes()) == 1
        assert len(oranks[o].get_abs_shapes()) == 1

        sshape = sranks[s].get_abs_shapes()[0]
        oshape = oranks[o].get_abs_shapes()[0]

        if sshape == oshape:
            equalities = [(sranks[sstart:s], oranks[ostart:o])]

            return equalities + \
                Partitioning.__equality_helper(sranks, oranks, s, o, s + 1, o + 1)

        elif isinstance(sshape, int) and isinstance(oshape, int):
            if sshape < oshape:
                return Partitioning.__equality_helper(
                    sranks, oranks, sstart, ostart, s, o + 1)

                             
            else:
                return Partitioning.__equality_helper(
                    sranks, oranks, sstart, ostart, s + 1, o)

        else:
            eq0 = Partitioning.__equality_helper(
                sranks, oranks, sstart, ostart, s + 1, o)
            eq1 = Partitioning.__equality_helper(
                sranks, oranks, sstart, ostart, s, o + 1)

            for eq in eq1:
                if eq not in eq0:
                    eq0.append(eq)

            return eq0

    def key(self) -> Tuple[Any, ...]:
        return self.decl_einsum, self.part_ranks
