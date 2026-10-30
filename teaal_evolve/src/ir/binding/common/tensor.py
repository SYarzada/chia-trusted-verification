

from typing import Any, Dict, List, Set, Tuple

from src.common import *
from src.ir.common import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.gen import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.partitioning import *
from src.ir.solver import *
from src.parse import *


class BoundTensor(Tensor):
    

    def __init__(
            self,
            elaborated_einsum: ElaboratedEinsum,
            map_einsum: MappedEinsum,
            format_tensor: FormattedTensor,
            binding_parser: BindingParser,
            memories: Dict[str, MemoryComponent]):

        self.format_tensor = format_tensor
        self.is_output = self.format_tensor.get_part_tensor(
        ) == map_einsum.get_reuse_spec().get_output_tensor()

        einsum = self.format_tensor.get_einsum()
        tensor = self.format_tensor.get_name()
        location = tensor + \
            " tensor binding (when computing the " + einsum + " Einsum)"

        tensor_bindings = binding_parser.get_tensor_bindings()
                                                
        if einsum not in tensor_bindings.keys():
            raise MissingEinsumError(einsum, tensor + " tensor binding")

                                                
        if tensor not in tensor_bindings[einsum].keys():
            raise MissingTensorError(tensor, location[len(tensor) + 1:])

        tensor_binding = binding_parser.get_tensor_bindings()[einsum][tensor]

        format_ranks = self.format_tensor.get_ranks()
        rank_names = [rank.get_name() for rank in format_ranks]
        ranks_dict = {
            name: part_rank for name,
            part_rank in zip(
                rank_names,
                format_ranks)}

        bound_memories = []
        if tensor_binding.component is not None:
            bound_memories.append(tensor_binding.component)

        if tensor_binding.components is not None:
            bound_memories.extend(tensor_binding.components)

                                                    
        if not bound_memories:
            raise MissingComponentError(None, location)

                                                         
        self.memories = []
        for memory in bound_memories:
            if memory.name not in memories.keys():
                raise UnknownComponentError(memory.name, location)

            self.memories.append((memories[memory.name], memory.port))

                                                                        
        for memory_component, _ in self.memories[1:]:
            if memory_component.get_ranks() != self.memories[0][0].get_ranks():
                mem_names = []
                for memory in bound_memories:
                    if memory.name not in mem_names:
                        mem_names.append(memory.name)
                raise MemoryRankMismatchError(
                    mem_names, self.get_name() + " tensor binding")

                                                   
        mem_ranks = set(self.memories[0][0].get_ranks()).union({"unbound"})
        ErrorUtils.check_mem_ranks(mem_ranks, list(
            tensor_binding.ranks.keys()), location)

        rank_order = [rank for ranks in tensor_binding.ranks.values()
                      for rank in ranks]
        ErrorUtils.check_ranks(set(rank_names), rank_order, location)

        location0 = tensor + " format (when computing Einsum " + einsum + ")"
        self.ranks = {}
        for mem_rank, ranks in tensor_binding.ranks.items():
            location1 = mem_rank + " rank for the " + tensor + \
                " tensor binding (when computing Einsum " + einsum + ")"

            if mem_rank == "unbound":
                                                                
                if ranks != rank_names[:len(ranks)]:
                    raise RankOrderMismatchError(location0, location1)

                loop_order = [map_rank.get_rank()
                              for map_rank in map_einsum.get_loop_order()]

                tensor_accesses = map_einsum.get_reuse_spec(
                ).maybe_get_tensor_accesses(tensor, location)
                for tensor_access in tensor_accesses:
                    accesses = tensor_access.get_accesses()

                    for i, rank in enumerate(ranks):
                        terms = accesses[ranks_dict[rank]].get_expr()
                        var_terms = [
                            term.get() for term in terms if isinstance(
                                term, VariableTerm)]

                        if not var_terms:
                            continue

                                                                                  
                                           
                        if len(var_terms) > 1:
                            raise IllegalUnboundRanksError([rank], location)

                                                                                  
                                               

            else:
                                                                 
                mem_order = [rank for rank in rank_names if rank in ranks]
                if mem_order != ranks:
                    raise RankOrderMismatchError(location0, location1)

            self.ranks[mem_rank] = [ranks_dict[rank] for rank in ranks]

        for memory_component, _ in self.memories:
                                                         
            bitwidth = self.format_tensor.get_bit_width()
            mem_bws = memory_component.get_bit_widths()
            if bitwidth and bitwidth not in mem_bws:
                raise BitwidthMismatchError(
                    tensor,
                    bitwidth,
                    memory_component.get_name(),
                    mem_bws)

                                                         
            bitwidths = self.format_tensor.get_bit_widths()
            if bitwidths:
                mismatched_bw = set(bitwidths).difference(set(mem_bws))
                if mismatched_bw:
                    raise BitwidthMismatchError(
                        tensor,
                        mismatched_bw.pop(),
                        memory_component.get_name(),
                        mem_bws)

                                                         
            datatype = self.format_tensor.get_data_type()
            mem_dts = memory_component.get_data_types()
            if datatype and datatype not in mem_dts:
                raise DatatypeMismatchError(
                    tensor,
                    datatype,
                    memory_component.get_name(),
                    mem_dts)

                                                         
            datatypes = self.format_tensor.get_data_types()
            if datatypes:
                mismatched_dt = set(datatypes).difference(set(mem_dts))
                if mismatched_dt:
                    raise DatatypeMismatchError(
                        tensor,
                        mismatched_dt.pop(),
                        memory_component.get_name(),
                        mem_dts)

                             
        self.constructor = tensor_binding.constructor

                                                               
        if not self.is_output and self.constructor:
            raise IllegalFieldError("constructor", location)

    def check_concrete(self, requires_constructor: bool):
        
        self.format_tensor.check_concrete()

        decl_tensor = self.format_tensor.get_decl_tensor()
        is_input = decl_tensor.get_is_cascade_input()
        is_output = decl_tensor.get_is_cascade_output()

        location = self.get_name() + " tensor binding (when computing the " + \
            self.get_einsum() + " Einsum)"

                                             
        if len(self.memories) > 1:
            raise IllegalFieldError("components", location)

                                                                  
        if self.ranks["unbound"] and (is_input or is_output):
            ranks = [rank.get_name() for rank in self.ranks["unbound"]]
            tensor_type = "input " if is_input else "output "
            cascade_location = "cascade " + tensor_type + location
            raise IllegalUnboundRanksError(ranks, cascade_location)

                                                                  
        if requires_constructor and not self.constructor:
            raise MissingFieldError("constructor", location)

        if not requires_constructor and self.constructor:
            raise IllegalFieldError("constructor", location)

    def check_template(self):
        
        self.format_tensor.check_template()

    def get_bit_width(self) -> int:
        
        bit_width = self.format_tensor.get_bit_width()
        assert bit_width is not None
        return bit_width

    def get_constructor(self) -> Optional[str]:
        
        return self.constructor

    def get_data_type(self) -> str:
        
        data_type = self.format_tensor.get_data_type()
        assert data_type is not None
        return data_type

    def get_einsum(self) -> str:
        
        return self.format_tensor.get_einsum()

    def get_format_tensor(self) -> FormattedTensor:
        
        return self.format_tensor

    def get_memories(self) -> List[Tuple[MemoryComponent, Optional[str]]]:
        
        return self.memories

    def get_memory(self) -> MemoryComponent:
        
        assert len(self.memories) == 1

        return self.memories[0][0]

    def get_name(self) -> str:
        
        return self.format_tensor.get_name()

    def get_port(self) -> Optional[str]:
        
        assert len(self.memories) == 1

        return self.memories[0][1]

    def get_ranks(self) -> List[PartitionedRank]:
        
        return self.format_tensor.get_ranks()

    def get_rank_binding(self) -> Dict[str, List[PartitionedRank]]:
        
        return self.ranks

    def match_layout(self,
                     other: "BoundTensor",
                     rank_match: Dict[PartitionedRank,
                                      PartitionedRank],
                     self_location: str,
                     other_location: str,
                     match_shape: bool):
        
        sloc = self.get_name() + " tensor binding for the " + \
            self_location + " Einsum " + self.get_einsum()
        oloc = other.get_name() + " tensor binding for the " + \
            other_location + " Einsum " + other.get_einsum()
        self.__check_component_match(
            other, self_location != "concrete", sloc, oloc)

        for mem_rank, sranks in self.ranks.items():
            oranks = other.ranks[mem_rank]
            prev_o_pos = -1
            for srank in sranks:
                                                       
                if srank not in rank_match.keys():
                    continue

                                                                           
                      
                orank = rank_match[srank]
                if orank not in oranks:
                    for o_mem_rank in other.ranks.keys():
                        if orank in other.ranks[o_mem_rank]:
                            break

                    raise MemoryRankBindingMismatchError(
                        srank.get_name(),
                        self.get_name(),
                        srank.get_einsum(),
                        self_location,
                        orank.get_name(),
                        other.get_name(),
                        orank.get_einsum(),
                        other_location,
                        mem_rank,
                        o_mem_rank)

                                                            
                o_pos = oranks.index(orank)
                if o_pos <= prev_o_pos:
                    sloc = "ranks bound to the " + mem_rank + " memory rank for the " + self.get_name() + \
                        " tensor in Einsum " + srank.get_einsum() + " in the " + self_location + " tensor binding"
                    oloc = "ranks bound to the " + mem_rank + " memory rank for the " + other.get_name() + \
                        " tensor in Einsum " + orank.get_einsum() + " in the " + other_location + " tensor binding"
                    raise RankOrderMismatchError(sloc, oloc)

                prev_o_pos = o_pos

                if match_shape:
                    sloc = srank.get_name() + " rank for the " + self.get_name() + " tensor in Einsum " + \
                        srank.get_einsum() + " in the " + self_location + " tensor binding"
                    oloc = orank.get_name() + " rank for the " + other.get_name() + " tensor in Einsum " + \
                        orank.get_einsum() + " in the " + other_location + " tensor binding"

                                                            
                    if srank.get_roots() != orank.get_roots():
                        raise RootRankMismatchError(sloc, oloc)

                                                                 
                    if srank.get_abs_shapes() != orank.get_abs_shapes():
                        raise ShapeMismatchError(
                            "the absolute shapes for the " +
                            sloc +
                            " and the " +
                            oloc,
                            "",
                            str(srank.get_abs_shapes()),
                            str(orank.get_abs_shapes()))

                                                                 
                    if srank.get_rel_shapes() != orank.get_rel_shapes():
                        raise ShapeMismatchError(
                            "the relative shapes for the " +
                            sloc +
                            " and the " +
                            oloc,
                            "",
                            str(srank.get_rel_shapes()),
                            str(orank.get_rel_shapes()))

    def match_with_reshape(self, other: "BoundTensor"):
        
        assert self.get_name() == other.get_name()

        sloc = self.get_name() + " tensor binding for Einsum " + self.get_einsum()
        oloc = other.get_name() + " tensor binding for Einsum " + other.get_einsum()
        self.__check_component_match(other, False, sloc, oloc)

        spart = self.get_format_tensor().get_part_tensor().get_partitioning()
        opart = other.get_format_tensor().get_part_tensor().get_partitioning()

        equalities = spart.get_solver_equalities(opart)

        for mem_rank in self.ranks.keys():
            sranks = PartitionedRank.scoped_ranks(
                [rank for rank in self.ranks[mem_rank] if rank.get_shape() != 1])
            oranks = PartitionedRank.scoped_ranks(
                [rank for rank in other.ranks[mem_rank] if rank.get_shape() != 1])

                                                                                
                                   
            if not OrderSolver.check_eq(equalities, sranks, oranks):
                prefix = mem_rank + " memory rank of the " + \
                    self.get_name() + " tensor binding (when computing Einsum "
                raise RankOrderMismatchError(
                    prefix + self.get_einsum() + ")",
                    prefix + other.get_einsum() + ")")

    def need_explicit_reshape(self, other: "BoundTensor") -> bool:
        
        for mem_rank in self.get_rank_binding():
            self_ranks = self.get_rank_binding()[mem_rank]
            other_ranks = other.get_rank_binding()[mem_rank]

            if len(self_ranks) != len(other_ranks):
                return True

            for self_rank, other_rank in zip(self_ranks, other_ranks):
                if self_rank.get_shape() != other_rank.get_shape():
                    return True

        return False

    def __check_component_match(
            self,
            other: "BoundTensor",
            is_template: bool,
            self_location: str,
            other_location: str):
        
        def format_memories(memories):
            locs = []
            for memory, port in memories:
                locs.append(memory.get_name())
                if port is not None:
                    locs[-1] += " with port " + port

            if len(memories) == 1:
                return locs[0]

            return "one of [" + ", ".join(locs) + "]"

        if is_template:
            if not set(other.memories).issubset(self.memories):
                raise ComponentMismatchError(
                    self_location,
                    other_location,
                    format_memories(self.memories),
                    format_memories(other.memories))

                              
        else:
            if set(self.memories) != set(other.memories):
                raise ComponentMismatchError(
                    self_location,
                    other_location,
                    format_memories(self.memories),
                    format_memories(other.memories))

    def to_str_repr(self) -> StringTensor:
        
        return self.format_tensor.to_str_repr()

    def key(self) -> Tuple[Any, ...]:
        return self.format_tensor, self.ranks, self.constructor, self.memories

    def __hash__(self) -> int:
        
        return hash(("BoundTensor", self.get_einsum(), self.get_name()))
