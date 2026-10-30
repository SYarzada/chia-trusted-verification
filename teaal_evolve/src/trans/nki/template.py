

from sympy import Symbol                
from typing import Dict, Optional

from src.common import *
from src.ir import *
from src.python import *
from src.trans.python import *


class NKITemplate:
    
    @staticmethod
    def translate_access(
            program: ElaboratedProgram,
            bound_tensor: BoundTensor,
            access_ranges: Dict[PartitionedRank,
                                Tuple[RankAccess[PartitionedRank],
                                      RankAccess[PartitionedRank]]],
            loop_ranks: Dict[PartitionedRank, str],
            renamed_tensors: Dict[str, str]) -> ECode:
        
        memory = bound_tensor.get_memory()

        template = memory.get_access_template()
        location = memory.get_name() + "'s access template"

        return ECode(
            NKITemplate.__translate_memory_template(
                program,
                template,
                location,
                bound_tensor,
                access_ranges=access_ranges,
                loop_ranks=loop_ranks,
                renamed_tensors=renamed_tensors))

    @staticmethod
    def translate_constructor(
            program: ElaboratedProgram, bound_tensor: BoundTensor) -> SCode:
        
        constructor = bound_tensor.get_constructor()
        assert constructor is not None
        memory = bound_tensor.get_memory()

        template = memory.get_constructors()[constructor]
        location = memory.get_name() + "'s `" + constructor + "` constructor"

        return SCode(
            NKITemplate.__translate_memory_template(
                program, template, location, bound_tensor))

    @staticmethod
    def translate_instruction(
            program: ElaboratedProgram,
            instruction: Instruction,
            loop_ranks: Dict[PartitionedRank, str],
            renamed_tensors: Dict[str, str]) -> SBlock:
        
        location = instruction.get_engine() + "'s " + instruction.get_instruction() + \
            " code generation"

        block = SBlock([])
        for template in instruction.get_line_gen():
            string = ""

            for piece in template.get():

                if isinstance(piece, CodeGeneration):
                    string += piece.get()
                    continue

                assert isinstance(piece, TemplateGeneration)

                fields = piece.get()
                piece_location = (str(piece), location)
                if fields[0] == "tensor":
                    if len(fields) < 3:
                        raise UnknownTemplateError(*piece_location)

                    tensor = fields[1]
                    tensor_accesses = instruction.maybe_get_tensor_accesses(
                        tensor, location)
                    bound_tensor = tensor_accesses[0].get_tensor()

                    access_ranges = BoundTensorAccess.merge_accesses(
                        tensor_accesses, program.get_rank_expression_fn(loop_ranks), tensor + " tensor")

                    if fields[2] == "access":
                        string += NKITemplate.translate_access(
                            program, bound_tensor, access_ranges, loop_ranks, renamed_tensors).gen()

                    else:
                        string += NKITemplate.__translate_tensor_fields(
                            program, fields[2:], piece_location, bound_tensor, access_ranges, loop_ranks, renamed_tensors)
                    continue

                elif fields[0] == "einsum":
                    einsum = fields[1]
                    bound_einsum = instruction.maybe_get_einsum(
                        einsum, location)

                    if fields[2] == "name":
                        string += bound_einsum.get_name()
                        continue

                    elif fields[2] == "ranks":
                        if len(fields) != 6:
                            raise UnknownTemplateError(*piece_location)

                        all_ranks = bound_einsum.get_loop_order()
                        bound_ranks = bound_einsum.get_bound_ranks()
                        if fields[3] == "all":
                            map_ranks = all_ranks
                        elif fields[3] == "bound":
                            map_ranks = bound_ranks
                        elif fields[3] == "unbound":
                            def is_unbound(
                                map_rank): return map_rank not in bound_ranks
                            map_ranks = list(filter(is_unbound, all_ranks))
                        else:
                            raise UnknownTemplateError(*piece_location)

                        contracted_ranks = bound_einsum.get_contracted_ranks()
                        if fields[4] == "all":
                            pass
                        elif fields[4] == "contracted":
                            def is_contracted(
                                map_rank): return map_rank in contracted_ranks
                            map_ranks = list(filter(is_contracted, map_ranks))
                        elif fields[4] == "uncontracted":
                            def is_uncontracted(
                                map_rank): return map_rank not in contracted_ranks
                            map_ranks = list(
                                filter(is_uncontracted, map_ranks))
                        else:
                            raise UnknownTemplateError(*piece_location)

                        part_ranks = [map_rank.get_rank()
                                      for map_rank in map_ranks]
                        string += NKITemplate.__translate_rank_field(
                            part_ranks, fields[5], piece_location, loop_ranks)
                        continue

                raise UnknownTemplateError(*piece_location)

            block.add(SCode(string))

        return block

    @staticmethod
    def translate_reshape(
            program: ElaboratedProgram,
            bound_tensor: BoundTensor) -> ECode:
        
        memory = bound_tensor.get_memory()

        template = memory.get_reshape_template()
        location = memory.get_name() + "'s reshape template"

        return ECode(
            NKITemplate.__translate_memory_template(
                program, template, location, bound_tensor))

    @staticmethod
    def __translate_memory_template(
            program: ElaboratedProgram,
            template: LineGeneration,
            location: str,
            bound_tensor: BoundTensor,
            access_ranges: Optional[Dict[PartitionedRank, Tuple[RankAccess[PartitionedRank], RankAccess[PartitionedRank]]]] = None,
            loop_ranks: Optional[Dict[PartitionedRank, str]] = None,
            renamed_tensors: Optional[Dict[str, str]] = None) -> str:
        
        string = ""
        for piece in template.get():

            if isinstance(piece, CodeGeneration):
                string += piece.get()
                continue

            assert isinstance(piece, TemplateGeneration)

            fields = piece.get()
            if fields[0] == "tensor":
                string += NKITemplate.__translate_tensor_fields(program, fields[1:], (str(
                    piece), location), bound_tensor, access_ranges, loop_ranks, renamed_tensors)

        return string

    @staticmethod
    def __translate_tensor_fields(
            program: ElaboratedProgram,
            fields: List[str],
            location: Tuple[str, str],
            bound_tensor: BoundTensor,
            access_ranges: Optional[Dict[PartitionedRank, Tuple[RankAccess[PartitionedRank], RankAccess[PartitionedRank]]]] = None,
            loop_ranks: Optional[Dict[PartitionedRank, str]] = None,
            renamed_tensors: Optional[Dict[str, str]] = None) -> str:

        if fields[0] == "name":
            if len(fields) > 1:
                raise UnknownTemplateError(*location)

            tensor = bound_tensor.get_name()
            if renamed_tensors is not None:
                return renamed_tensors[tensor]
            else:
                return tensor

        if fields[0] == "ranks":
            if len(fields) < 2:
                raise UnknownTemplateError(*location)

            if fields[1] == "linearized_offset":
                if access_ranges is None or loop_ranks is None:
                    raise IllegalTemplateError(*location)

                offset = 0
                multiplier = 1

                for mem_rank in reversed(
                        bound_tensor.get_memory().get_ranks()):
                    for trank in reversed(
                            bound_tensor.get_rank_binding()[mem_rank]):
                        start_access = access_ranges[trank][0].get_expr()
                        for term in start_access:
                            if isinstance(term, ConstantTerm):
                                offset += term.get() * multiplier

                        multiplier *= trank.get_shape()

                return Equation.trans(offset).gen()

            if len(fields) != 3:
                raise UnknownTemplateError(*location)

            mem_rank = fields[1]
            if mem_rank not in bound_tensor.get_rank_binding().keys():
                rank_loc = location[0] + " template in the " + location[1]
                raise UnknownMemoryRankError(mem_rank, rank_loc)

            tensor_ranks = bound_tensor.get_rank_binding()[mem_rank]
            if fields[2] == "accesses":
                if access_ranges is None or loop_ranks is None:
                    raise IllegalTemplateError(*location)

                access_str = []
                for trank in tensor_ranks:
                    start_access = access_ranges[trank][0].get_expr()
                    end_access = access_ranges[trank][1].get_expr()

                                                                       
                                
                    access_sympy = 0
                    if start_access == end_access and \
                            all(isinstance(term, ConstantTerm)
                                for term in start_access):
                        for term in start_access:
                            access_sympy += term.get()

                        access_str.append(Equation.trans(access_sympy).gen())
                        continue

                                                 
                    start = 0
                    for term in start_access:
                        if isinstance(term, ConstantTerm):
                            start += term.get()

                                                          

                    end = 0
                    for term in end_access:
                        if isinstance(term, ConstantTerm):
                            end += term.get()
                            continue

                        assert isinstance(term, VariableTerm)

                        end += term.get().get_shape() - 1

                    end += 1

                    access = ""
                    if start != 0:
                        access += Equation.trans(start).gen()
                    access += ":"
                    if end != trank.get_shape():
                        access += Equation.trans(end).gen()

                    access_str.append(access)

                if access_str:
                    return ", ".join(access_str)
                else:
                    return "0"

            elif fields[2] == "pattern":
                if access_ranges is None or loop_ranks is None:
                    raise IllegalTemplateError(*location)

                pattern: List[EList] = []
                multiplier = 1
                for all_mem_rank in reversed(
                        bound_tensor.get_memory().get_ranks()):
                    for trank in reversed(
                            bound_tensor.get_rank_binding()[all_mem_rank]):
                                                                         
                                  
                        if all_mem_rank != mem_rank:
                            multiplier *= trank.get_shape()
                            continue

                        start_access = access_ranges[trank][0].get_expr()
                        end_access = access_ranges[trank][1].get_expr()

                                                               
                        if start_access == end_access and \
                                all(isinstance(term, ConstantTerm)
                                    for term in start_access):
                            multiplier *= trank.get_shape()
                            continue

                                                     
                        start = 0
                        for term in start_access:
                            if isinstance(term, ConstantTerm):
                                start += term.get()

                                                              

                        end = 0
                        for term in end_access:
                            if isinstance(term, ConstantTerm):
                                end += term.get()
                                continue

                            assert isinstance(term, VariableTerm)

                            end += term.get().get_shape() - 1

                        end += 1

                        pattern.insert(
                            0, EList([Equation.trans(multiplier), Equation.trans(end - start)]))

                        multiplier *= trank.get_shape()

                    if mem_rank == all_mem_rank:
                        if len(pattern) > 0:
                            return EList(pattern).gen()
                        else:
                            return EList(
                                [EList([Equation.trans(multiplier), EInt(1)])]).gen()

            part_ranks = tensor_ranks

            return NKITemplate.__translate_rank_field(
                part_ranks, fields[2], location, loop_ranks)

        if fields[0] == "bit_width":
            if len(fields) > 1:
                raise UnknownTemplateError(*location)

            return str(bound_tensor.get_bit_width())

        if fields[0] == "data_type":
            if len(fields) > 1:
                raise UnknownTemplateError(*location)

            return bound_tensor.get_data_type()

        raise UnknownTemplateError(*location)

    @staticmethod
    def __translate_rank_field(
            part_ranks: List[PartitionedRank],
            field: str,
            location: Tuple[str, str],
            loop_ranks: Optional[Dict[PartitionedRank, str]] = None) -> str:

        if field == "rank_vars":
            assert loop_ranks is not None

            return "[" + ", ".join(loop_ranks[prank]
                                   for prank in part_ranks) + "]"

        if field == "shapes":
            shapes = [Equation.trans(prank.get_shape())
                      for prank in part_ranks]
            if shapes:
                return ", ".join(shape.gen() for shape in shapes)
            else:
                return "1"

        raise UnknownTemplateError(*location)
