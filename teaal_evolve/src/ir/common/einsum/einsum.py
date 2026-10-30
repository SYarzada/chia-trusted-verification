
from sympy import Symbol                
from typing import Any, Tuple

from src.common import *
from src.ir.common.access import *
from src.ir.common.op import *
from src.ir.common.operand import *
from src.ir.common.string import *


class Einsum(Base):
    

    def __init__(
            self,
            name: str,
            reuse_spec: ReuseSpecification,
            op_spec: OperatorSpecification):
        self.name = name
        self.reuse_spec = reuse_spec
        self.op_spec = op_spec

    def gen(self) -> str:
        
        reuse_spec = self.reuse_spec.gen()
        op_spec = self.op_spec.gen()

        if op_spec:
            return reuse_spec + " " + op_spec
        else:
            return reuse_spec

    def get_name(self) -> str:
        
        return self.name

    def get_op_spec(self) -> OperatorSpecification:
        
        return self.op_spec

    def get_reuse_spec(self) -> ReuseSpecification:
        
        return self.reuse_spec

    def is_form_match(
            self,
            other: "Einsum",
            component: str,
            instruction: str) -> bool:
        

        if not self.get_op_spec().is_match(other.get_op_spec()):
            return False

        return self.get_reuse_spec().is_form_match(
            other.get_reuse_spec(), component, instruction)

    def split_for_solver(self) -> "List[Einsum]":
        
        name = self.get_name()

        map_op = self.get_op_spec().get_map()
        reduce_op = self.get_op_spec().get_reduce()

        str_einsum = self.to_str_repr()
        if reduce_op is None:
            return [str_einsum]

                                                
        cascade = []
        if map_op is None:
            reduce_einsum = str_einsum

        else:
                                            
            map_einsum, reduce_einsum = self.__split_map_reduce(str_einsum)
            cascade.append(map_einsum)

                                      

                                                    
        contracted = reduce_einsum.get_reuse_spec().get_contracted_ranks()
        iter_space = list(
            reduce_einsum.get_reuse_spec().get_iter_ranks().values())
        iter_space.sort(key=lambda r: r.get_name())

                                                    
        iter_tensor_ranks = []
        iter_ranks = []
        for rank in iter_space:
            if rank in contracted:
                iter_rank = StringRank(rank.get_name() + "_ITER")

                iter_tensor_ranks.append(iter_rank)
                iter_ranks.append(iter_rank)

            else:
                iter_tensor_ranks.append(rank)

        iter_tensor = StringTensor(name + "_reduce", iter_tensor_ranks)

        init_name = name + "_reduce_init"
        operands = [
            self.__make_iter_operands(
                iter_tensor,
                iter_space,
                iter_ranks,
                len(iter_ranks))[0],
            TemplateOperand(
                name +
                "_reduce_init_val")]

        cascade.append(self.__make_copy(init_name, iter_space, operands))

                                    
        iter_operands = self.__make_iter_operands(
            iter_tensor, iter_space, iter_ranks, 0)
        iter_operands += [reduce_einsum.get_reuse_spec().get_operands()[1]]

        iter_name = name
        iter_ranks_dict = {rank.get_name(): rank for rank in iter_space}
        iter_reuse_spec = ReuseSpecification[StringRank, StringTensor, StringTensorAccess](
            iter_name, iter_ranks_dict, iter_operands)
        iter_op_spec = OperatorSpecification(iter_name, reduce_op, None)

        cascade.append(Einsum(iter_name, iter_reuse_spec, iter_op_spec))

                                                     
        for i, iterative_rank in enumerate(iter_ranks):
                                                                    
                         
            if i == 0:
                continue

            iter_name = name + "_reduce_" + iterative_rank.get_name() + "_update"
            iter_operands = self.__make_iter_operands(
                iter_tensor, iter_space, iter_ranks, i)

            cascade.append(
                self.__make_copy(
                    iter_name,
                    iter_space,
                    iter_operands))

                                                    
        update_accesses = {}
        for iter_rank, tensor_rank in zip(iter_space, iter_tensor_ranks):
            if iter_rank in contracted:
                update_accesses[tensor_rank] = RankAccess(
                    [ConstantTerm(Symbol(iter_rank.get_name()))])
            else:
                update_accesses[tensor_rank] = RankAccess(
                    [VariableTerm(iter_rank)])

        update_operands = [
            reduce_einsum.get_reuse_spec().get_operands()[0],
            self.__make_iter_operands(
                iter_tensor,
                iter_space,
                iter_ranks,
                len(iter_ranks))[1]]
        cascade.append(
            self.__make_copy(
                name + "_copy",
                iter_space,
                update_operands))

        return cascade

    def to_str_repr(self) -> "Einsum":
        
        return Einsum(
            self.get_name(),
            self.get_reuse_spec().to_str_repr(),
            self.get_op_spec())

    def __make_copy(self, name: str,
                    comp_iter_space: List[StringRank],
                    operands: List[Operand[StringTensorAccess]]) -> "Einsum":
        
        iter_ranks = {}
        for rank in comp_iter_space:
            assert isinstance(operands[0], TensorOperand)
            for access in operands[0].get().get_accesses().values():
                for term in access.get_expr():
                    if isinstance(term, VariableTerm) and term.get() == rank:
                        iter_ranks[rank.get_name()] = rank

        reuse_spec = ReuseSpecification[StringRank, StringTensor, StringTensorAccess](
            name, iter_ranks, operands)
        op_spec = OperatorSpecification(name, None, None)

        return Einsum(name, reuse_spec, op_spec)

    def __make_iter_operands(self,
                             tensor: StringTensor,
                             iter_space: List[StringRank],
                             iter_ranks: List[StringRank],
                             i: int) -> List[Operand[StringTensorAccess]]:
        
        output_accesses = {}
        input_accesses = {}
        j = 0
        for iter_rank, tensor_rank in zip(iter_space, tensor.get_ranks()):
            default_access = RankAccess([VariableTerm(iter_rank)])
            if tensor_rank in iter_ranks:
                if j < i:
                    output_accesses[tensor_rank] = RankAccess(
                        [ConstantTerm(0)])
                    input_accesses[tensor_rank] = RankAccess(
                        [ConstantTerm(Symbol(iter_rank.get_name()))])
                elif j == i:
                    output_accesses[tensor_rank] = \
                        RankAccess([VariableTerm(iter_rank), ConstantTerm(1)])
                    input_accesses[tensor_rank] = default_access

                       
                else:
                    output_accesses[tensor_rank] = default_access
                    input_accesses[tensor_rank] = default_access

                j += 1
            else:
                output_accesses[tensor_rank] = default_access
                input_accesses[tensor_rank] = default_access

        return [TensorOperand(StringTensorAccess(tensor, output_accesses)),
                TensorOperand(StringTensorAccess(tensor, input_accesses))]

    def __split_map_reduce(
            self, str_einsum: "Einsum") -> "Tuple[Einsum, Einsum]":
        
        name = self.get_name()

        map_op = self.get_op_spec().get_map()
        reduce_op = self.get_op_spec().get_reduce()

                                       
        operands = str_einsum.get_reuse_spec().get_operands()
        map_tmp_ranks = []
        for operand in operands:
            if not isinstance(operand, TensorOperand):
                continue

            for access in operand.get().get_accesses().values():
                for term in access.get_expr():
                    if not isinstance(term, VariableTerm):
                        continue

                                                              
                    if term.get() not in map_tmp_ranks:
                        map_tmp_ranks.append(term.get())

        map_tmp_accesses = {rank: RankAccess(
            [VariableTerm(rank)]) for rank in map_tmp_ranks}

        map_tmp = TensorOperand(StringTensorAccess(
            StringTensor(name + "_map", map_tmp_ranks),
            map_tmp_accesses))

        map_reuse_spec: ReuseSpecification[StringRank, StringTensor, StringTensorAccess] = ReuseSpecification(
            name + "_map", str_einsum.get_reuse_spec().get_iter_ranks(), [map_tmp] + operands[1:])
        reduce_reuse_spec: ReuseSpecification[
            StringRank, StringTensor, StringTensorAccess] = \
            ReuseSpecification(name, str_einsum.get_reuse_spec().get_iter_ranks(), [operands[0], map_tmp])

                                          
        map_op_spec = OperatorSpecification(name + "_map", map_op, None)
        reduce_op_spec = OperatorSpecification(name, None, reduce_op)

        return (Einsum(name + "_map", map_reuse_spec, map_op_spec),
                Einsum(name, reduce_reuse_spec, reduce_op_spec))

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.reuse_spec, self.op_spec
