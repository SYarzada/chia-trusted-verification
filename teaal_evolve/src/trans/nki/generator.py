
from typing import Dict, Set, Tuple

from src.ir import *
from src.python import *
from src.trans.nki.template import NKITemplate
from src.trans.python import *


class NKIGenerator:
    

    def __init__(self, program: ElaboratedProgram):
        self.program = program
        block_stack = [SBlock([])]

        loop_ranks: Dict[PartitionedRank, str] = {}

        for node in self.program.sort_nodes():
            if isinstance(node, BodyNode):
                block_stack[-1].add(self.__add_constructors(node))

                                              
                block, renamed_tensors = self.__reshape_inputs_for_instruction(
                    node.get_instruction())
                block_stack[-1].add(block)

                                          
                body = NKITemplate.translate_instruction(
                    self.program, node.get_instruction(), loop_ranks, renamed_tensors)

                block_stack[-1].add(self.__wrap_in_offset(node,
                                    body, loop_ranks))

            elif isinstance(node, EndLoopNode):
                block_stack.pop(-1)

            elif isinstance(node, LoopNode):
                block_stack[-1].add(self.__add_constructors(node))

                                                                           
                loop_name = node.get_rank().get_name()
                if "-" in loop_name:
                    loop_rank = node.get_rank().get_ranks()[0]
                    payload = loop_rank.get_einsum().lower() + "_" + loop_rank.get_name().lower()
                elif "." in loop_name:
                    payload = loop_name.replace(".", "_").lower()
                else:
                    payload = loop_name.lower()

                for map_rank in node.get_rank().get_ranks():
                    loop_ranks[map_rank.get_rank()] = payload

                                       
                if node.get_rank().get_is_sequential():
                    iter_func = "sequential_range"
                else:
                    iter_func = "affine_range"

                shape = Equation.trans(
                    node.get_rank().get_shape() +
                    node.get_rank().get_max_offset())
                expr = EMethod(EVar("nl"), iter_func, [AJust(shape)])
                body = SBlock([])

                for_ = SFor(PVar(payload), expr, body)

                block_stack[-1].add(self.__wrap_in_offset(node,
                                    for_, loop_ranks))
                block_stack.append(body)

            elif isinstance(node, StartNode):
                pass
            else:
                assert False                    

                                          
        args = []
        for decl_tensor in self.program.get_cascade_inputs():
            args.append(EVar(decl_tensor.get_name()))

        for symbol in self.program.get_symbols():
            args.append(EVar(str(symbol)))

        outputs = [EVar(output.get_name())
                   for output in self.program.get_cascade_outputs()]
        if len(outputs) == 1:
            block_stack[0].add(SReturn(outputs[0]))
        elif len(outputs) > 1:
            block_stack[0].add(SReturn(ETuple(outputs)))

        self.code = SFunc(self.program.get_kernel_name(), args, block_stack[0])

    def __str__(self) -> str:
        
        return self.code.gen(0)

    def __add_constructors(self, node: Node) -> Statement:
        
        block = SBlock([])

        bound_tensors = self.program.get_declared_tensor(node)

                                                 
        for bound_tensor in bound_tensors:
            block.add(NKITemplate.translate_constructor(
                self.program, bound_tensor))

        return block

    def __reshape_inputs_for_instruction(
            self, instruction: Instruction) -> Tuple[Statement, Dict[str, str]]:
        
        renamed_tensors = {}
        block = SBlock([])
        for bound_einsum in instruction.get_einsums():
            for bound_tensor in bound_einsum.get_reuse_spec().get_tensors():
                tensor = bound_tensor.get_name()
                first_access = self.program.get_first_access(tensor)

                explicit_reshape = bound_tensor.need_explicit_reshape(
                    first_access)

                if not explicit_reshape:
                    renamed_tensors[tensor] = tensor
                    continue

                new_name = tensor + "_for_" + bound_einsum.get_name()
                reshape = NKITemplate.translate_reshape(
                    self.program, bound_tensor)

                block.add(SAssign(AVar(new_name), reshape))
                renamed_tensors[tensor] = new_name

        return block, renamed_tensors

    def __wrap_in_offset(self, node: Node, body: Statement,
                         loop_ranks: Dict[PartitionedRank, str]) -> Statement:
        
        info = self.program.get_offset(node)

                                                                              
        if not info:
            return body

                                                    
                                                             
        bound_rank, offset = info

        part_rank = bound_rank.get_ranks()[0].get_rank()
        loop_var = loop_ranks[part_rank]

        cond = EBinOp(EBinOp(EVar(loop_var), OGtEq(), EInt(offset)),
                      OLAnd(),
                      EBinOp(EVar(loop_var),
                             OLt(),
                             Equation.trans(bound_rank.get_shape() + offset)))

        return SIf((cond, body), [], None)
