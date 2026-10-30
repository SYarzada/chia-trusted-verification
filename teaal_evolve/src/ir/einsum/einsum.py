

from lark import Token
from lark.tree import Tree
from sympy import Symbol                
from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

from src.common import *
from src.ir.cond import *
from src.ir.common import *
from src.ir.einsum.rank import DeclaredRank
from src.ir.einsum.tensor_access import DeclaredTensorAccess
from src.ir.einsum.tensor import DeclaredTensor


class DeclaredEinsum(Einsum):
    

    def __init__(self,
                 name: Optional[str],
                 expr: Tree,
                 ranks: Dict[str,
                             DeclaredRank],
                 tensors: Dict[str,
                               DeclaredTensor]):
        
        tensor_trees = list(expr.find_data("tensor"))
        tensor_names = [str(tensor_tree.children[0])
                        for tensor_tree in tensor_trees]

        if name is None:
            name = tensor_names[0]

        location = name + " Einsum"

                                                 
        tensor_accesses = []
        uncontracted_ranks = set()
        contracted_ranks = set()
        for i, (tensor_name, tensor_tree) in enumerate(
                zip(tensor_names, tensor_trees)):

                                                 
            if tensor_name not in tensors.keys():
                raise UnknownTensorError(tensor_name, location)
            tensor = tensors[tensor_name]

            tensor_access = DeclaredTensorAccess(
                name, tensor, ranks, tensor_tree)
            for rank_access in tensor_access.get_accesses().values():
                for term in rank_access.get_expr():
                    if not isinstance(term, VariableTerm):
                        continue

                    decl_rank = term.get()

                                                                    
                    if i == 0:
                        uncontracted_ranks.add(decl_rank)

                                                                  
                    elif i > 0 and decl_rank not in uncontracted_ranks:
                        contracted_ranks.add(decl_rank)

            tensor_accesses.append(tensor_access)

                                                                 
        i = 0
        operands: List[Operand] = []
        for child in expr.children:
            if child.data == "int_input" or child.data == "float_input":
                                                                            
                                  
                assert isinstance(child.children[0], Token)

                operands.append(FloatOperand(float(child.children[0])))
                continue

            if child.data == "template_input":
                operands.append(TemplateOperand(str(child.children[0])))
                continue

            if child.data == "tensor":
                operands.append(TensorOperand(tensor_accesses[i]))
                i += 1
                continue

            if child.data == "map" or child.data == "reduce":
                break

                                                                            
            raise NotImplementedError(
                "Unknown parse tree with root " +
                child.data)                    

        iter_ranks = {}
        for operand in operands:
            if not isinstance(operand, TensorOperand):
                continue

            for rank_access in operand.get().get_accesses().values():
                for term in rank_access.get_expr():
                    if isinstance(term, VariableTerm):
                        rank = term.get().get_name()
                        iter_ranks[rank] = term.get()

        reuse_spec: \
            ReuseSpecification[DeclaredRank,
                               DeclaredTensor,
                               DeclaredTensorAccess] \
            = ReuseSpecification(name, iter_ranks, operands)

                                     
        map_tree = list(expr.find_data("map"))
        reduce_tree = list(expr.find_data("reduce"))

                                                                       
        if contracted_ranks and not reduce_tree:
            raise MissingOperatorError("reduce", location)

                                                                           
               
        if not contracted_ranks and reduce_tree:
            raise ExtraOperatorError("reduce", location)

        map_op = None
        reduce_op = None
        if map_tree:
            map_op = DeclaredEinsum.__build_op(map_tree[0])
        if reduce_tree:
            reduce_op = DeclaredEinsum.__build_op(reduce_tree[0])
        op_spec = OperatorSpecification(name, map_op, reduce_op)

        if len(reuse_spec.get_inputs()) > 1:
                                                                           
                                                                           
                                     
            if op_spec.get_map() is None:
                raise MissingOperatorError("map", location)

        for tensor_access in tensor_accesses:
            for tensor_rank, rank_access in tensor_access.get_accesses().items():
                terms = rank_access.get_expr()

                                                                             
                if len(terms) == 1 and isinstance(
                        terms[0], VariableTerm) and terms[0].get() == tensor_rank:
                    continue

                                                                     
                                                         
                if tensor_rank in reuse_spec.get_iter_ranks().values():
                    tensor = tensor_access.get_tensor()
                    pos = tensor.get_ranks().index(tensor_rank)
                    raise RankNameCollisionError(
                        name, tensor.get_name(), tensor_rank.get_name(), pos)

        super().__init__(name, reuse_spec, op_spec)

    def check_concrete(self):
        
        self.get_reuse_spec().check_concrete()
        self.get_op_spec().check_concrete()

    def check_template(self):
        
        self.get_reuse_spec().check_template()
        self.get_op_spec().check_template()

    def get_conds(self) -> List[Condition]:
        
        conds = []
        for tensor_access in self.get_reuse_spec().get_tensor_accesses():
            for cond in tensor_access.get_conds():
                if cond not in conds:
                    conds.append(cond)

        return conds

    @staticmethod
    def __build_op(tree) -> Operator:
        
        simple_op = list(tree.find_data("simple_op"))
        curried_op = list(tree.find_data("curried_op"))

        if simple_op:
            return SimpleOperator(str(simple_op[0].children[0]))

        name = str(curried_op[0].children[0])

        args: List[Argument] = []
        for arg_tree in curried_op[0].children[1:]:
            if arg_tree.data == "str_arg":
                args.append(StringArgument(str(arg_tree.children[0])))

            elif arg_tree.data == "int_arg":
                                                                      
                args.append(IntegerArgument(int(arg_tree.children[0])))

                                         
            elif arg_tree.data == "template_arg":
                args.append(TemplateArgument(str(arg_tree.children[0])))

            else:
                                                                         
                raise NotImplementedError                    

        return CurriedOperator(name, args)

    def __hash__(self) -> int:
        
        return hash(("DeclaredEinsum", self.name))
