

from typing import Any, Dict, List, Optional, Tuple

from src.common import *
from src.ir.common.op.concrete_arg import ConcreteArgument
from src.ir.common.op.curried_op import CurriedOperator
from src.ir.common.op.int_arg import IntegerArgument
from src.ir.common.op.op import Operator
from src.ir.common.op.simple_op import SimpleOperator
from src.ir.common.op.str_arg import StringArgument
from src.ir.common.op.template_arg import TemplateArgument


class OperatorSpecification(Base):
    

    def __init__(
            self,
            einsum_name: str,
            map_op: Optional[Operator],
            reduce_op: Optional[Operator]):

        self.einsum = einsum_name
        self.map = map_op
        self.reduce = reduce_op

    def check_concrete(self):
        
        for opt_op in [self.map, self.reduce]:
            if opt_op is None:
                continue

            if isinstance(opt_op, SimpleOperator):
                continue

            if isinstance(opt_op, CurriedOperator):
                for arg in opt_op.get_args():
                    if isinstance(arg, TemplateArgument):
                        raise TemplateInConcreteError(
                            arg.get(), self.einsum + " Einsum")
                continue

            raise NotImplementedError                    

    def check_template(self):
        
                                           
        pass

    def gen(self) -> str:
        
        if not self.map and not self.reduce:
            return ""

        string = ";;"

        if self.map:
            string += " M " + self.map.gen()

        if self.reduce:
            string += " V " + self.reduce.gen()

        return string

    def get_einsum(self) -> str:
        
        return self.einsum

    def get_map(self) -> Optional[Operator]:
        
        return self.map

    def get_reduce(self) -> Optional[Operator]:
        
        return self.reduce

    def get_ops(self) -> List[Optional[Operator]]:
        
        return [self.map, self.reduce]

    def is_match(self, other: "OperatorSpecification") -> bool:
        
        if not OperatorSpecification.__match_op(self.map, other.map)[0]:
            return False

        return OperatorSpecification.__match_op(self.reduce, other.reduce)[0]

    def match_template(
            self,
            other: "OperatorSpecification",
            component: str,
            instruction: str) -> List[Tuple[Optional[str], str, str]]:
        
        concrete_ops = self.get_ops()
        template_ops = other.get_ops()
        op_types = ["map", "reduce"]

        templates: List[Tuple[Optional[str], str, str]] = []
        for op_type, cop, top in zip(op_types, concrete_ops, template_ops):
            is_match, op_templates = OperatorSpecification.__match_op(cop, top)

            if not is_match:
                cop_str = cop.gen() if cop else "None"
                top_str = top.gen() if top else "None"

                raise TemplateMismatchError(
                    [self.einsum],
                    op_type + " operator",
                    component,
                    instruction,
                    top_str,
                    cop_str)

            templates.extend(op_templates)

        return templates

    @staticmethod
    def __match_op(cop: Optional[Operator], top: Optional[Operator]
                   ) -> Tuple[bool, List[Tuple[Optional[str], str, str]]]:
        
                                                                       
                               
        if cop == top:
            return True, []

                                                            
        if cop is None:
            assert top is not None
            return False, []

                                                            
        if top is None:
            assert cop is not None
            return False, []

                                                                           
                  
        if isinstance(cop, SimpleOperator) or \
                isinstance(top, SimpleOperator):
            return False, []

                                                         
        assert isinstance(cop, CurriedOperator)
        assert isinstance(top, CurriedOperator)

        cargs = cop.get_args()
        targs = top.get_args()

        if len(cargs) != len(targs):
            return False, []

        templates: List[Tuple[Optional[str], str, str]] = []
        for carg, targ in zip(cargs, targs):
            if carg == targ:
                continue

                                                 
            if isinstance(targ, ConcreteArgument):
                return False, []

                                                  
            assert isinstance(targ, TemplateArgument)

            templates.append((None, targ.get(), carg.gen()))

        return True, templates

    def key(self) -> Tuple[Any, ...]:
        return self.einsum, self.map, self.reduce
