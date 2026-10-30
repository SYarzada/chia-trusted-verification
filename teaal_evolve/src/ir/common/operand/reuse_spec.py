
from typing import Any, cast, Dict, Generic, List, Set, Tuple, TypeVar

from src.common import *
from src.ir.common.access import *
from src.ir.common.op import *
from src.ir.common.operand.float_ import FloatOperand
from src.ir.common.operand.operand import Operand
from src.ir.common.operand.template import TemplateOperand
from src.ir.common.operand.tensor import TensorOperand
from src.ir.common.rank import Rank
from src.ir.common.string import *
from src.ir.common.tensor import Tensor
from src.ir.solver import *


R = TypeVar("R")
T = TypeVar("T")
TA = TypeVar("TA")
R2 = TypeVar("R2")
T2 = TypeVar("T2")
TA2 = TypeVar("TA2")


class ReuseSpecification(Base, Generic[R, T, TA]):
    

    def __init__(self,
                 einsum_name: str,
                 iter_ranks: Dict[str,
                                  R],
                 operands: List[Operand[TA]]):
        self.einsum = einsum_name
        self.operands = operands
        self.iter_ranks = iter_ranks

    def check_concrete(self):
        
        for operand in self.operands:
            if isinstance(operand, TemplateOperand):
                raise TemplateInConcreteError(
                    operand.get(), self.einsum + " Einsum")

    def check_template(self):
        
                                           
        pass

    def gen(self) -> str:
        
        operands = [operand.gen() for operand in self.operands]

        return operands[0] + " = " + " . ".join(operands[1:])

    def get_all_ranks(self) -> Set[R]:
        
        ranks = set(self.iter_ranks.values())

        for operand in self.operands:
            if not isinstance(operand, TensorOperand):
                continue

            ranks.update(operand.get().get_tensor().get_ranks())

        return ranks

    def get_contracted_ranks(self) -> List[R]:
        
        uncontracted_ranks = set()
        output_accesses = self.get_output().get().get_accesses()
        for rank_access in output_accesses.values():
            for term in rank_access.get_expr():
                if isinstance(term, VariableTerm):
                    uncontracted_ranks.add(term.get())

        contracted_ranks = []
        for tensor_access in self.get_tensor_accesses()[1:]:
            for rank_access in tensor_access.get_accesses().values():
                for term in rank_access.get_expr():
                    if not isinstance(term, VariableTerm):
                        continue

                    rank = term.get()
                    if rank not in uncontracted_ranks and \
                            rank not in contracted_ranks:
                        contracted_ranks.append(rank)

        return contracted_ranks

    def get_einsum(self) -> str:
        
        return self.einsum

    def get_inputs(self) -> List[Operand[TA]]:
        
        return self.operands[1:]

    def get_iter_ranks(self) -> Dict[str, R]:
        
        return self.iter_ranks

    def get_operands(self) -> List[Operand[TA]]:
        
        return self.operands

    def get_output(self) -> TensorOperand[TA]:
        
        return cast(TensorOperand, self.operands[0])

    def get_output_tensor(self) -> T:
        
        return self.get_output().get().get_tensor()

    def get_solver_repr(self) -> SolverEinsum:
        
        tensors = []
        for operand in self.operands:
            if not isinstance(operand, TensorOperand):
                continue

            accesses = []
            for access_ir in operand.get().get_accesses().values():

                access = []
                for term in access_ir.get_expr():
                    if isinstance(term, VariableTerm):
                        access.append(term.get().get_name().lower())

                    if isinstance(term, ConstantTerm):
                        const = term.get()
                        if isinstance(const, int):
                            access.append(const)
                        else:
                            access.append(str(const))

                accesses.append(access)

            tensor_name = operand.get().get_tensor().get_name()
            tensors.append((tensor_name, accesses))

        return (self.einsum, tensors)

    def get_tensors(self) -> List[T]:
        
        tensors = []
        for operand in self.operands:
            if isinstance(operand, TensorOperand):
                tensors.append(operand.get().get_tensor())
        return tensors

    def get_tensor_accesses(self) -> List[TA]:
        
        tensor_accesses = []
        for operand in self.operands:
            if isinstance(operand, TensorOperand):
                tensor_accesses.append(operand.get())
        return tensor_accesses

    def is_form_match(
            self,
            other: "ReuseSpecification",
            component: str,
            instruction: str) -> bool:
        
        opt_error, templates = ReuseSpecification.__match_template_form(
            self, other, component, instruction)

        return opt_error is None

    def match_template_form(
            self,
            other: "ReuseSpecification",
            component: str,
            instruction: str) -> List[Tuple[Optional[str], str, str]]:
        
        opt_error, templates = ReuseSpecification.__match_template_form(
            self, other, component, instruction)

        if opt_error:
            einsums, location, component, instruction, expected, got = opt_error
            raise TemplateMismatchError(
                einsums, location, component, instruction, expected, got)

        return templates

    def maybe_get_tensor_accesses(
            self,
            tensor: str,
            location: str) -> List[TA]:
        
                                                                
        tensor_accesses = []
        for operand in self.operands:
            if isinstance(operand, TensorOperand) and \
                    operand.get().get_tensor().get_name() == tensor:
                tensor_accesses.append(operand.get())

        if not tensor_accesses:
            raise UnknownTensorError(tensor, location)

        return tensor_accesses

    def to_str_repr(
            self) -> "ReuseSpecification[StringRank, StringTensor, StringTensorAccess]":
        
        str_iter_space = {rank.get_name(): rank.to_str_repr()
                          for rank in self.iter_ranks.values()}
        str_operands: List[Operand[StringTensorAccess]] = []
        for operand in self.operands:
            if isinstance(
                    operand,
                    TemplateOperand) or isinstance(
                    operand,
                    FloatOperand):
                str_operands.append(operand)
                continue

            tensor_access = operand.get()

            str_tensor = tensor_access.get_tensor().to_str_repr()

            str_accesses: Dict[StringRank, RankAccess[StringRank]] = {}
            for rank, access in tensor_access.get_accesses().items():
                str_rank = rank.to_str_repr()

                str_access: List[Term[StringRank]] = []
                for term in access.get_expr():
                    if not isinstance(term, VariableTerm):
                        str_access.append(term)
                        continue

                    str_term = VariableTerm(term.get().to_str_repr())
                    str_access.append(str_term)

                str_accesses[str_rank] = RankAccess(str_access)

            str_operands.append(
                TensorOperand(
                    StringTensorAccess(
                        str_tensor,
                        str_accesses)))

        return ReuseSpecification(self.einsum, str_iter_space, str_operands)

    def update_spec(
            self, iter_ranks: Dict[str, R2], tensor_accesses: List[TA2]) -> "ReuseSpecification[R2, T2, TA2]":
        
        i = 0
        new_operands: List[Operand[TA2]] = []
        for old_operand in self.operands:
            if isinstance(old_operand, FloatOperand) or \
                    isinstance(old_operand, TemplateOperand):
                new_operands.append(old_operand)
                continue

                                                                    
            assert isinstance(old_operand, TensorOperand)
            assert old_operand.get().get_tensor().get_name() == \
                tensor_accesses[i].get_tensor().get_name()

            new_operands.append(TensorOperand(tensor_accesses[i]))
            i += 1

        return ReuseSpecification(self.einsum, iter_ranks, new_operands)

    def __match_template_form(
            self,
            other: "ReuseSpecification",
            component: str,
            instruction: str
        ) -> Tuple[Optional[Tuple[List[str], str, str, str, str, str]],
                   List[Tuple[Optional[str], str, str]]]:
        
        coperands = self.operands
        toperands = other.operands

                                                                  
        if len(coperands) != len(toperands):
            error = (
                [self.einsum],
                "reuse specification",
                component,
                instruction,
                str(len(toperands) - 1) + " input(s)",
                str(len(coperands) - 1) + " input(s)")
            return error, []

        templates: List[Tuple[Optional[str], str, str]] = \
            [("einsum", other.einsum, self.einsum)]

        for i, (coperand, toperand) in enumerate(zip(coperands, toperands)):

                                                       
            if isinstance(coperand, FloatOperand) and \
                    isinstance(toperand, TemplateOperand):
                templates.append(
                    (None, toperand.get(), str(
                        coperand.get())))
                continue

                                                                           
                           
            if not isinstance(coperand, type(toperand)):
                error = (
                    [self.einsum],
                    Utils.nth_as_string(i) +
                    " input of the reuse specification",
                    component,
                    instruction,
                    toperand.get_type(),
                    coperand.get_type())
                return error, []

                                                       
            if isinstance(coperand, TensorOperand):
                templates.append(("tensor",
                                  toperand.get().get_tensor().get_name(),
                                  coperand.get().get_tensor().get_name()))

                                                                 
            if isinstance(coperand, FloatOperand) and \
                    coperand != toperand:
                error = (
                    [self.einsum],
                    Utils.nth_as_string(i) +
                    " input of the reuse specification",
                    component,
                    instruction,
                    str(toperand.get()),
                    str(coperand.get()))
                return error, []

        return None, templates

    def key(self) -> Tuple[Any, ...]:
        return self.einsum, self.operands
