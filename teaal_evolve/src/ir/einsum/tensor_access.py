

from lark import Token
from lark.tree import Tree
from sympy import Symbol                
from typing import Any, Dict, List, Tuple

from src.common import *
from src.ir.common import *
from src.ir.cond import *
from src.ir.einsum.rank import DeclaredRank
from src.ir.einsum.tensor import DeclaredTensor


class DeclaredTensorAccess(TensorAccess[DeclaredRank, DeclaredTensor]):
    

    def __init__(self,
                 einsum: str,
                 tensor: DeclaredTensor,
                 ranks: Dict[str, DeclaredRank],
                 tensor_tree: Tree):
        
        self.einsum = einsum
        self.tensor = tensor

        self.conds: List[Condition] = []

        location = self.einsum + " Einsum"

                                                               
        accesses_trees = next(tensor_tree.find_data("accesses")).children

        if len(accesses_trees) != len(tensor.get_ranks()):
            raise RankNumberMismatchError(
                tensor.get_name(), location, len(
                    tensor.get_ranks()), len(accesses_trees))

        self.accesses = {}
        for tensor_rank, access in zip(tensor.get_ranks(), accesses_trees):
            tensor_rank_name = tensor_rank.get_name()

            terms: List[Term[DeclaredRank]] = []
            shape = 0
            for token in access.children:
                assert isinstance(token, Token)

                if token.type == "INT":
                    terms.append(ConstantTerm(int(token)))
                    shape += int(token)
                    continue

                if token.isupper():
                    if str(token) in ranks.keys():
                        terms.append(ConstantTerm(
                            ranks[str(token)].get_shape()))
                        shape += ranks[str(token)].get_shape()
                    else:
                        terms.append(ConstantTerm(Symbol(str(token))))
                        shape += Symbol(str(token))
                    continue

                access_rank_name = str(token).upper()

                                                                          
                if access_rank_name not in ranks:
                    ranks[access_rank_name] = DeclaredRank(
                        access_rank_name, Symbol(access_rank_name))

                terms.append(VariableTerm(ranks[access_rank_name]))
                shape += ranks[access_rank_name].get_shape() - 1

            self.accesses[tensor_rank] = RankAccess[DeclaredRank](terms)

            maybe_cond = ConditionUtils.shape_leq_or_cond(
                shape +
                1,
                tensor_rank.get_shape(),
                "the access " +
                self.accesses[tensor_rank].gen() +
                " and the rank " +
                tensor_rank_name +
                " of tensor " +
                self.tensor.get_name(),
                location)

            if maybe_cond and maybe_cond not in self.conds:
                self.conds.append(maybe_cond)

    def get_accesses(self) -> Dict[DeclaredRank, RankAccess[DeclaredRank]]:
        
        return self.accesses

    def get_conds(self) -> List[Condition]:
        
        return self.conds

    def get_einsum(self) -> str:
        
        return self.einsum

    def get_tensor(self) -> DeclaredTensor:
        
        return self.tensor

    def key(self) -> Tuple[Any, ...]:
        return self.tensor, self.accesses
