

from collections import Counter

from lark.tree import Tree
from typing import Any, cast, Dict, List, Optional, Set, Tuple

from src.common import *
from src.parse.lark import *
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase
from src.common import *


class Binding(Base):
    

    def __init__(self, model: BindingModel):
                                    
        self.einsum_to_binding: Dict[str, Tuple[int, int]] = {}
        if model.engine is None:
            self.engine_bindings = []
        else:
            for i, engine_binding in enumerate(model.engine):
                                                                    
                if engine_binding.einsum and engine_binding.einsums:
                    raise IllegalFieldError(
                        "einsum",
                        "[" +
                        ", ".join(
                            engine_binding.einsums) +
                        "] engine binding")
                if not engine_binding.einsum and not engine_binding.einsums:
                    raise MissingFieldError(
                        "einsums", Utils.nth_as_string(i) + " engine binding")

                                                     
                if engine_binding.einsum:
                    engine_binding.einsums = [engine_binding.einsum]
                    engine_binding.einsum = None

                einsums = engine_binding.einsums
                assert isinstance(einsums, list)

                for j, einsum in enumerate(einsums):
                    if einsum in self.einsum_to_binding.keys():
                        raise RepeatedBindingError(
                            "engine", einsum + " Einsum")

                    self.einsum_to_binding[einsum] = (i, j)

                if isinstance(engine_binding.ranks, list):
                    engine_binding.ranks = {einsums[0]: engine_binding.ranks}

                location = "`ranks` of the " + \
                    "[" + ", ".join(einsums) + "] engine binding"
                ErrorUtils.check_einsums(set(einsums), list(
                    engine_binding.ranks.keys()), location)

            self.engine_bindings = model.engine

                                    
        self.tensor: Dict[str, Dict[str, TensorBindingModel]] = {}
        if model.tensor is not None:
            for tensor_binding in model.tensor:
                if tensor_binding.einsum not in self.tensor.keys():
                    self.tensor[tensor_binding.einsum] = {}

                tensor_bindings = self.tensor[tensor_binding.einsum]
                if tensor_binding.name in tensor_bindings.keys():
                    raise RepeatedBindingError(
                        "tensor",
                        tensor_binding.name +
                        " tensor in the " +
                        tensor_binding.einsum +
                        " Einsum")

                tensor_bindings[tensor_binding.name] = tensor_binding

                                            
        self.pipelining: Dict[str, List[Tuple[Tree, int]]] = {}
        if model.pipelining is not None:
            for pipelining in model.pipelining:
                if pipelining.loop in self.pipelining.keys():
                    raise RepeatedBindingError(
                        "pipelining", pipelining.loop + " loop")

                stmts = []
                for stmt in pipelining.order:
                    stmts.append(
                        (Statement.parse(
                            stmt.statement),
                            stmt.offset))

                self.pipelining[pipelining.loop] = stmts

    @classmethod
    def from_dict(cls, dict_: dict) -> "Binding":
        
        return Binding(BindingModel.model_validate(dict_["binding"]))

    @classmethod
    def from_file(cls, filename: str) -> "Binding":
        
        return Binding.from_dict(ParseBase.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Binding":
        
        return Binding.from_dict(ParseBase.parse_str(string))

    def get_engine(self, einsum: str) -> str:
        
        return self.__getattr(einsum, "component")

    def get_engine_bindings(self) -> List[EngineBindingModel]:
        
        return self.engine_bindings

    def get_engine_ranks(self, einsum: str) -> List[str]:
        
        ranks_dict = self.__getattr(einsum, "ranks")
        assert isinstance(ranks_dict, dict)

        return ranks_dict[einsum]

    def get_instruction(self, einsum: str) -> str:
        
        return self.__getattr(einsum, "instruction")

    def get_pipelining(self) -> Dict[str, List[Tuple[Tree, int]]]:
        
        return self.pipelining

    def get_position(self, einsum: str) -> int:
        
        return self.einsum_to_binding[einsum][1]

    def get_tensor_bindings(self) -> Dict[str, Dict[str, TensorBindingModel]]:
        
        return self.tensor

    def __getattr(self, einsum: str, attr: str) -> Any:
        
        if einsum not in self.einsum_to_binding.keys():
            raise MissingEinsumError(einsum, "engine binding")

        binding = self.engine_bindings[self.einsum_to_binding[einsum][0]]

        return getattr(binding, attr)

    def key(self) -> Tuple[Any, ...]:
        
        return self.engine_bindings, self.tensor, self.pipelining
