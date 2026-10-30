

from lark.tree import Tree
from typing import Any, Dict, List, Tuple

from src.parse.arch.component import Component
from src.parse.lark.code_gen import CodeGeneration
from src.parse.model import *


class Memory(Component):
    def __init__(self, model: MemoryModel):
        self.model = model

        self.constructors = {
            name: CodeGeneration.parse(constructor) for name,
            constructor in self.model.constructors.items()}

        self.access_template = CodeGeneration.parse(self.model.access_template)
        self.reshape_template = CodeGeneration.parse(
            self.model.reshape_template)

    @classmethod
    def from_dict(cls, dict_: dict) -> "Memory":
        
        return Memory(MemoryModel.model_validate(dict_))

    def get_attributes(self) -> MemoryAttributesModel:
        
        return self.model.attributes

    def get_access_template(self) -> Tree:
        
        return self.access_template

    def get_constructors(self) -> Dict[str, Tree]:
        
        return self.constructors

    def get_name(self) -> str:
        
        return self.model.name

    def get_ranks(self) -> List[str]:
        
        return [rank.name for rank in self.model.attributes.ranks]

    def get_reshape_template(self) -> Tree:
        
        return self.reshape_template

    def key(self) -> Tuple[Any, ...]:
        
        return self.model,
