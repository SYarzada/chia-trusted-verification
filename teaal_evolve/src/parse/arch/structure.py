

from typing import Any, Dict, List, Tuple, Union

from src.common import *
from src.parse.arch.component import Component
from src.parse.arch.engine import Engine
from src.parse.arch.memory import Memory
from src.parse.model import *


class Structure(Base):
    

    def __init__(self, model: StructureModel):
        self.nodes: List[Union[Component, Structure]] = []
        for node in model.nodes:
            if "tag" not in node:
                raise UntaggedNodeError(node)

            elif node["tag"] == "!Engine":
                self.nodes.append(Engine.from_dict(node))

            elif node["tag"] == "!Memory":
                self.nodes.append(Memory.from_dict(node))

            elif node["tag"] == "!Hierarchical" or node["tag"] == "!Parallel":
                self.nodes.append(Structure.from_dict(node))

            else:
                                                                              
                                                                          
                raise NotImplementedError                    

    def get_components(self) -> List[Component]:
        
        components: List[Component] = []

        for node in self.nodes:
            if isinstance(node, Component):
                components.append(node)

            else:
                components.extend(node.get_components())

        return components

    @classmethod
    def from_dict(cls, dict_: dict) -> "Structure":
        
        return Structure(StructureModel.model_validate(dict_))

    def key(self) -> Tuple[Any]:
        
        return self.nodes,
