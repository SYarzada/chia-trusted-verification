

from typing import Any, Dict, Optional, Tuple

from src.common import *
from src.parse.arch import *
from src.parse.common.node import BaseNode
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase


class Architecture(Base):
    

    def __init__(self, model: ArchitectureModel) -> None:
        struct_nodes = {"tag": "!Hierarchical", "nodes": model.nodes}
        self.arch = Structure.from_dict(struct_nodes)

        self.components = {
            component.get_name(): component for component in self.arch.get_components()}

    @classmethod
    def from_dict(cls, dict_: dict) -> "Architecture":
        
        yaml = BaseNode.flatten_spec(dict_["architecture"])
        return Architecture(ArchitectureModel.model_validate(yaml))

    @classmethod
    def from_file(cls, filename: str) -> "Architecture":
        
        return Architecture.from_dict(ParseBase.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Architecture":
        
        return Architecture.from_dict(ParseBase.parse_str(string))

    def get_components(self) -> Dict[str, Component]:
        
        return self.components

    def key(self) -> Tuple[Any]:
        
        return self.arch,
