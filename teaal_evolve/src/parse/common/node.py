

import abc

from ruamel.yaml.nodes import *                
from typing import Any, Tuple

from src.common import *


class BaseNode(Base, metaclass=abc.ABCMeta):

    def __init__(self, yaml: dict):
        self.spec = yaml

    @staticmethod
    def flatten_spec(spec) -> Any:
        
        if isinstance(spec, BaseNode):
            return BaseNode.flatten_spec(spec.spec)
        elif isinstance(spec, dict):
            return {
                key: BaseNode.flatten_spec(val) for key,
                val in spec.items()}
        elif isinstance(spec, list):
            return [BaseNode.flatten_spec(val) for val in spec]
        else:
            return spec

    def key(self) -> Tuple[Any]:
        
        return self.spec,

    @classmethod
    def from_yaml(cls, constructor, node):
        
        yaml = BaseNode.make_pythonic(node)
        return cls(yaml)

    @staticmethod
    def make_pythonic(node: Any) -> Any:
        if isinstance(node, ScalarNode):
            full_tag = node.tag
                                                                           
                                               
            if full_tag is None:
                return None                    

            tag = full_tag.split(":")[-1]
            if tag == "int":
                return int(node.value)

            elif tag == "float":
                return float(node.value)

            elif tag == "str":
                return node.value

            elif tag == "null":
                return None

            else:
                raise ValueError("Unknown YAML tag: " +
                                 str(node.tag))                    

        elif isinstance(node, SequenceNode):
            return [BaseNode.make_pythonic(elem) for elem in node.value]

        elif isinstance(node, MappingNode):
            preamble = {}
            if node.tag and node.tag[0] == "!":
                preamble["tag"] = node.tag

            return preamble | {
                key.value: BaseNode.make_pythonic(value) for key,
                value in node.value}

        else:
            raise ValueError(
                "Unable to parse node: " +
                str(node))                    


class EngineNode(BaseNode):
    
    yaml_tag = u"!Engine"


class HierarchicalNode(BaseNode):
    
    yaml_tag = u"!Hierarchical"


class MemoryNode(BaseNode):
    
    yaml_tag = u"!Memory"


class ParallelNode(BaseNode):
    
    yaml_tag = u"!Parallel"


NODES = [EngineNode, HierarchicalNode, MemoryNode, ParallelNode]
