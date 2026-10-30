
from lark.tree import Tree
from typing import Any, Dict, List, Set, Tuple

from src.common import *
from src.ir.binding.common import *
from src.ir.einsum import *
from src.ir.format import *
from src.ir.mapping import *
from src.ir.memory import *
from src.ir.template import *
from src.parse import *


class EngineComponent(Base):
    

    def __init__(self, engine_parser: EngineParser,
                 memories: Dict[str, MemoryComponent]):
        self.name = engine_parser.get_name()
        self.default_attributes = engine_parser.get_default_attributes()

        self.templates = []
        self.templates_dict = {}
        for template_name, template_parser in engine_parser.get_templates().items():
            template = ElaboratedTemplate(
                self.name, self.default_attributes, template_parser, memories)

            self.templates.append(template)
            self.templates_dict[template_name] = template

    def get_name(self) -> str:
        
        return self.name

    def get_templates(self) -> List[ElaboratedTemplate]:
        
        return self.templates

    def maybe_get_template(
            self,
            name: str,
            location: str) -> ElaboratedTemplate:
        
                                                     
        if name not in self.templates_dict.keys():
            raise UnknownInstructionError(name, location)

        return self.templates_dict[name]

    def key(self) -> Tuple[Any, ...]:
        return self.name, self.default_attributes, self.templates
