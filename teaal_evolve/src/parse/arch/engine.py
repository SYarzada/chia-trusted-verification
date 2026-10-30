

from lark.tree import Tree
from typing import Any, Dict, List, Tuple

from src.common import *
from src.parse.arch.component import Component
from src.parse.arch.template import Template
from src.parse.model import *


class Engine(Component):
    def __init__(self, model: EngineModel):
        self.model = model

        self.templates = {}
        for template_model in self.model.templates:
            GlobalState.set_location_prefix(
                self.model.name + "'s " + template_model.name + " template")

            template = Template(template_model)
            self.templates[template.get_name()] = template

            GlobalState.clear_location_prefix()

    @classmethod
    def from_dict(cls, dict_: dict) -> "Engine":
        
        return Engine(EngineModel.model_validate(dict_))

    def get_default_attributes(self) -> Optional[TemplateAttributesModel]:
        
        return self.model.default_attributes

    def get_name(self) -> str:
        
        return self.model.name

    def get_templates(self) -> Dict[str, Template]:
        
        return self.templates

    def key(self) -> Tuple[Any, ...]:
        
        return self.model,
