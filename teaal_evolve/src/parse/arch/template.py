

from lark.tree import Tree
from typing import Any, Dict, List, Tuple

from src.common import *
from src.parse.lark.code_gen import CodeGeneration
from src.parse.model import *
from src.parse.spec.binding import Binding
from src.parse.spec.einsum import Einsum
from src.parse.spec.format import Format
from src.parse.spec.mapping import Mapping
from src.parse.spec.options import Options


class Template(Base):
    def __init__(self, model: TemplateModel):
        self.name = model.name
        self.attributes = model.attributes
        self.einsum = Einsum(model.einsum)
        self.mapping = Mapping(model.mapping)
        self.format = Format.from_dict({"format": model.format_})
        self.binding = Binding(model.binding)
        self.options = Options.from_dict({"options": model.options})
        self.code_gen = [CodeGeneration.parse(line) for line in model.code_gen]

    @classmethod
    def from_dict(cls, dict_: dict) -> "Template":
        
        return Template(TemplateModel.model_validate(dict_))

    def get_attributes(self) -> Optional[TemplateAttributesModel]:
        
        return self.attributes

    def get_binding(self) -> Binding:
        
        return self.binding

    def get_code_gen(self) -> List[Tree]:
        
        return self.code_gen

    def get_einsum(self) -> Einsum:
        
        return self.einsum

    def get_format(self) -> Format:
        
        return self.format

    def get_mapping(self) -> Mapping:
        
        return self.mapping

    def get_name(self) -> str:
        
        return self.name

    def get_options(self) -> Options:
        
        return self.options

    def key(self) -> Tuple[Any, ...]:
        
        return self.name, self.einsum, self.mapping, self.format, self.binding, self.code_gen
