

from lark.tree import Tree
from pydantic import BaseModel
from typing import Any, cast, Dict, List, Optional, Tuple

from src.common import *
from src.parse.lark.equation import Equation
from src.parse.model import *
from src.parse.spec.base import Base as ParseBase


class Options(Base):
    

    def __init__(self, model: OptionsModel):
        self.spec = model.options

    @classmethod
    def from_dict(cls, dict_: dict) -> "Options":
        
        return Options(
            OptionsModel.model_validate({"options": dict_["options"]}))

    @classmethod
    def from_file(cls, filename: str) -> "Options":
        
        return Options(
            OptionsModel.model_validate(
                {"options": ParseBase.parse_file(filename)["options"]}))

    @classmethod
    def from_str(cls, string: str) -> "Options":
        
        return Options(
            OptionsModel.model_validate(
                {"options": ParseBase.parse_str(string)["options"]}))

    def get_fields(self) -> Dict[str, Any]:
        
        return self.spec

    def key(self) -> Tuple[Any]:
        
        return self.spec,
