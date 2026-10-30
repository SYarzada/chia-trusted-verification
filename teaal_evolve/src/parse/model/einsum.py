

from pydantic import BaseModel, ConfigDict, Discriminator, Tag
from typing import Annotated, Any, Dict, List, Union

from src.common import *


def expression_discriminator(v: Any) -> str:
    
    if isinstance(v, str):
        return "str"

    if isinstance(v, dict):
        if len(v) == 1:
            return "dict"
        else:
            raise MalformedInputError(
                "expression", v, "each list entry should have exactly one named expression")

    raise MalformedInputError(
        "expression", v, "should be of type `str` or `dict`")


class EinsumModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    declaration: Dict[str, List[str]]
    expressions: List[Annotated[Union[Annotated[str,
                                                Tag("str")],
                                      Annotated[Dict[str,
                                                     str],
                                                Tag("dict")]],
                                Discriminator(expression_discriminator)]]
    shape: Dict[str, int] = {}
