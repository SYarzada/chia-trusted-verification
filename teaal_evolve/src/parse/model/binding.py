

from pydantic import AliasGenerator, BaseModel, ConfigDict, Discriminator, Tag
from typing import Annotated, Any, Dict, List, Optional, Union

from src.common import *


def ranks_discriminator(v: Any) -> str:
    
    if isinstance(v, list):
        return "list"

    if isinstance(v, dict):
        return "dict"

    raise MalformedInputError(
        "bound ranks",
        v,
        "should be of type `list` or `dict`")


class EngineBindingModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    einsum: Optional[str] = None
    einsums: Optional[List[str]] = None
    component: str
    instruction: str
    ranks: Annotated[Union[Annotated[List[str],
                                     Tag("list")],
                           Annotated[Dict[str,
                                          List[str]],
                                     Tag("dict")]],
                     Discriminator(ranks_discriminator)]


class ComponentModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    name: str
    port: Optional[str] = None


class TensorBindingModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    name: str
    einsum: str
    component: Optional[ComponentModel] = None
    components: Optional[List[ComponentModel]] = None
    constructor: Optional[str] = None
    ranks: Dict[str, List[str]]


class PipelinedStatementModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    statement: str
    offset: int = 0


class PipeliningModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    loop: str
    order: List[PipelinedStatementModel]


class BindingModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    engine: Optional[List[EngineBindingModel]] = []
    tensor: Optional[List[TensorBindingModel]] = []
    pipelining: Optional[List[PipeliningModel]] = []
