

from enum import Enum
from pydantic import AliasGenerator, BaseModel, ConfigDict, Discriminator, Field, Tag
from typing import Annotated, Any, Dict, List, Optional, Union

from src.common import *
from src.parse.model.binding import BindingModel
from src.parse.model.einsum import EinsumModel
from src.parse.model.format import TensorFormatModel
from src.parse.model.mapping import MappingModel
from src.parse.model.options import OptionsModel


def latency_discriminator(param: str, v: Any) -> str:
    
    if isinstance(v, int):
        return "int"

    if isinstance(v, dict):
        return "dict"

    raise MalformedInputError(
        "specified " + param,
        v,
        "should be of type `int` or `dict`")


def rename_field(field_name: str) -> str:
    
    if field_name == "format_":
        return "format"
    else:
        return field_name.replace("_", "-")


class CostMetricEnum(str, Enum):
    compute = "compute"
    memory = "memory"


class TemplateAttributesModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    cost_metrics: Optional[List[CostMetricEnum]] = None
    frequency: Optional[float] = None
    op_init_latency: Optional[Annotated[
        Union[Annotated[int, Tag("int")],
              Annotated[Dict[str, int], Tag("dict")]],
        Discriminator(lambda v: latency_discriminator("op-init-latency", v))]
    ] = None
    instruction_overhead: Optional[int] = None
    instruction_init_latency: Optional[int] = None
    instruction_latency: Optional[int] = None


class TemplateModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=rename_field,
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    name: str
    attributes: Optional[TemplateAttributesModel] = None
    einsum: EinsumModel
    mapping: MappingModel
    format_: Dict[str, TensorFormatModel]
    binding: BindingModel
    options: Dict[str, Any]
    code_gen: List[str]


class EngineModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    tag: str = Field(pattern=r'^!Engine$')
    name: str
    default_attributes: Optional[TemplateAttributesModel] = None
    templates: List[TemplateModel]
