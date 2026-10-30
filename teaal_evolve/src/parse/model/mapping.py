


from pydantic import AliasGenerator, BaseModel, ConfigDict, Discriminator, Tag
from typing import Annotated, Any, Dict, List, Optional, Union

from src.common import *


def fused_rank_discriminator(v: Any) -> str:
    
    if isinstance(v, list):
        return "list"

    if isinstance(v, dict):
        if len(v) == 1:
            return "dict"
        else:
            raise MalformedInputError(
                "fused ranks",
                v,
                "each list entry should have exactly one named fused loop rank")

    raise MalformedInputError(
        "fused ranks",
        v,
        "should be of type `list` or `dict`")


class SpacetimeModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    space: List[str]
    time: List[str]


class FusionModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    einsums: List[str]
    ranks: List[Annotated[Union[Annotated[List[str], Tag("list")],
                                Annotated[Dict[str, List[str]], Tag("dict")]],
                          Discriminator(fused_rank_discriminator)]]


class MappingModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    partitioning: Optional[Dict[str, Dict[str, List[str]]]] = {}
    loop_order: Dict[str, List[str]]
    spacetime: Dict[str, SpacetimeModel]
    fusion: Optional[List[FusionModel]] = []
