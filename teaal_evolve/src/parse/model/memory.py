

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field
from typing import Dict, List, Optional


class MemoryRankModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    name: str


class PortModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    name: str
    engines: List[str]
    bandwidth: Optional[float] = None


class MemoryAttributesModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    ranks: List[MemoryRankModel]
    bit_widths: List[int]
    data_types: List[str]
    ports: Optional[List[PortModel]] = None


class MemoryModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    tag: str = Field(pattern=r'^!Memory$')
    name: str
    attributes: MemoryAttributesModel
    constructors: Dict[str, str]
    access_template: str
    reshape_template: str
