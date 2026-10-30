


from pydantic import AliasGenerator, BaseModel, ConfigDict
from typing import Dict, List, Optional


class TensorFormatModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name.replace("_", "-"),
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    rank_order: Dict[str, List[str]]
    bit_width: Optional[int] = None
    bit_widths: Optional[List[int]] = None
    data_type: Optional[str] = None
    data_types: Optional[List[str]] = None


class FormatModel(BaseModel):
    
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda field_name: field_name[:-1],
            serialization_alias=lambda field_name: field_name.title(),
        ),
        extra="forbid"
    )

    format_: Optional[Dict[str, TensorFormatModel]]
