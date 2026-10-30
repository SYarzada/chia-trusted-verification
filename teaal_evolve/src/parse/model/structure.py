

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field
from typing import Any, Dict, List


class StructureModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    tag: str = Field(pattern=r'^!(Hierarchical)|(Parallel)$')
    nodes: List[Dict[str, Any]]
