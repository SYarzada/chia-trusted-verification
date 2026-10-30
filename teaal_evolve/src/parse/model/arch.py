

from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List


class ArchitectureModel(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    nodes: List[Dict[str, Any]]
