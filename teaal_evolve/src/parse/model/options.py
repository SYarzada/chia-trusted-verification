


from pydantic import BaseModel
from typing import Any, Dict, Optional


class OptionsModel(BaseModel):
    
    options: Dict[str, Any]
