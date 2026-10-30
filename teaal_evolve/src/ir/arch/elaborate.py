

from typing import Any, cast, Dict, Tuple

from src.common import *
from src.ir.common import *
from src.ir.engine import *
from src.ir.memory import *
from src.ir.solver import *
from src.ir.template import *
from src.parse import *


class ElaboratedArchitecture(Base):
    

    def __init__(self, arch_parser: ArchitectureParser):
        self.memories = {}
        self.engines = {}
        for name, comp in arch_parser.get_components().items():
            if isinstance(comp, MemoryParser):
                self.memories[name] = MemoryComponent(comp)

        for name, comp in arch_parser.get_components().items():
            if isinstance(comp, EngineParser):
                self.engines[name] = EngineComponent(comp, self.memories)

                                                           

    def get_memories(self) -> Dict[str, MemoryComponent]:
        
        return self.memories

    def get_engines(self) -> Dict[str, EngineComponent]:
        
        return self.engines

    def maybe_get_template(
            self,
            engine: str,
            instruction: str,
            location: str) -> ElaboratedTemplate:
        
                                                
        if engine not in self.engines.keys():
            raise UnknownComponentError(engine, location)

        return self.engines[engine].maybe_get_template(instruction, location)

    def key(self) -> Tuple[Any, ...]:
        return self.engines, self.memories
