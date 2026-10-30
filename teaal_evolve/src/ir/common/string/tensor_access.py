
from typing import Any, Dict, Tuple

from src.ir.common.access import *
from src.ir.common.string.rank import StringRank
from src.ir.common.string.tensor import StringTensor


class StringTensorAccess(TensorAccess[StringRank, StringTensor]):
    

    def __init__(self,
                 tensor: StringTensor,
                 accesses: Dict[StringRank,
                                RankAccess[StringRank]]):
        self.tensor = tensor
        self.accesses = accesses

    def get_accesses(self) -> Dict[StringRank, RankAccess[StringRank]]:
        
        return self.accesses

    def get_tensor(self) -> StringTensor:
        
        return self.tensor

    def key(self) -> Tuple[Any, ...]:
        return self.tensor, self.accesses
