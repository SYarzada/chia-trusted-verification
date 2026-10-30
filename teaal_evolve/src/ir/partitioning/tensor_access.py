

from typing import Any, List, Dict, Tuple

from src.common import *
from src.ir.common import *
from src.ir.cond import *
from src.ir.einsum import *
from src.ir.partitioning.partitioning import Partitioning
from src.ir.partitioning.rank import PartitionedRank
from src.ir.partitioning.tensor import PartitionedTensor


class PartitionedTensorAccess(
        TensorAccess[PartitionedRank, PartitionedTensor]):
    

    def __init__(self,
                 decl_tensor_access: DeclaredTensorAccess,
                 partitioning: Partitioning):

        self.decl_tensor_access = decl_tensor_access
        self.conds: List[Condition] = []

        decl_tensor = self.decl_tensor_access.get_tensor()
        einsum = self.decl_tensor_access.get_einsum()

        self.tensor = PartitionedTensor(decl_tensor, partitioning)

        decl_accesses = self.decl_tensor_access.get_accesses()
        old_conds = len(partitioning.get_conds())
        self.accesses = partitioning.apply_to_accesses(self.decl_tensor_access)

        for cond in partitioning.get_conds()[old_conds:]:
            if cond not in self.conds:
                self.conds.append(cond)

    def get_accesses(self) -> Dict[PartitionedRank,
                                   RankAccess[PartitionedRank]]:
        
        return self.accesses

    def get_conds(self) -> List[Condition]:
        
        return self.conds

    def get_decl_tensor_access(self) -> DeclaredTensorAccess:
        
        return self.decl_tensor_access

    def get_einsum(self) -> str:
        
        return self.decl_tensor_access.get_einsum()

    def get_tensor(self) -> PartitionedTensor:
        
        return self.tensor

    def key(self) -> Tuple[Any, ...]:
        return self.decl_tensor_access, self.tensor, self.accesses
