
from typing import Optional
from sympy import Basic                

from src.common import *
from src.ir.cond.cond import Condition
from src.ir.cond.equal import EqualCondition
from src.ir.cond.leq import LessThanEqualCondition


class ConditionUtils:
    
    def shape_leq_or_cond(
            shape0: Basic,
            shape1: Basic,
            between: str,
            location: str) -> Optional[Condition]:
        
        diff = shape0 - shape1
        if not isinstance(diff, int) and diff.free_symbols:
            return LessThanEqualCondition(shape0, shape1)

        if diff > 0:
            raise ShapeMismatchError(
                between, location, ">= " + str(shape0), str(shape1))

        return None

    def shape_equal_or_cond(
            shape0: Basic,
            shape1: Basic,
            between: str,
            location: str) -> Optional[Condition]:
        
        diff = shape0 - shape1
        if not isinstance(diff, int) and diff.free_symbols:
            return EqualCondition(shape0, shape1)

        if diff != 0:
            raise ShapeMismatchError(
                between, location, str(shape0), str(shape1))

        return None
