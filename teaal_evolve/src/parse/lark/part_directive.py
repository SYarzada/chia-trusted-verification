

from lark import Lark
from lark.tree import Tree

from src.common import *


class PartitioningDirective:
    
    grammar = """
        ?start: "uniform_shape(" size ")" -> uniform_shape
              | "flatten()" -> flatten

        ?size: NUMBER -> int_sz
             | NAME -> str_sz

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    parser = Lark(grammar)

    @staticmethod
    def parse(info: str) -> Tree:
        try:
            tree = PartitioningDirective.parser.parse(info)
            return tree

        except Exception as e:
            raise MalformedInputError("partitioning directive", info, None)
