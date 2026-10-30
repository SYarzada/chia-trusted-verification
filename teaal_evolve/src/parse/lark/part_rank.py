

from lark import Lark
from lark.tree import Tree

from src.common import *


class PartitioningRank:
    
    grammar = """
        ?start: NAME -> str
              | "(" NAME ("," NAME)* ")" -> tuple

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    parser = Lark(grammar)

    @staticmethod
    def parse(info: str) -> Tree:
        try:
            tree = PartitioningRank.parser.parse(info)
            return tree

        except Exception as e:
            raise MalformedInputError("partitioning rank", info, None)
