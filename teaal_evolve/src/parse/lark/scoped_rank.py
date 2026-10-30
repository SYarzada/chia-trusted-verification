

from lark import Lark
from lark.tree import Tree

from src.common import *


class ScopedRank:
    
    grammar = """
        ?start: NAME "." NAME -> rank

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    parser = Lark(grammar)

    @staticmethod
    def parse(info: str) -> Tree:
        try:
            tree = ScopedRank.parser.parse(info)
            return tree

        except Exception as e:
            raise MalformedInputError("scoped rank", info, None)
