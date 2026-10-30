

from lark import Lark
from lark.tree import Tree

from src.common import *


class Statement:
    
    grammar = """
        ?start: NAME -> name
              | einsums ".instruction" -> instruction
              | NAME "." NAME -> rank

        ?einsums: NAME -> einsums
                | "(" NAME ",)" -> einsums
                | "(" NAME ("," NAME)+ ")" -> einsums

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    parser = Lark(grammar)

    @staticmethod
    def parse(info: str) -> Tree:
        try:
            tree = Statement.parser.parse(info)
            return tree

        except Exception as e:
            raise MalformedInputError("pipelining statement", info, None)
