

from lark import Lark, Token
from lark.exceptions import UnexpectedEOF
from lark.tree import Tree

from src.common import *


class Equation:
    
    grammar = """
        ?start: tensor "=" input (" . " input)* (";;" (map|reduce|(map reduce)))? -> einsum

        ?input: tensor
              | "<" NAME ">" -> template_input
              | INT -> int_input
              | FLOAT -> float_input

        ?tensor: NAME "[" accesses "]" -> tensor

        ?accesses: access? ("," access)* -> accesses

        ?access: term ("+" term)* -> access

        ?term: NAME
             | INT


        ?map: "M" op -> map

        ?reduce: "V" op -> reduce

        ?op: NAME -> simple_op
           | NAME "[" arg ("," arg)* "]" -> curried_op

        ?arg: NAME -> str_arg
            | INT -> int_arg
            | FLOAT -> float_arg
            | "<" NAME ">" -> template_arg

        %import common.CNAME -> NAME
        %import common.SIGNED_INT -> INT
        %import common.SIGNED_FLOAT -> FLOAT
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(equation: str) -> Tree:
        try:
            tree = Equation.parser.parse(equation)

            return tree

        except Exception:
            raise MalformedInputError("Einsum", equation, None)
