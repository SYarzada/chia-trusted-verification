

from lark import Lark, Token
from lark.exceptions import UnexpectedEOF
from lark.tree import Tree

from src.common import *


class CodeGeneration:
    
    grammar = """
        ?start: string? (template string?)* -> stmt

        ?string :  /[^<>]+/ -> string

        ?template: "<" NAME ("." NAME)* ">" -> template

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(code_gen: str) -> Tree:
        try:
            tree = CodeGeneration.parser.parse(code_gen)

            return tree

        except Exception:
            raise MalformedInputError(
                "code generation statement", code_gen, None)
