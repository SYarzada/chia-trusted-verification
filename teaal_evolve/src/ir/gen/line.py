

from lark import Token
from lark.tree import Tree
from typing import Any, cast, Dict, List, Optional, Tuple

from src.common import *
from src.ir.gen.base import BaseGeneration
from src.ir.gen.code import CodeGeneration
from src.ir.gen.template import TemplateGeneration


class LineGeneration(Base):
    def __init__(self, tree: Tree):
        self.tree = tree

        self.line: List[BaseGeneration] = []
        for child in self.tree.children:
            if child.data == "string":
                self.line.append(CodeGeneration(str(child.children[0])))

            elif child.data == "template":
                self.line.append(
                    TemplateGeneration(
                        cast(
                            List[Token],
                            child.children)))

            else:
                                                                            
                       
                raise NotImplementedError                    

    def copy_blank(self) -> "LineGeneration":
        
        return LineGeneration(self.tree)

    def get(self) -> List[BaseGeneration]:
        
        return self.line

    def update(self, type_: Optional[str], old: str, new: str):
        
        for i, piece in enumerate(self.line):
            if isinstance(piece, TemplateGeneration):
                if type_:
                    piece.update(type_, old, new)

                elif piece.get() == [old]:
                    self.line[i] = CodeGeneration(new)

    def key(self) -> Tuple[Any, ...]:
        return self.line,

    def __str__(self) -> str:
        return "".join(str(field) for field in self.line)
