

from typing import List

from src.python.base import Payload


class PTuple(Payload):
    

    def __init__(self, payloads: List[Payload]) -> None:
        self.payloads = payloads

    def gen(self, parens: bool) -> str:
        
        payload = ", ".join([p.gen(True) for p in self.payloads])
        if parens:
            return "(" + payload + ")"
        return payload


class PVar(Payload):
    

    def __init__(self, var: str) -> None:
        self.var = var

    def gen(self, parens: bool) -> str:
        
        return self.var
