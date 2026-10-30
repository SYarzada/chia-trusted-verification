
from sympy import Float                
from typing import Any, Dict, List, Tuple


class YAMLGenerator:
    

    def __init__(self, obj: Any):
        self.obj = obj

    def get(self) -> Any:
        
        return self.obj

    def __str__(self) -> str:
        
        return YAMLGenerator.__make_str(self.obj, 0)[0]

    @staticmethod
    def __make_str(obj: Any, depth: int) -> Tuple[str, bool]:
        
        if isinstance(obj, dict):
            dlines = []
            for key, val in obj.items():
                assert isinstance(key, str)

                line = key + ":"
                string, is_multiline = YAMLGenerator.__make_str(val, depth + 1)

                if is_multiline:
                    line += "\n" + "  " * (depth + 1)

                else:
                    line += " "

                line += string

                dlines.append(line)

            return ("\n" + "  " * depth).join(dlines), True

        elif isinstance(obj, float) or isinstance(obj, Float):
            return str(obj), False

        elif isinstance(obj, int):
            return str(obj), False

        elif isinstance(obj, list):
            elems = [YAMLGenerator.__make_str(val, depth + 1) for val in obj]
            if not elems:
                return "[]", False

            llines, is_multilines = tuple(zip(*elems))

            joined = "[" + ", ".join(llines) + "]"
            if not any(is_multilines) and len(joined) < 80 - depth * 2:
                return joined, False

            return "- " + ("\n" + "  " * depth + "- ").join(llines), True

        elif isinstance(obj, str):
            return obj, False

        elif obj is None:
            return "", False

        else:
            raise ValueError("Unknown YAML object type: " + type(obj).__name__)
