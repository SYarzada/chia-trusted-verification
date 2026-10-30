
from typing import List

from src.ir import *
from src.parse import *
from src.trans.teaal.generator import TeAALGenerator
from src.trans.yaml import *


class TeAALMain:

    @staticmethod
    def build_implementation_yaml(input_yaml: str) -> str:
        
        einsum_parser = EinsumParser.from_str(input_yaml)
        options_parser = OptionsParser.from_str(input_yaml)

        elaborated_einsum = ElaboratedEinsum(einsum_parser, options_parser)

        mapping = TeAALGenerator.build_mapping(elaborated_einsum)
        format_ = TeAALGenerator.build_format(elaborated_einsum)
        binding = TeAALGenerator.build_binding(elaborated_einsum)

        return str(
            YAMLGenerator(
                mapping.get() | format_.get() | binding.get())) + "\n"

    @staticmethod
    def generate_from_einsum(
            output_fn: str,
            strings: List[str] = [],
            files: List[str] = []):
        
        for file in files:
            with open(file, "r") as f:
                strings.append(f.read())

        input_yaml = "\n".join(strings)

        output_yaml = TeAALMain.build_implementation_yaml(input_yaml)
        with open(output_fn, "w") as f:
            f.write(input_yaml + "\n" + output_yaml)
