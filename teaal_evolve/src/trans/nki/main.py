

from typing import List

from src.ir import *
from src.parse import *
from src.trans.nki.generator import NKIGenerator


class NKIMain:
    @staticmethod
    def check(strings: List[str] = [], files: List[str] = []):
        str_yaml = "\n".join(strings) + "\n"
        for file in files:
            with open(file, "r") as f:
                str_yaml += f.read() + "\n"

        yaml = YamlParser.parse_str(str_yaml)

        if "einsum" not in yaml.keys():
            print("`einsum` specification not found")
            return

        if "options" not in yaml.keys():
            print("`options` specification not found")
            return

        einsum_parser = EinsumParser.from_dict(yaml)
        options_parser = OptionsParser.from_dict(yaml)

        elaborated_einsum = ElaboratedEinsum(einsum_parser, options_parser)
        elaborated_einsum.check_concrete()

        print("`einsum` specification checks passed")

        if "mapping" not in yaml.keys():
            print("`mapping` specification not found")
            return

        mapping_parser = MappingParser.from_dict(yaml)
        elaborated_mapping = ElaboratedMapping(
            elaborated_einsum, mapping_parser)
        elaborated_mapping.check_concrete()

        print("`mapping` specification checks passed")

        if "format" not in yaml.keys():
            print("`format` specification not found")
            return

        format_parser = FormatParser.from_dict(yaml)
        elaborated_format = ElaboratedFormat(
            elaborated_einsum, elaborated_mapping, format_parser)
        elaborated_format.check_concrete()

        print("`format` specification checks passed")

        if "binding" not in yaml.keys():
            print("`binding` specification not found")
            return

        arch_parser = ArchitectureParser.from_dict(yaml)
        elaborated_arch = ElaboratedArchitecture(arch_parser)

        binding_parser = BindingParser.from_dict(yaml)
        elaborated_program = ElaboratedProgram(
            einsum_parser,
            mapping_parser,
            format_parser,
            binding_parser,
            arch_parser,
            options_parser)
        print("`binding` specification checks passed")

                                                                            
        generator = NKIGenerator(elaborated_program)
        print("`architecture` specification checks passed")

        print("Please proceed to compilation")

    @staticmethod
    def compile(strings: List[str] = [], files: List[str] = []) -> str:
        
        yaml = "\n".join(strings) + "\n"
        for file in files:
            with open(file, "r") as f:
                yaml += f.read() + "\n"

        yaml_dict = YamlParser.parse_str(yaml)
        einsum_parser = EinsumParser.from_dict(yaml_dict)
        mapping_parser = MappingParser.from_dict(yaml_dict)
        format_parser = FormatParser.from_dict(yaml_dict)
        binding_parser = BindingParser.from_dict(yaml_dict)
        arch_parser = ArchitectureParser.from_dict(yaml_dict)
        options_parser = OptionsParser.from_dict(yaml_dict)

        elaborated_program = ElaboratedProgram(
            einsum_parser,
            mapping_parser,
            format_parser,
            binding_parser,
            arch_parser,
            options_parser)

        generator = NKIGenerator(elaborated_program)

        return str(generator)
