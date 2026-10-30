
from typing import Dict, List, Tuple

from src.ir import *
from src.parse import *
from src.trans.model.binder import Binder
from src.trans.model.bottleneck import BottleneckAnalysisModel
from src.trans.model.dependency import DependencyAwareModel
from src.trans.model.roof import ComputeRoofModel
from src.trans.teaal import *
from src.trans.yaml import *


class ModelMain:
    
    @staticmethod
    def get_bottleneck_latency(
        output_fn: str,
        shapes: Dict[str, int],
        strings: List[str] = [],
        files: List[str] = []
    ):
        
        yaml = YamlParser.parse_str(ModelMain.__read(strings, files))

        einsum_parser = EinsumParser.from_dict(yaml)
        mapping_parser = MappingParser.from_dict(yaml)
        format_parser = FormatParser.from_dict(yaml)
        binding_parser = BindingParser.from_dict(yaml)
        arch_parser = ArchitectureParser.from_dict(yaml)
        options_parser = OptionsParser.from_dict(yaml)

        elaborated_einsum = ElaboratedEinsum(
            einsum_parser, options_parser)
        elaborated_mapping = ElaboratedMapping(
            elaborated_einsum, mapping_parser)
        elaborated_format = ElaboratedFormat(
            elaborated_einsum, elaborated_mapping, format_parser)
        elaborated_arch = ElaboratedArchitecture(arch_parser)
        elaborated_binding = ElaboratedBinding(
            elaborated_einsum,
            elaborated_mapping,
            elaborated_format,
            binding_parser,
            elaborated_arch)

        model = BottleneckAnalysisModel(elaborated_binding, shapes)

        latency_info = model.get_latency()
        with open(output_fn, "w") as f:
            f.write(str(model.display_latency(latency_info)))

    @staticmethod
    def get_dependency_aware_trace(output_fn: str,
                                   shapes: Dict[str, int],
                                   strings: List[str] = [],
                                   files: List[str] = []):
        
        yaml = YamlParser.parse_str(ModelMain.__read(strings, files))

        einsum_parser = EinsumParser.from_dict(yaml)
        mapping_parser = MappingParser.from_dict(yaml)
        format_parser = FormatParser.from_dict(yaml)
        binding_parser = BindingParser.from_dict(yaml)
        arch_parser = ArchitectureParser.from_dict(yaml)
        options_parser = OptionsParser.from_dict(yaml)

        program = ElaboratedProgram(
            einsum_parser,
            mapping_parser,
            format_parser,
            binding_parser,
            arch_parser,
            options_parser)

        traces = DependencyAwareModel.get_traces(program, shapes)

        with open(output_fn, "w") as f:
            f.write(DependencyAwareModel.display_traces(traces))

    @staticmethod
    def get_roof(output_fn: str,
                 shapes: Dict[str, int],
                 strings: List[str] = [],
                 files: List[str] = []):
        
        yaml = YamlParser.parse_str(ModelMain.__read(strings, files))

        einsum_parser = EinsumParser.from_dict(yaml)
        options_parser = OptionsParser.from_dict(yaml)
        elaborated_einsum = ElaboratedEinsum(einsum_parser, options_parser)

        arch_parser = ArchitectureParser.from_dict(yaml)
        elaborated_arch = ElaboratedArchitecture(arch_parser)

        yaml = ModelMain.__display_cascade(elaborated_einsum, shapes).get()

        matches = Binder.find_all_matches(elaborated_einsum, elaborated_arch)
        roof = ComputeRoofModel.find_roof(
            elaborated_einsum, elaborated_arch, shapes, matches)

        matches_yaml = Binder.display_all_matches(
            elaborated_einsum, elaborated_arch, matches).get()
        roof_yaml = ComputeRoofModel.display_roof(
            elaborated_einsum, elaborated_arch, roof).get()

        yaml["binding"] = matches_yaml | roof_yaml

        with open(output_fn, "w") as f:
            f.write(str(YAMLGenerator(yaml)) + "\n")

    @staticmethod
    def match_all(
            output_fn: str,
            strings: List[str] = [],
            files: List[str] = []):
        
        yaml = YamlParser.parse_str(ModelMain.__read(strings, files))

        einsum_parser = EinsumParser.from_dict(yaml)
        options_parser = OptionsParser.from_dict(yaml)
        elaborated_einsum = ElaboratedEinsum(einsum_parser, options_parser)

        arch_parser = ArchitectureParser.from_dict(yaml)
        elaborated_arch = ElaboratedArchitecture(arch_parser)

        yaml = ModelMain.__display_cascade(elaborated_einsum, {}).get()

        matches = Binder.find_all_matches(elaborated_einsum, elaborated_arch)
        matches_yaml = Binder.display_all_matches(
            elaborated_einsum, elaborated_arch, matches).get()

        yaml["binding"] = matches_yaml

        with open(output_fn, "w") as f:
            f.write(str(YAMLGenerator(yaml)) + "\n")

    @staticmethod
    def __display_cascade(elaborated_einsum: ElaboratedEinsum,
                          shapes: Dict[str, int]) -> YAMLGenerator:
        
        cascade = []
        for decl_einsum in elaborated_einsum.get_einsums():
            cascade.extend(decl_einsum.split_for_solver())

        for rank_name, rank in elaborated_einsum.get_ranks().items():
            if rank_name not in shapes and isinstance(rank.get_shape(), int):
                shapes[rank_name] = rank.get_shape()

        yaml = TeAALGenerator.build_einsum(cascade, shapes).get()

        yaml["options"] = {"cascade-outputs": [output.get_name()
                                               for output in elaborated_einsum.get_cascade_outputs()]}

        return YAMLGenerator(yaml)

    @staticmethod
    def __read(strings: List[str], files: List[str]) -> str:
        
        str_yaml = "\n".join(strings) + "\n"
        for file in files:
            with open(file, "r") as f:
                str_yaml += f.read() + "\n"

        return str_yaml
