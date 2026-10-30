
from typing import Any, Dict, List, Tuple

from src.ir import *
from src.trans.yaml import *


class TeAALGenerator:
    @staticmethod
    def build_binding(elaborated_einsum: ElaboratedEinsum) -> YAMLGenerator:
        
        einsums = [decl_einsum.get_name()
                   for decl_einsum in elaborated_einsum.get_einsums()]
        tensors, uses = TeAALGenerator.__get_tensor_uses(elaborated_einsum)
        inputs = {decl_tensor.get_name()
                  for decl_tensor in elaborated_einsum.get_cascade_inputs()}

        engines = []
        for einsum in einsums:
            engines.append({"einsum": einsum,
                            "ranks": ["FILL_ME"],
                            "component": "FILL_ME",
                            "instruction": "FILL_ME"})

        tensor_info: List[Dict[str, Any]] = []
        for tensor in tensors:
            for i, einsum in enumerate(uses[tensor]):
                tensor_info.append(
                    {"name": tensor, "einsum": einsum, "component": {"name": "FILL_ME"}})

                if i == 0 and tensor not in inputs:
                    tensor_info[-1]["constructor"] = "FILL_ME"

                tensor_info[-1]["ranks"] = {"unbound": ["FILL_ME"],
                                            "FILL_ME": "FILL_ME"}

        spec = {"engine": engines, "tensor": tensor_info}

        return YAMLGenerator({"binding": spec})

    @staticmethod
    def build_einsum(cascade: List[Einsum],
                     shapes: Dict[str, int]) -> YAMLGenerator:
        
        declaration: Dict[str, List[str]] = {}
        expressions: List[Any] = []
        for einsum in cascade:
            for tensor in einsum.get_reuse_spec().get_tensors():
                if tensor.get_name() in declaration.keys():
                    continue

                declaration[tensor.get_name()] = [rank.get_name()
                                                  for rank in tensor.get_ranks()]

            if einsum.get_name() == einsum.get_reuse_spec(
            ).get_output().get().get_tensor().get_name():
                expressions.append(einsum.gen())

            else:
                expressions.append({einsum.get_name(): einsum.gen()})

        spec = {"declaration": declaration, "expressions": expressions}

        if shapes:
            spec["shape"] = shapes

        return YAMLGenerator({"einsum": spec})

    @staticmethod
    def build_format(elaborated_einsum: ElaboratedEinsum) -> YAMLGenerator:
        
        tensors, uses = TeAALGenerator.__get_tensor_uses(elaborated_einsum)

        spec = {}
        for tensor in tensors:
            rank_orders = {}
            for einsum in uses[tensor]:
                rank_orders[einsum] = ["FILL_ME"]

            format_ = {
                "rank-order": rank_orders,
                "bit-width": "FILL_ME",
                "data-type": "FILL_ME"}

            spec[tensor] = format_

        return YAMLGenerator({"format": spec})

    @staticmethod
    def build_mapping(elaborated_einsum: ElaboratedEinsum) -> YAMLGenerator:
        
        einsums = [decl_einsum.get_name()
                   for decl_einsum in elaborated_einsum.get_einsums()]

        partitioning = {}
        loop_order = {}
        spacetime = {}
        for einsum in einsums:
            partitioning[einsum] = {"FILL_ME": ["FILL_ME"]}
            loop_order[einsum] = ["FILL_ME"]
            spacetime[einsum] = {"space": ["FILL_ME"], "time": ["FILL_ME"]}

        fusion = [{"einsums": einsums, "ranks": [["FILL_ME"]]}]

        spec = {
            "partitioning": partitioning,
            "loop-order": loop_order,
            "spacetime": spacetime,
            "fusion": fusion}

        return YAMLGenerator({"mapping": spec})

    @staticmethod
    def __get_tensor_uses(
            elaborated_einsum: ElaboratedEinsum) -> Tuple[List[str], Dict[str, List[str]]]:
        
        tensors = []
        uses: Dict[str, List[str]] = {}
        for decl_einsum in elaborated_einsum.get_einsums():
            for decl_tensor in decl_einsum.get_reuse_spec().get_tensors():
                tensor = decl_tensor.get_name()

                if tensor not in tensors:
                    tensors.append(tensor)
                    uses[tensor] = []

                uses[tensor].append(decl_einsum.get_name())

        return tensors, uses
