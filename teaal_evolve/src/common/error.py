
from sympy import Basic                
from typing import Any, List, Optional, Tuple

from src.common.global_state import GlobalState
from src.common.utils import Utils


class BitwidthMismatchError(Exception):
    def __init__(
            self,
            tensor: str,
            bitwidth: int,
            memory: str,
            bitwidths: List[int]):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Bitwidth mismatch error " +
            ("in the " + prefix + " " if prefix else "")
            + "for tensor " +
            tensor +
            " with bitwidth " +
            str(bitwidth) +
            ". Allowed bitwidths in " +
            memory +
            " are " +
            str(bitwidths))


class ComponentMismatchError(Exception):
    def __init__(
            self,
            location0: str,
            location1: str,
            expected: str,
            got: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Mismatched component" +
            (" in the " + prefix if prefix else "") +
            " between the " +
            location0 +
            " and the " +
            location1 +
            ". Expected " +
            expected +
            ", got " +
            got
        )


class CoordinateRangeMergeError(Exception):
    def __init__(self, location: str):
        super().__init__(
            "Unable to merge the coordinate ranges" +
            GlobalState.get_location_message(location))


class DatatypeMismatchError(Exception):
    def __init__(
            self,
            tensor: str,
            data_type: str,
            memory: str,
            data_types: List[str]):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Data type mismatch error " +
            ("in the " + prefix + " " if prefix else "")
            + "for tensor " +
            tensor +
            " with data type " +
            data_type +
            ". Allowed data types in " +
            memory +
            " are [" +
            ", ".join(data_types) +
            "]")


class DuplicateEinsumError(Exception):
    def __init__(self, einsum: str, location: str):
        super().__init__(
            "Duplicate Einsum " +
            einsum +
            GlobalState.get_location_message(location))


class DuplicateEinsumNameError(Exception):
    def __init__(self, name: str, loc1: int, loc2: int):
        super().__init__(
            "Duplicate Einsum name " +
            name +
            GlobalState.get_location_message("expressions") +
            ". Used at positions " +
            str(loc1) +
            " and " +
            str(loc2))


class DuplicateMemoryRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Duplicate memory rank " +
            rank +
            GlobalState.get_location_message(location))


class DuplicateRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Duplicate rank " +
            rank +
            GlobalState.get_location_message(location))


class DuplicateStatementError(Exception):
    def __init__(self, statement: str, location: str):
        super().__init__(
            "Duplicate statement " +
            statement +
            GlobalState.get_location_message(location))


class DuplicateTensorError(Exception):
    def __init__(self, tensor: str, location: str):
        super().__init__(
            "Duplicate tensor " +
            tensor +
            GlobalState.get_location_message(location))


class EinsumNumberMismatchError(Exception):
    def __init__(
            self,
            einsums: str,
            component: str,
            template: str,
            expected: int,
            got: int):
        super().__init__(
            "Mismatching number of Einsums bound to the " +
            component +
            "'s " +
            template +
            " template. Expected " +
            str(expected) +
            ", got " +
            str(got) +
            " (specifically Einsums " +
            einsums +
            ")")


class ExtraOperatorError(Exception):
    def __init__(self, op_type: str, location: str):
        super().__init__(
            "Extra " +
            op_type +
            " operator" +
            GlobalState.get_location_message(location))


class FusionRankMismatchError(Exception):
    def __init__(self, ranks: List[str], location: str, reason: str):
        super().__init__(
            "Ranks [" +
            ", ".join(ranks) +
            "] cannot be fused" +
            GlobalState.get_location_message(location) +
            ". " +
            reason)


class IllegalFieldError(Exception):
    def __init__(self, field: str, location: str):
        super().__init__(
            "Illegal field `" +
            field +
            "`" +
            GlobalState.get_location_message(location))


class IllegalOutputTensorError(Exception):
    def __init__(self, tensor: str, location: str):
        super().__init__(
            "Illegal output tensor " +
            tensor +
            GlobalState.get_location_message(location))


class IllegalPartitioningDirectiveError(Exception):
    def __init__(self, directive: str, ranks: Tuple[str, ...], location: str):
        super().__init__(
            "Illegal partitioning of " +
            ("rank " + ranks[0] if len(ranks) == 1 else "ranks (" + ", ".join(ranks) + ")") +
            " with directive `" +
            directive +
            "`" +
            GlobalState.get_location_message(location))


class IllegalPartitioningError(Exception):
    def __init__(self, ranks: Tuple[str, ...], location: str):
        super().__init__(
            "Illegal partitioning of " +
            ("rank " + ranks[0] if len(ranks) == 1 else "ranks (" + ", ".join(ranks) + ")") +
            GlobalState.get_location_message(location))


class IllegalSymbolicShapeError(Exception):
    def __init__(self, shape: Basic, location: str, message: str):
        super().__init__(
            "Symbolic shape " +
            str(shape) +
            " not allowed" +
            GlobalState.get_location_message(location) +
            ". " +
            message)


class IllegalTemplateError(Exception):
    def __init__(self, template: str, location: str):
        super().__init__(
            "Illegal template " +
            template +
            GlobalState.get_location_message(location))


class IllegalUnboundRanksError(Exception):
    def __init__(self, ranks: List[str], location: str):
        super().__init__(
            "Illegal unbound ranks [" +
            ", ".join(ranks) +
            "]" +
            GlobalState.get_location_message(location))


class ImperfectFactorizationError(Exception):
    def __init__(
            self,
            base_shape: int,
            lower_shape: int,
            einsum: str,
            rank: str):
        super().__init__(
            "Unable to divide " +
            str(base_shape) +
            " evenly into " +
            str(lower_shape) +
            GlobalState.get_location_message(einsum + " partitioning") +
            " for rank " +
            rank)


class InstructionMismatchError(Exception):
    def __init__(self, expected: str, got: str, location: str):
        super().__init__(
            "Mismatched instruction" +
            GlobalState.get_location_message(location) +
            ". Expected " +
            expected +
            ", got " +
            got
        )


class MalformedInputError(Exception):
    def __init__(self, type_: str, expr: Any, explanation: Optional[str]):
        super().__init__(
            "Malformed " +
            type_ +
            ": " +
            str(expr) +
            (". " + explanation if explanation else ""))


class MemoryRankBindingMismatchError(Exception):
    def __init__(
            self,
            rank0: str,
            tensor0: str,
            einsum0: str,
            location0: str,
            rank1: str,
            tensor1: str,
            einsum1: str,
            location1: str,
            expected: str,
            got: str):
        super().__init__(
            "Memory rank binding mismatch between the " +
            rank0 +
            " rank of tensor " +
            tensor0 +
            " in Einsum " +
            einsum0 +
            GlobalState.get_location_message(location0 + " tensor binding") +
            " and the " +
            rank1 +
            " rank of tensor " +
            tensor1 +
            " in Einsum " +
            einsum1 +
            GlobalState.get_location_message(location1 + " tensor binding") +
            ". Expected " +
            expected +
            ", got " +
            got)


class MemoryRankMismatchError(Exception):
    def __init__(
            self, memories: List[str], location: str):
        super().__init__(
            "Mismatched memory ranks for memories [" +
            ", ".join(memories) +
            "]" +
            GlobalState.get_location_message(location))


class MissingComponentError(Exception):
    def __init__(self, component: Optional[str], location: str):
        super().__init__(
            "Missing component" +
            (" " + component if component else "") +
            GlobalState.get_location_message(location))


class MissingEinsumError(Exception):
    def __init__(self, einsum: Optional[str], location: str):
        super().__init__(
            "Missing Einsum" +
            (" " + einsum if einsum else "") +
            GlobalState.get_location_message(location))


class MissingFieldError(Exception):
    def __init__(self, field: str, location: str):
        super().__init__(
            "Missing field `" +
            field +
            "`" +
            GlobalState.get_location_message(location))


class MissingFusionError(Exception):
    def __init__(self, einsums: str, ranks: List[str]):
        super().__init__(
            "Einsums " +
            einsums +
            " share an instruction but ranks [" +
            ", ".join(ranks) +
            "] are not fused" +
            GlobalState.get_location_message("fusion mapping"))


class MissingMemoryRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Missing memory rank " +
            rank +
            GlobalState.get_location_message(location))


class MissingOperatorError(Exception):
    def __init__(self, op_type: str, location: str):
        super().__init__(
            "Missing " +
            op_type +
            " operator" +
            GlobalState.get_location_message(location))


class MissingRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Missing rank " +
            rank +
            GlobalState.get_location_message(location))


class MissingStatementError(Exception):
    def __init__(self, statement: str, location: str):
        super().__init__(
            "Missing statement " +
            statement +
            GlobalState.get_location_message(location))


class MissingTensorError(Exception):
    def __init__(self, tensor: str, location: str):
        super().__init__(
            "Missing tensor " +
            tensor +
            GlobalState.get_location_message(location))


class MultipleAssignmentError(Exception):
    def __init__(self, tensor: str):
        super().__init__(
            "Tensor " +
            tensor +
            " is updated multiple times" +
            GlobalState.get_location_message("expressions"))


class MultipleEngineBindingError(Exception):
    def __init__(self):
        super().__init__(
            "Multiple engine bindings specified" +
            GlobalState.get_location_message("engine binding"))


class MultiplePartitioningError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Multiple partitionings on rank " +
            rank +
            GlobalState.get_location_message(location))


class MultiplePortsError(Exception):
    def __init__(self, ports: List[str], einsum: str, tensor: str):
        super().__init__(
            "Multiple potential ports [" +
            ", ".join(ports) +
            "] for accessing tensor " +
            tensor +
            GlobalState.get_location_message(
                "Einsum " +
                einsum +
                "'s tensor " +
                tensor +
                " binding") +
            ". Specify a port")


class MultipleUnconcretizedRankVariablesError(Exception):
    def __init__(self, rank_vars: List[str], tensor: str, einsum: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Multiple unconcretized rank variables [" +
            ", ".join(rank_vars) +
            "] in the access of tensor " +
            tensor +
            " when computing " +
             ("the " + prefix + "'s " if prefix else "") +
            "Einsum " +
            einsum)


class PartitioningMismatchError(Exception):
    def __init__(self, einsum: str, tensor: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__("Mismatched partitioning for " +
                         ("the " + prefix + "'s " if prefix else "") +
                         "Einsum " +
                         einsum +
                         " on tensor " +
                         tensor +
                         ". Tensor and loop ranks must use the same partitioning")


class RankNameCollisionError(Exception):
    def __init__(self, einsum: str, tensor: str, rank: str, pos: int):

        prefix = GlobalState.get_location_prefix()
        super().__init__("Rank " +
                         rank +
                         " used in " +
                         ("the " + prefix + "'s " if prefix else "") +
                         "Einsum " +
                         einsum +
                         " to define the iteration space and describe the " +
                         Utils.nth_as_string(pos) +
                         " rank of tensor " +
                         tensor +
                         " (using 0-based indexing)")


class RankNumberMismatchError(Exception):
    def __init__(
            self,
            tensor: str,
            location: str,
            expected: int,
            got: int):
        super().__init__(
            "Mismatching number of ranks for tensor " +
            tensor +
            GlobalState.get_location_message(location) +
            ". Expected " + str(expected) + ", got " + str(got))


class RankOrderMismatchError(Exception):
    def __init__(self, location0: str, location1: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__("Rank order mismatch " +
                         ("in the " + prefix + " " if prefix else "") +
                         "between the " +
                         location0 +
                         " specification and the " +
                         location1 +
                         " specification")


class RepeatedBindingError(Exception):
    def __init__(self, binding_type: str, msg: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Repeated " +
            binding_type +
            " binding for the " + msg +
            (" in the " + prefix if prefix else ""))


class RootRankMismatchError(Exception):
    def __init__(self, location0: str, location1: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__("Root rank mismatch " +
                         ("in the " + prefix + " " if prefix else "") +
                         "between the " +
                         location0 +
                         " specification and the " +
                         location1 +
                         " specification")


class ShapeMismatchError(Exception):
    def __init__(
            self,
            between: str,
            location: str,
            expected: str,
            got: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Mismatching shapes between " +
            between +
            GlobalState.get_location_message(location) +
            ". Expected shape " +
            expected +
            ", got shape " +
            got)


class SpacetimeMismatchError(Exception):
    def __init__(self, ranks: str, location: str):
        super().__init__(
            "Spacetime mismatch across ranks " +
            ranks +
            GlobalState.get_location_message(location))


class TemplateInConcreteError(Exception):
    def __init__(self, template: str, location: str):
        super().__init__(
            "Template <" +
            template +
            "> in the concrete " +
            location +
            " specification")


class TemplateMismatchError(Exception):
    def __init__(
            self,
            einsums: List[str],
            location: str,
            component: str,
            template: str,
            expected: Optional[str],
            got: Optional[str]):
        assert (expected is None) == (got is None)

        single = len(einsums) == 1
        if single:
            einsums_str = einsums[0]
        else:
            einsums_str = "[" + ", ".join(einsums) + "]"

        super().__init__(
            "Concrete " +
            ("Einsum " if single else "Einsums ") +
            einsums_str +
            "'s " +
            location +
            " specification does not match the " +
            location +
            " specification of the " +
            component +
            "'s " +
            template +
            " template" +
            (". Expected " + expected + ", got " + got
                if expected is not None and got is not None else ""))


class TensorNumberMismatchError(Exception):
    def __init__(
            self,
            einsum: str,
            component: str,
            instruction: str,
            location: str,
            expected: int,
            got: int):
        super().__init__(
            "Mismatching number of tensors for Einsum " +
            einsum +
            " bound to the " +
            component +
            "'s " +
            instruction +
            " template" +
            GlobalState.get_location_message(location) +
            ". Expected " + str(expected) + ", got " + str(got))


class UnboundRankNumberMismatchError(Exception):
    def __init__(
            self,
            einsums: str,
            got: List[int]):
        super().__init__(
            "Mismatching number of unbound ranks for Einsums " +
            einsums +
            GlobalState.get_location_message(einsums + " engine binding") +
            ". Got [" +
            ", ".join(str(num) for num in got) +
            "]")


class UnfusedEinsumError(Exception):
    def __init__(self, einsums: List[str], location: str):
        super().__init__(
            "Einsums [" +
            ", ".join(einsums) +
            "] must be fused together" +
            GlobalState.get_location_message(location))


class UnfusedRankError(Exception):
    def __init__(self, ranks: List[Tuple[str, str]], location: str):
        super().__init__(
            "Ranks [" +
            ", ".join(einsum + "." + rank for einsum, rank in ranks) +
            "] must be fused together" +
            GlobalState.get_location_message(location))


class UnknownComponentError(Exception):
    def __init__(self, component: str, location: str):
        super().__init__(
            "Unknown component " +
            component +
            GlobalState.get_location_message(location))


class UnknownEinsumError(Exception):
    def __init__(self, einsum: str, location: str):
        super().__init__(
            "Unknown Einsum " +
            einsum +
            GlobalState.get_location_message(location))


class UnknownInstructionError(Exception):
    def __init__(self, instruction: str, location: str):
        super().__init__(
            "Unknown instruction " +
            instruction +
            GlobalState.get_location_message(location))


class UnknownMemoryRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Unknown memory rank " +
            rank +
            GlobalState.get_location_message(location))


class UnknownPortError(Exception):
    def __init__(self, port: str, memory: str, engine: str, location: str):
        super().__init__(
            "Unknown port " +
            port +
            " for the " +
            memory +
            " memory, accessed by the " +
            engine +
            " engine," +
            GlobalState.get_location_message(location))


class UnknownRankError(Exception):
    def __init__(self, rank: str, location: str):
        super().__init__(
            "Unknown rank " +
            rank +
            GlobalState.get_location_message(location))


class UnknownStatementError(Exception):
    def __init__(self, statement: str, location: str):
        super().__init__(
            "Unknown statement " +
            statement +
            GlobalState.get_location_message(location))


class UnknownTemplateError(Exception):
    def __init__(self, template: str, location: str):
        super().__init__(
            "Unknown template " +
            template +
            GlobalState.get_location_message(location))


class UnknownTensorError(Exception):
    def __init__(self, tensor: str, location: str):
        super().__init__(
            "Unknown tensor " +
            tensor +
            GlobalState.get_location_message(location))


class UntaggedNodeError(Exception):
    def __init__(self, node: dict):
        super().__init__("Untagged architecture node: " + str(node))


class UnusedTensorError(Exception):
    def __init__(self, tensor: str):
        prefix = GlobalState.get_location_prefix()
        super().__init__(
            "Tensor " +
            tensor +
            (" in the " + prefix if prefix else "") +
            " is declared but not used")


class WriteAfterReadError(Exception):
    def __init__(self, tensor: str):
        super().__init__(
            "Tensor " +
            tensor +
            " is written to after being read" +
            GlobalState.get_location_message("expressions"))
