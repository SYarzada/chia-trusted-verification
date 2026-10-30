

from collections import Counter

from typing import Callable, List, Set, TypeVar

from src.common.error import *


T = TypeVar("T")


class ErrorUtils:
    

    @staticmethod
    def check_einsums(corr: Set[str], given_list: List[str], location: str):
        
        def duplicate_error(einsum: str):
            raise DuplicateEinsumError(einsum, location)

        def unknown_error(einsum: str):
            raise UnknownEinsumError(einsum, location)

        def missing_error(einsum: str):
            raise MissingEinsumError(einsum, location)

        ErrorUtils.__check_all(
            corr,
            given_list,
            duplicate_error,
            unknown_error,
            missing_error)

    @staticmethod
    def check_mem_ranks(corr: Set[str], given_list: List[str], location: str):
        
        def duplicate_error(rank: str):
            raise DuplicateMemoryRankError(rank, location)

        def unknown_error(rank: str):
            raise UnknownMemoryRankError(rank, location)

        def missing_error(rank: str):
            raise MissingMemoryRankError(rank, location)

        ErrorUtils.__check_all(
            corr,
            given_list,
            duplicate_error,
            unknown_error,
            missing_error)

    @staticmethod
    def check_one_fusion_group(
            groups: List[Tuple[str, ...]],
            einsums: List[str],
            location: str):
        
        def mismatch_error(expected: Tuple[str, ...], got: Tuple[str, ...]):
            raise UnfusedEinsumError(einsums, location)

        ErrorUtils.__check_one(groups, mismatch_error)

    @staticmethod
    def check_one_num_ranks(
            num_ranks: List[Basic],
            tensor: str,
            location: str):
        
        def mismatch_error(expected: int, got: int):
            raise RankNumberMismatchError(tensor, location, expected, got)

        ErrorUtils.__check_one(num_ranks, mismatch_error)

    @staticmethod
    def check_one_shape(shapes: List[Basic], between: str, location: str):
        
        def mismatch_error(expected: Basic, got: Basic):
            raise ShapeMismatchError(
                between, location, str(expected), str(got))

        ErrorUtils.__check_one(shapes, mismatch_error)

    @staticmethod
    def check_ranks(corr: Set[str], given_list: List[str], location: str):
        
        def duplicate_error(rank: str):
            raise DuplicateRankError(rank, location)

        def unknown_error(rank: str):
            raise UnknownRankError(rank, location)

        def missing_error(rank: str):
            raise MissingRankError(rank, location)

        ErrorUtils.__check_all(
            corr,
            given_list,
            duplicate_error,
            unknown_error,
            missing_error)

    @staticmethod
    def check_statements(corr: Set[str], given_list: List[str], location: str):
        
        def duplicate_error(statement: str):
            raise DuplicateStatementError(statement, location)

        def unknown_error(statement: str):
            raise UnknownStatementError(statement, location)

        def missing_error(statement: str):
            raise MissingStatementError(statement, location)

        ErrorUtils.__check_all(
            corr,
            given_list,
            duplicate_error,
            unknown_error,
            missing_error)

    @staticmethod
    def check_tensors(corr: Set[str], given_list: List[str], location: str):
        
        def duplicate_error(rank: str):
            raise DuplicateTensorError(rank, location)

        def unknown_error(tensor: str):
            raise UnknownTensorError(tensor, location)

        def missing_error(tensor: str):
            raise MissingTensorError(tensor, location)

        ErrorUtils.__check_all(
            corr, given_list, duplicate_error, unknown_error, missing_error)

    @staticmethod
    def __check_all(corr: Set[str],
                    given_list: List[str],
                    duplicate_error: Callable[[str],
                                              None],
                    unknown_error: Callable[[str],
                                            None],
                    missing_error: Callable[[str],
                                            None]):
        given = set(given_list)

        if len(given) < len(given_list):
            for rank, count in Counter(given_list).items():
                                                         
                if count > 1:
                    duplicate_error(rank)

        ErrorUtils.__check_unknown_missing(
            corr, given, unknown_error, missing_error)

    @staticmethod
    def __check_one(
            options_list: List[T], mismatch_error: Callable[[T, T], None]):

        options_set = set(options_list)

        if len(options_set) > 1:
            expected = options_list[0]
            options_set.remove(expected)

            mismatch_error(expected, options_set.pop())

    @staticmethod
    def __check_unknown_missing(corr: Set[str], given: Set[str], unknown_error: Callable[[
            str], None], missing_error: Callable[[str], None]):
        
                                      
        extra = given.difference(corr)
        if extra:
            unknown_error(extra.pop())

                                      
        missing = corr.difference(given)
        if missing:
            missing_error(missing.pop())
