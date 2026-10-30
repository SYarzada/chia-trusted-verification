from __future__ import annotations

from chia_trusted_verification.config import base_search_engine

from .base import SearchStrategy
from .llm_agent import LLMAgentSearchStrategy
from .structured import StructuredRandomSearchStrategy, StructuredWalkSearchStrategy
from .skydiscover import AlphaEvolveSearchStrategy, OpenEvolveSearchStrategy


def build_search_strategy(engine: str) -> SearchStrategy:
    base = base_search_engine(engine)
    if base == "structured_random":
        return StructuredRandomSearchStrategy()
    if base == "structured_walk":
        return StructuredWalkSearchStrategy()
    if base == "openevolve":
        return OpenEvolveSearchStrategy(name=engine)
    if base == "alphaevolve":
        return AlphaEvolveSearchStrategy(name=engine)
    if base == "llm_agent":
        return LLMAgentSearchStrategy(name=engine)
    raise ValueError(f"unsupported search engine: {engine}")
