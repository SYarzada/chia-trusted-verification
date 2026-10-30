from __future__ import annotations

from chia_trusted_verification.config import base_search_engine, search_feedback_mode


def test_agentic_search_aliases():
    assert base_search_engine("openevolve_structured") == "openevolve"
    assert base_search_engine("alphaevolve_guided") == "alphaevolve"
    assert base_search_engine("llm_agent_structured") == "llm_agent"
    assert search_feedback_mode("openevolve_structured") == "score_only"
    assert search_feedback_mode("openevolve_guided") == "guided"
    assert search_feedback_mode("llm_agent_structured") == "score_only"
    assert search_feedback_mode("llm_agent_guided") == "guided"
