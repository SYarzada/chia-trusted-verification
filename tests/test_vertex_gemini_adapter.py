from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chia_trusted_verification"
    / "search"
    / "vertex_gemini.py"
)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "search" / "llm_agent.yaml"


def test_vertex_adapter_forces_only_first_tool_turn():
    text = ADAPTER.read_text(encoding="utf-8")
    assert 'mode="ANY"' in text
    assert 'allowed_function_names=[fn.name for fn in function_decls]' in text
    assert 'and not tool_call_seen' in text
    assert 'turn_config = auto_config' in text
    assert 'tool_call_seen = True' in text
    assert 'thinking_level=self.first_tool_thinking_level' in text
    assert 'thinking_level=self.first_tool_retry_thinking_level' in text
    assert 'supports_thinking_level = not self.model.startswith("gemini-2.5")' in text
    assert 'forced_tool_retry_count < self.max_forced_tool_retries' in text
    assert 'call_id = getattr(fc, "id", None)' in text
    assert 'except TypeError as exc:' in text
    assert 'used name-matched' in text


def test_llm_agent_defaults_to_direct_vertex_with_forced_first_call():
    text = CONFIG.read_text(encoding="utf-8")
    assert "provider: vertex_gemini" in text
    assert "require_first_tool_call: true" in text
    assert "max_tokens: 65536" in text
    assert "thinking_level: medium" in text
    assert "first_tool_thinking_level: low" in text
    assert "first_tool_retry_thinking_level: minimal" in text
    assert "max_forced_tool_retries: 1" in text
