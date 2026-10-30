from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from chia.base.llm_call import QueryResult
from chia.models.vertex import (
    ContentBlockedError,
    MaxOutputTokensError,
    RateLimitError,
    VertexGeminiLLM,
)


_BLOCK_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    }
)


class TrustedVertexGeminiLLM(VertexGeminiLLM):
    """Run Vertex Gemini with required trusted tool calls.

    Use ANY while a tool call is required, then return to AUTO.
    """

    def __init__(
        self,
        *args: Any,
        require_first_tool_call: bool = True,
        required_tool_calls: int | None = None,
        thinking_level: str = "medium",
        first_tool_thinking_level: str = "low",
        first_tool_retry_thinking_level: str = "minimal",
        max_forced_tool_retries: int = 1,
        rate_limit_max_retries: int = 5,
        rate_limit_initial_backoff_s: float = 30.0,
        rate_limit_max_backoff_s: float = 480.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.require_first_tool_call = bool(require_first_tool_call)
        self.required_tool_calls = (
            None if required_tool_calls is None else max(0, int(required_tool_calls))
        )
        self.thinking_level = str(thinking_level)
        self.first_tool_thinking_level = str(first_tool_thinking_level)
        self.first_tool_retry_thinking_level = str(first_tool_retry_thinking_level)
        self.max_forced_tool_retries = max(0, int(max_forced_tool_retries))
        self.rate_limit_max_retries = max(0, int(rate_limit_max_retries))
        self.rate_limit_initial_backoff_s = max(
            0.0, float(rate_limit_initial_backoff_s)
        )
        self.rate_limit_max_backoff_s = max(
            self.rate_limit_initial_backoff_s,
            float(rate_limit_max_backoff_s),
        )

    async def _run_generate_async(self, user_message: str, tools=None) -> QueryResult:
        import asyncio
        from contextlib import AsyncExitStack

        from google import genai
        from google.genai import types
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            **self.client_kwargs,
        )

        stream_parts: list[str] = []
        stream_parts.append("=" * 80 + "\n")
        stream_parts.append(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"generate_content ({self.model})\n"
        )
        stream_parts.append("=" * 80 + "\n\n")
        truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
        stream_parts.append(f"[User Message]\n{truncated}\n\n")

        async with AsyncExitStack() as stack:
            function_decls = []
            dispatch: dict[str, tuple[Any, str]] = {}
            for tool in tools or []:
                port = getattr(tool, "port", 8000)
                url = f"http://{tool.hostname}:{port}/{tool.name}/mcp"
                transport = await stack.enter_async_context(streamable_http_client(url))
                read, write = transport[0], transport[1]
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                for fn in listed.tools:
                    api_name = f"{tool.name}__{fn.name}"[:64]
                    function_decls.append(
                        types.FunctionDeclaration(
                            name=api_name,
                            description=fn.description or " ",
                            parameters=self._sanitize_schema(
                                fn.inputSchema
                                or {"type": "object", "properties": {}}
                            ),
                        )
                    )
                    dispatch[api_name] = (session, fn.name)

            config_kwargs: dict[str, Any] = {
                "max_output_tokens": self.max_tokens,
                # Handle MCP tool calls in this loop.
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            }
            if self.system_message:
                config_kwargs["system_instruction"] = self.system_message
            if function_decls:
                config_kwargs["tools"] = [
                    types.Tool(function_declarations=function_decls)
                ]

            # Gemini 2.5 rejects thinking_level, so use its default.
            supports_thinking_level = not self.model.startswith("gemini-2.5")
            auto_kwargs = dict(config_kwargs)
            if supports_thinking_level:
                auto_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            auto_config = types.GenerateContentConfig(**auto_kwargs)
            forced_tool_config = auto_config
            first_tool_config = auto_config
            first_tool_retry_config = auto_config

            if function_decls and (
                self.require_first_tool_call
                or (
                    self.required_tool_calls is not None
                    and self.required_tool_calls > 0
                )
            ):
                forced_kwargs = dict(config_kwargs)
                forced_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[fn.name for fn in function_decls],
                    )
                )

                # Use normal thinking for later required tool calls.
                if supports_thinking_level:
                    forced_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.thinking_level
                    )
                forced_tool_config = types.GenerateContentConfig(**forced_kwargs)

                # Use lower thinking for the first required tool call.
                first_kwargs = dict(forced_kwargs)
                if supports_thinking_level:
                    first_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.first_tool_thinking_level
                    )
                first_tool_config = types.GenerateContentConfig(**first_kwargs)

                retry_kwargs = dict(first_kwargs)
                if supports_thinking_level:
                    retry_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.first_tool_retry_thinking_level
                    )
                first_tool_retry_config = types.GenerateContentConfig(**retry_kwargs)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)],
                )
            ]
            meta: dict[str, Any] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read": 0,
                "num_turns": 0,
                "tool_calls": 0,
                "forced_first_tool_call": bool(
                    self.require_first_tool_call and function_decls
                ),
            }
            final_text = ""
            tool_call_seen = False
            forced_tool_retry_count = 0

            for _ in range(self.max_tool_iterations):
                required_remaining = (
                    self.required_tool_calls is not None
                    and meta["tool_calls"] < self.required_tool_calls
                )

                if required_remaining and function_decls:
                    if self.require_first_tool_call and not tool_call_seen:
                        turn_config = (
                            first_tool_config
                            if forced_tool_retry_count == 0
                            else first_tool_retry_config
                        )
                    else:
                        turn_config = forced_tool_config
                elif (
                    self.require_first_tool_call
                    and function_decls
                    and not tool_call_seen
                ):
                    turn_config = (
                        first_tool_config
                        if forced_tool_retry_count == 0
                        else first_tool_retry_config
                    )
                else:
                    turn_config = auto_config
                resp = None
                for rate_limit_attempt in range(
                    self.rate_limit_max_retries + 1
                ):
                    try:
                        resp = await asyncio.to_thread(
                            client.models.generate_content,
                            model=self.model,
                            contents=contents,
                            config=turn_config,
                        )
                        break
                    except Exception as exc:
                        translated = self._translate_error(exc)

                        if (
                            isinstance(translated, RateLimitError)
                            and rate_limit_attempt
                            < self.rate_limit_max_retries
                        ):
                            delay_s = min(
                                self.rate_limit_initial_backoff_s
                                * (2 ** rate_limit_attempt),
                                self.rate_limit_max_backoff_s,
                            )
                            stream_parts.append(
                                "[Rate Limit] "
                                f"retry {rate_limit_attempt + 1}/"
                                f"{self.rate_limit_max_retries} "
                                f"after {delay_s:.0f}s\\n"
                            )
                            await asyncio.sleep(delay_s)
                            continue

                        if translated is not None:
                            raise translated from exc
                        raise

                if resp is None:
                    raise RuntimeError(
                        "Vertex generation ended without a response"
                    )

                meta["num_turns"] += 1
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    meta["input_tokens"] += (
                        getattr(usage, "prompt_token_count", 0) or 0
                    )
                    meta["output_tokens"] += (
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    meta["reasoning_tokens"] += (
                        getattr(usage, "thoughts_token_count", 0) or 0
                    )
                    meta["cache_read"] += (
                        getattr(usage, "cached_content_token_count", 0) or 0
                    )

                candidate = (resp.candidates or [None])[0]
                if candidate is None or candidate.content is None:
                    block = self._prompt_block_reason(resp)
                    if block:
                        raise ContentBlockedError(
                            node_id=self._get_node_id(),
                            block_reason=block,
                            raw_message=f"prompt blocked: {block}",
                        )
                    break

                finish = getattr(candidate, "finish_reason", None)
                finish_name = getattr(
                    finish, "name", str(finish) if finish else ""
                )
                turn_text_parts: list[str] = []
                function_calls = []
                for part in candidate.content.parts or []:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        function_calls.append(fc)
                        args_str = json.dumps(dict(fc.args or {}))
                        if len(args_str) > 2000:
                            args_str = args_str[:2000] + "\n... [truncated]"
                        stream_parts.append(
                            f"[Tool Call: {fc.name}]\nArgs: {args_str}\n\n"
                        )
                    elif getattr(part, "thought", False) and part.text:
                        stream_parts.append(f"[Thinking]\n{part.text}\n\n")
                    elif getattr(part, "text", None):
                        turn_text_parts.append(part.text)
                        stream_parts.append(f"[Response]\n{part.text}\n\n")

                if turn_text_parts:
                    final_text = "".join(turn_text_parts)

                if finish_name == "MAX_TOKENS":
                    if (
                        self.require_first_tool_call
                        and function_decls
                        and not tool_call_seen
                        and forced_tool_retry_count < self.max_forced_tool_retries
                    ):
                        forced_tool_retry_count += 1
                        final_text = ""
                        stream_parts.append(
                            "[DEBUG] Forced first tool turn hit MAX_TOKENS; "
                            f"retrying with thinking_level={self.first_tool_retry_thinking_level}\n\n"
                        )
                        # Keep truncated content out of the conversation history.
                        continue
                    raise MaxOutputTokensError(
                        node_id=self._get_node_id(),
                        raw_message="response truncated at max_output_tokens",
                        partial_text=final_text,
                    )

                contents.append(candidate.content)
                if finish_name in _BLOCK_FINISH_REASONS and not function_calls:
                    raise ContentBlockedError(
                        node_id=self._get_node_id(),
                        block_reason=finish_name,
                        raw_message=f"response blocked: {finish_name}",
                    )

                if not function_calls:
                    if (
                        self.required_tool_calls is not None
                        and meta["tool_calls"] < self.required_tool_calls
                    ):
                        raise RuntimeError(
                            "Gemini returned final text before exhausting the fixed "
                            f"trusted-verification budget: {meta['tool_calls']}/"
                            f"{self.required_tool_calls} tool calls"
                        )

                    if (
                        self.require_first_tool_call
                        and function_decls
                        and not tool_call_seen
                    ):
                        raise RuntimeError(
                            "Gemini returned no function call even though the first "
                            "trusted-verification call was required"
                        )
                    break

                tool_call_seen = True
                meta["tool_calls"] += len(function_calls)
                response_parts = []
                for fc in function_calls:
                    session, fn_name = dispatch.get(fc.name, (None, None))
                    if session is None:
                        result_text = f"Unknown tool: {fc.name}"
                        is_error = True
                    else:
                        try:
                            mcp_result = await session.call_tool(
                                fn_name, dict(fc.args or {})
                            )
                            result_text = self._mcp_result_to_text(mcp_result)
                            is_error = bool(getattr(mcp_result, "isError", False))
                        except Exception as exc:
                            result_text = f"Tool execution error: {exc}"
                            is_error = True

                    logged = result_text
                    if len(logged) > 2000:
                        logged = logged[:2000] + "\n... [truncated]"
                    label = "Tool Result (error)" if is_error else "Tool Result"
                    stream_parts.append(f"[{label}]\n{logged}\n\n")
                    response_payload = (
                        {"error": result_text}
                        if is_error
                        else {"result": result_text}
                    )
                    response_kwargs = {
                        "name": fc.name,
                        "response": response_payload,
                    }
                    call_id = getattr(fc, "id", None)
                    if call_id:
                        try:
                            response_part = types.Part.from_function_response(
                                id=call_id,
                                **response_kwargs,
                            )
                        except TypeError as exc:
                            # Older google-genai versions may reject id= here.
                            # Fall back to matching the tool response by name.
                            if "id" not in str(exc):
                                raise
                            response_part = types.Part.from_function_response(
                                **response_kwargs
                            )
                            stream_parts.append(
                                "[DEBUG] google-genai from_function_response() "
                                "does not accept id=; used name-matched "
                                "compatibility response instead.\n\n"
                            )
                    else:
                        response_part = types.Part.from_function_response(
                            **response_kwargs
                        )
                    response_parts.append(response_part)
                contents.append(types.Content(role="user", parts=response_parts))
            else:
                stream_parts.append(
                    f"[DEBUG] Reached max_tool_iterations={self.max_tool_iterations}\n"
                )

        self._last_metadata = {
            key: value
            for key, value in meta.items()
            if value not in (0, None, False, "")
        }
        stream_parts.append("-" * 80 + "\n\n")
        if self._log_prefix is not None:
            with open(f"{self._log_prefix}.log", "a", encoding="utf-8") as handle:
                handle.write("".join(stream_parts))

        result = QueryResult(
            result=final_text,
            returncode=0,
            stderr="",
            stream_result="".join(stream_parts),
        )
        # Keep token and tool-call metadata in the returned result.
        result.metadata = dict(self._last_metadata)
        return result


def build_vertex_gemini_agent(**kwargs: Any) -> TrustedVertexGeminiLLM:
    return TrustedVertexGeminiLLM(**kwargs)
