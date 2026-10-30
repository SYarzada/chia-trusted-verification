from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.chia.runtime import get, require_chia_tool
from chia_trusted_verification.chia.tool import TrustedVerificationTool
from chia_trusted_verification.config import ExperimentConfig, search_feedback_mode
from chia_trusted_verification.types import SearchProblem, SearchResult

from .base import SearchStrategy


class LLMAgentSearchStrategy(SearchStrategy):
    def __init__(self, *, name: str = "llm_agent") -> None:
        self.name = name

    def run(
        self,
        *,
        problem: SearchProblem,
        initial_program: Path,
        evaluator: TrustedVerificationEvaluator,
        config: ExperimentConfig,
    ) -> SearchResult:
        require_chia_tool()
        settings = self._settings(config.search_config)
        agent_settings = dict(settings.get("agent", {}))
        tool_settings = dict(settings.get("tool", {}))
        max_evaluations = int(agent_settings.get("max_evaluations", config.iterations))
        tool_resources = {
            str(k): float(v) for k, v in dict(tool_settings.get("resources", {})).items()
        }
        task_options = {"resources": tool_resources} if tool_resources else None
        self._prepare_shared_result_log(evaluator.result_log)
        tool = self._build_tool(
            problem=problem,
            config=config,
            evaluator=evaluator,
            max_evaluations=max_evaluations,
            task_options=task_options,
            feedback_mode=search_feedback_mode(config.engine),
        )
        agent = self._build_agent(problem, config, agent_settings)
        prompt = self._agent_prompt(
            problem, max_evaluations, feedback_mode=search_feedback_mode(config.engine)
        )
        try:
            response = get(agent.prompt.chia_remote(agent, prompt, tools=[tool]))
        finally:
            tool.stop()

        if bool(agent_settings.get("require_exact_evaluations", False)):
            proposal_count = self._proposal_count(evaluator.result_log)
            if proposal_count != max_evaluations:
                raise RuntimeError(
                    "fixed-budget LLM agent evaluated "
                    f"{proposal_count} proposals; expected exactly {max_evaluations}"
                )

        raw = getattr(response, "result", response)
        response_metadata = _response_metadata(response)

        # Select the best candidate from trusted verification history.
        # The model's final response may be worse than an earlier verified candidate.
        trusted_best = self._best_trusted_proposal(evaluator.result_log)

        if trusted_best is not None:
            (
                best_source,
                trusted_best_score,
                trusted_best_sequence,
                trusted_best_status,
            ) = trusted_best
            selected_from_trusted_history = True
        else:
            selected_from_trusted_history = False
            trusted_best_score = None
            trusted_best_sequence = None
            trusted_best_status = None

            if raw is None or not str(raw).strip():
                best_source = initial_program.read_text(encoding="utf-8")
            else:
                best_source = _extract_source(str(raw))

        output = Path(config.output_dir) / "llm_agent"
        output.mkdir(parents=True, exist_ok=True)
        best_path = output / f"best_candidate{Path(problem.file_name).suffix or '.txt'}"
        best_path.write_text(best_source, encoding="utf-8")
        final_result = evaluator.evaluate_source(best_source, role="selected_final")
        return SearchResult(
            engine=getattr(self, "name", config.engine),
            best_score=float(final_result.metrics.get("combined_score", 0.0)),
            best_solution=best_source,
            metrics={
                "tool_evaluation_budget": max_evaluations,
                "final_status": final_result.status,
                "final_metrics": final_result.metrics,
                "selected_from_trusted_history": selected_from_trusted_history,
                "trusted_history_best_score": trusted_best_score,
                "trusted_history_best_sequence": trusted_best_sequence,
                "trusted_history_best_status": trusted_best_status,
                "agent_response_metadata": response_metadata,
            },
            output_dir=str(output),
        )

    def _best_trusted_proposal(
        self,
        result_log: str | Path,
    ) -> tuple[str, float, int, str] | None:
        """Return the best candidate from trusted verification results.

        Passing candidates rank above failures. Within each group, higher
        scores rank first. Ties keep the earliest candidate.
        """
        path = Path(result_log)
        if not path.exists():
            return None

        best_key: tuple[int, float, int] | None = None
        best_value: tuple[str, float, int, str] | None = None

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("role") != "proposal":
                continue

            source_ref = record.get("candidate_source_file")
            if not source_ref:
                continue

            source_path = Path(str(source_ref))
            if not source_path.is_absolute():
                source_path = (path.parent / source_path).resolve()

            if not source_path.is_file():
                continue

            metrics = record.get("metrics") or {}
            raw_score = metrics.get("combined_score", metrics.get("score", 0.0))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0

            status = str(record.get("status", "fail"))
            try:
                sequence = int(record.get("evaluation_sequence") or 0)
            except (TypeError, ValueError):
                sequence = 0

            key = (
                1 if status == "pass" else 0,
                score,
                -sequence,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_value = (
                    source_path.read_text(encoding="utf-8"),
                    score,
                    sequence,
                    status,
                )

        return best_value

    def _proposal_count(self, result_log: str | Path) -> int:
        path = Path(result_log)
        if not path.exists():
            return 0

        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("role") == "proposal":
                count += 1
        return count

    def _prepare_shared_result_log(self, result_log: str | Path) -> None:
        """Allow the Docker worker to append to shared search logs.

        The host may create these files before the worker starts.
        Update permissions only for the LLM-agent run.
        """
        target = Path(result_log).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.parent.chmod(0o777)
        except OSError:
            pass
        candidate_dir = target.parent / "candidate_programs"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            candidate_dir.chmod(0o777)
        except OSError:
            pass
        for shared in (
            target,
            target.with_suffix(target.suffix + ".lock"),
            target.with_suffix(target.suffix + ".seq"),
        ):
            if shared.exists():
                try:
                    shared.chmod(0o666)
                except OSError:
                    pass

    def _settings(self, path: str | Path) -> dict[str, Any]:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise TypeError("LLM-agent search configuration must be a YAML mapping")
        if not isinstance(raw.get("agent", {}), dict):
            raise TypeError("agent search configuration must be a mapping")
        if not isinstance(raw.get("tool", {}), dict):
            raise TypeError("tool configuration must be a mapping")
        return dict(raw)

    def _build_tool(
        self,
        *,
        problem: SearchProblem,
        config: ExperimentConfig,
        evaluator: TrustedVerificationEvaluator,
        max_evaluations: int,
        task_options: dict[str, Any] | None,
        feedback_mode: str,
    ) -> TrustedVerificationTool:
        return TrustedVerificationTool(
            name="trusted_verification",
            backend_name=config.backend_name,
            backend_settings=config.backend_settings,
            backend_resources=config.backend_resources,
            intent=config.intent.to_dict(),
            initial_source=problem.initial_source,
            max_evaluations=max_evaluations,
            result_log=str(evaluator.result_log),
            run_metadata=evaluator.run_metadata,
            feedback_mode=feedback_mode,
            task_options=task_options,
        )

    def _build_agent(
        self,
        problem: SearchProblem,
        config: ExperimentConfig,
        settings: dict[str, Any],
    ) -> Any:
        provider = str(settings.get("provider", "opencode_vertex")).strip().lower()
        model = str(config.model or settings.get("model") or "").strip()
        timeout_seconds = int(settings.get("timeout_seconds", 600))
        retries = int(settings.get("retries", 3))
        system_message = problem.guidance.system_message
        if provider == "opencode_vertex":
            from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

            project = self._google_project(settings)
            location = str(settings.get("location", "global"))
            if not model:
                raise ValueError("opencode_vertex requires a Gemini model")
            if not model.startswith("google-vertex/"):
                model = f"google-vertex/{model}"
            model_id = model.split("/", 1)[1]
            vertex = AdditionalModelProvider(
                id="google-vertex",
                npm="@ai-sdk/google-vertex",
                name="Google Vertex AI",
                models=[model_id],
                options={"project": project, "location": location},
            )
            return OpenCodeLLM(
                model=model,
                system_message=system_message,
                timeout_seconds=timeout_seconds,
                retries=retries,
                additional_providers=[vertex],
            )
        if provider == "vertex_gemini":
            from .vertex_gemini import build_vertex_gemini_agent

            project = self._google_project(settings)
            location = str(settings.get("location", "global"))
            if not model:
                raise ValueError("vertex_gemini requires a Gemini model")
            model = model.removeprefix("google-vertex/")
            return build_vertex_gemini_agent(
                model=model,
                system_message=system_message,
                project=project,
                location=location,
                timeout_seconds=timeout_seconds,
                retries=retries,
                max_tokens=int(settings.get("max_tokens", 65536)),
                max_tool_iterations=int(
                    settings.get(
                        "max_tool_iterations",
                        int(settings.get("max_evaluations", config.iterations)) + 5,
                    )
                ),
                require_first_tool_call=bool(settings.get("require_first_tool_call", True)),
                thinking_level=str(settings.get("thinking_level", "medium")),
                first_tool_thinking_level=str(
                    settings.get("first_tool_thinking_level", "low")
                ),
                first_tool_retry_thinking_level=str(
                    settings.get("first_tool_retry_thinking_level", "minimal")
                ),
                max_forced_tool_retries=int(
                    settings.get("max_forced_tool_retries", 1)
                ),
                required_tool_calls=(
                    int(settings.get("max_evaluations", config.iterations))
                    if bool(settings.get("require_exact_evaluations", False))
                    else None
                ),
            )
        if provider == "antigravity":
            from chia.models.antigravity import AntigravityLLM

            return AntigravityLLM(
                model=model or None,
                system_message=system_message,
                timeout_seconds=timeout_seconds,
                retries=retries,
                resume_session=True,
            )
        if provider == "opencode":
            from chia.models.opencode import OpenCodeLLM

            return OpenCodeLLM(
                model=model or None,
                system_message=system_message,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
        raise ValueError(
            "unsupported CHIA agent provider; expected one of "
            "opencode_vertex, vertex_gemini, antigravity, opencode"
        )

    def _google_project(self, settings: dict[str, Any]) -> str:
        direct = settings.get("project")
        if direct:
            return str(direct)
        env_name = str(settings.get("project_env", "GOOGLE_CLOUD_PROJECT"))
        project = os.environ.get(env_name)
        if not project:
            raise RuntimeError(
                f"Google Cloud project is not configured; set agent.project or {env_name}"
            )
        return project

    def _agent_prompt(
        self, problem: SearchProblem, max_evaluations: int, *, feedback_mode: str = "guided"
    ) -> str:
        feedback_instruction = (
            "Use the returned status, metrics, evidence, diagnostics, and structured "
            "feedback to choose or repair the next candidate."
            if feedback_mode == "guided"
            else "Use only the returned trusted status and scalar metrics to choose the next "
            "candidate. Localized failed-obligation diagnostics and repair feedback are "
            "intentionally withheld in this structured-search baseline."
        )
        pieces = [
            "You control the candidate-search loop.",
            (
                "You MUST use the trusted_verification_evaluate_candidate tool before "
                "selecting a final candidate. Evaluate at least one non-seed proposal. "
                "Every candidate you consider for final selection must first be submitted "
                "to trusted verification."
            ),
            (
                "The trusted_verification_evaluate_candidate tool always runs prepare, "
                "verify, evaluate, and feedback through the configured trusted backend "
                "before returning a result."
            ),
            (
                f"You MUST invoke trusted verification exactly {max_evaluations} times. "
                "Call the tool once per search step. Do not return a final candidate until "
                "the tool reports remaining_evaluations == 0. "
                + feedback_instruction
            ),
            (
                "Do not assume a candidate is acceptable without trusted verification. "
                "Do not return a final candidate that has not received a trusted-verification "
                "result. When you finish searching, return only the complete candidate source "
                "you want selected as the final result."
            ),
        ]
        if problem.guidance.problem_description:
            pieces.append("Problem:\n" + problem.guidance.problem_description)
        pieces.append("Initial candidate:\n```\n" + problem.initial_source + "\n```")
        return "\n\n".join(pieces)


def _extract_source(response: str) -> str:
    text = response.strip()
    first_fence = text.find("```")
    if first_fence < 0:
        return text + ("\n" if not text.endswith("\n") else "")
    first_newline = text.find("\n", first_fence)
    last_fence = text.rfind("```")
    if first_newline < 0 or last_fence <= first_newline:
        return text + ("\n" if not text.endswith("\n") else "")
    body = text[first_newline + 1:last_fence].strip("\n")
    return body + "\n"


def _response_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("usage", "metrics", "metadata", "cost", "token_usage"):
        value = getattr(response, name, None)
        if value is None:
            continue
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                pass
        elif hasattr(value, "__dict__"):
            try:
                value = vars(value)
            except Exception:
                pass
        if isinstance(value, (dict, list, tuple, str, int, float, bool)):
            metadata[name] = value
    return metadata
