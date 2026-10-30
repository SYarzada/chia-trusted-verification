from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from chia_trusted_verification.results import (
    append_jsonl_sequenced,
    candidate_record,
    persist_candidate_source,
)
from chia_trusted_verification.types import PipelineResult, VerificationIntent

from .nodes import evaluate_artifact, feedback_for_search, prepare_artifact, verify_artifact
from .runtime import ChiaTool


class TrustedVerificationTool(ChiaTool):
    def setup(
        self,
        backend_name: str,
        backend_settings: Mapping[str, Any],
        backend_resources: Mapping[str, Mapping[str, float]],
        intent: Mapping[str, Any],
        initial_source: str,
        max_evaluations: int,
        result_log: str | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        feedback_mode: str = "guided",
    ) -> None:
        self.backend_name = str(backend_name)
        self.backend_settings = dict(backend_settings)
        self.backend_resources = {
            str(stage): {str(k): float(v) for k, v in resources.items()}
            for stage, resources in backend_resources.items()
        }
        self.intent = VerificationIntent.from_dict(intent)
        self.initial_source = initial_source
        self.max_evaluations = int(max_evaluations)
        self.evaluations = 0
        self.result_log = None if result_log is None else Path(result_log).resolve()
        self.run_metadata = dict(run_metadata or {})
        self.feedback_mode = str(feedback_mode)
        if self.feedback_mode not in {"guided", "score_only"}:
            raise ValueError(f"unsupported feedback_mode: {self.feedback_mode!r}")
        self.run_started_epoch = _optional_float(self.run_metadata.get("run_started_epoch"))
        self.mcp.add_tool(
            self.evaluate_candidate,
            name=f"{self.name}_evaluate_candidate",
        )

    def evaluate_candidate(self, candidate_source: str) -> dict[str, Any]:
        """Run the complete trusted verification pipeline on one candidate and return status, metrics, evidence, diagnostics, and structured feedback."""
        if self.evaluations >= self.max_evaluations:
            raise RuntimeError(
                f"trusted verification budget exhausted after {self.max_evaluations} evaluations"
            )
        self.evaluations += 1
        started = time.perf_counter()
        context = {
            "allow_seed_candidate": candidate_source == self.initial_source,
            "agent_evaluation_index": self.evaluations,
            "role": "proposal",
        }
        intent = self.intent.to_dict()
        prepared = self._blocking(
            "prepare",
            prepare_artifact,
            candidate_source,
            context,
            self.backend_name,
            self.backend_settings,
            intent,
        )
        verified = self._blocking(
            "verify",
            verify_artifact,
            prepared,
            self.backend_name,
            self.backend_settings,
            intent,
        )
        evaluated = self._blocking(
            "evaluate",
            evaluate_artifact,
            verified,
            self.backend_name,
            self.backend_settings,
            intent,
        )
        final = self._blocking(
            "feedback",
            feedback_for_search,
            evaluated,
            self.backend_name,
            self.backend_settings,
            intent,
        )
        result = PipelineResult.from_dict(final)
        pipeline_wall_time_s = time.perf_counter() - started
        if self.result_log is not None:
            elapsed = None
            if self.run_started_epoch is not None:
                elapsed = max(0.0, time.time() - self.run_started_epoch)
            candidate_source_file, candidate_source_sha256 = persist_candidate_source(
                self.result_log, candidate_source
            )
            append_jsonl_sequenced(
                self.result_log,
                candidate_record(
                    run_metadata=self.run_metadata,
                    evaluation_index=self.evaluations,
                    role="proposal",
                    evaluation_origin="chia_tool",
                    result=result,
                    pipeline_wall_time_s=pipeline_wall_time_s,
                    elapsed_since_run_start_s=elapsed,
                    candidate_source_file=candidate_source_file,
                    candidate_source_sha256=candidate_source_sha256,
                ),
            )
        if getattr(self, "feedback_mode", "guided") == "score_only":
            return {
                "schema_version": result.schema_version,
                "backend": result.backend,
                "intent_id": result.intent_id,
                "status": result.status,
                "metrics": dict(result.metrics),
                "evaluation_index": self.evaluations,
                "remaining_evaluations": self.max_evaluations - self.evaluations,
            }
        # Return only useful verification feedback to the model.
        # Full evidence and certificates are saved in trusted_results.jsonl.
        # This keeps the conversation within the model's context limit.
        return {
            "backend": result.backend,
            "intent_id": result.intent_id,
            "status": result.status,
            "metrics": dict(result.metrics),
            "feedback": dict(result.feedback),
            "diagnostics": list(result.diagnostics),
            "evaluation_index": self.evaluations,
            "remaining_evaluations": self.max_evaluations - self.evaluations,
        }

    def _blocking(self, stage: str, node: Any, *args: Any) -> Any:
        resources = self.backend_resources.get(stage, {})
        handle = node.options(resources=resources) if resources else node
        tag = f"agent-{self.backend_name}-{self.evaluations:04d}-{stage}"
        return handle.chia_remote_blocking(*args, _chia_tag=tag)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
