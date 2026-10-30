from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from chia_trusted_verification.config import base_search_engine, search_feedback_mode
from chia_trusted_verification.results import (
    append_jsonl_sequenced,
    candidate_record,
    persist_candidate_source,
)
from chia_trusted_verification.types import PipelineResult, VerificationIntent

from .nodes import (
    evaluate_artifact,
    evaluate_artifact_local,
    feedback_for_search,
    feedback_for_search_local,
    prepare_artifact,
    prepare_artifact_local,
    verify_artifact,
    verify_artifact_local,
)
from .runtime import CHIA_AVAILABLE, get, require_chia


class TrustedVerificationEvaluator:
    def __init__(
        self,
        *,
        backend_name: str,
        backend_settings: Mapping[str, Any],
        backend_resources: Mapping[str, Mapping[str, float]],
        intent: VerificationIntent,
        engine: str,
        initial_source: str,
        result_log: str | Path,
        run_metadata: Mapping[str, Any] | None = None,
        use_chia: bool = True,
    ) -> None:
        self.backend_name = str(backend_name)
        self.backend_settings = dict(backend_settings)
        self.backend_resources = {
            str(stage): {str(k): float(v) for k, v in resources.items()}
            for stage, resources in backend_resources.items()
        }
        self.intent = intent
        self.engine = engine
        self.initial_source = initial_source
        self.result_log = Path(result_log).resolve()
        self.run_metadata = dict(run_metadata or {})
        self.run_started_epoch = _optional_float(self.run_metadata.get("run_started_epoch"))
        self.use_chia = use_chia
        self._write_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._evaluation_index = 0
        if use_chia:
            require_chia()

    def __call__(self, program_path: str) -> Any:
        source = Path(program_path).read_text(encoding="utf-8")
        result = self.evaluate_source(source, role="proposal")
        return self.format_engine_result(result)

    def evaluate_source(self, source: str, *, role: str = "proposal") -> PipelineResult:
        evaluation_index = self._next_evaluation_index()
        started = time.perf_counter()
        context = {
            "allow_seed_candidate": source == self.initial_source,
            "evaluation_index": evaluation_index,
            "role": role,
        }
        intent = self.intent.to_dict()
        if self.use_chia:
            if not CHIA_AVAILABLE:
                require_chia()
            prefix = f"{self.backend_name}-{self.run_metadata.get('run_id', 'run')}-{evaluation_index:05d}"
            prepared = self._remote(
                "prepare",
                prepare_artifact,
                source,
                context,
                self.backend_name,
                self.backend_settings,
                intent,
                tag=f"{prefix}-prepare",
            )
            verified = self._remote(
                "verify",
                verify_artifact,
                prepared,
                self.backend_name,
                self.backend_settings,
                intent,
                tag=f"{prefix}-verify",
            )
            evaluated = self._remote(
                "evaluate",
                evaluate_artifact,
                verified,
                self.backend_name,
                self.backend_settings,
                intent,
                tag=f"{prefix}-evaluate",
            )
            final = get(
                self._remote(
                    "feedback",
                    feedback_for_search,
                    evaluated,
                    self.backend_name,
                    self.backend_settings,
                    intent,
                    tag=f"{prefix}-feedback",
                )
            )
        else:
            prepared = prepare_artifact_local(
                source,
                context,
                self.backend_name,
                self.backend_settings,
                intent,
            )
            verified = verify_artifact_local(
                prepared,
                self.backend_name,
                self.backend_settings,
                intent,
            )
            evaluated = evaluate_artifact_local(
                verified,
                self.backend_name,
                self.backend_settings,
                intent,
            )
            final = feedback_for_search_local(
                evaluated,
                self.backend_name,
                self.backend_settings,
                intent,
            )

        result = PipelineResult.from_dict(final)
        pipeline_wall_time_s = time.perf_counter() - started
        self._append_result(
            evaluation_index=evaluation_index,
            role=role,
            result=result,
            pipeline_wall_time_s=pipeline_wall_time_s,
            source=source,
        )
        return result

    def _next_evaluation_index(self) -> int:
        with self._counter_lock:
            self._evaluation_index += 1
            return self._evaluation_index

    def _remote(self, stage: str, node: Any, *args: Any, tag: str) -> Any:
        resources = self.backend_resources.get(stage, {})
        handle = node.options(resources=resources) if resources else node
        return handle.chia_remote(*args, _chia_tag=tag)

    def _append_result(
        self,
        *,
        evaluation_index: int,
        role: str,
        result: PipelineResult,
        pipeline_wall_time_s: float,
        source: str,
    ) -> None:
        elapsed = None
        if self.run_started_epoch is not None:
            elapsed = max(0.0, time.time() - self.run_started_epoch)
        candidate_source_file, candidate_source_sha256 = persist_candidate_source(
            self.result_log, source
        )
        record = candidate_record(
            run_metadata=self.run_metadata,
            evaluation_index=evaluation_index,
            role=role,
            evaluation_origin="trusted_evaluator",
            result=result,
            pipeline_wall_time_s=pipeline_wall_time_s,
            elapsed_since_run_start_s=elapsed,
            candidate_source_file=candidate_source_file,
            candidate_source_sha256=candidate_source_sha256,
        )
        append_jsonl_sequenced(self.result_log, record, self._write_lock)


    def format_engine_result(self, result: PipelineResult) -> Any:
        metrics = dict(result.metrics)
        base_engine = base_search_engine(self.engine)
        guided = search_feedback_mode(self.engine) == "guided"
        trusted = {
            "trusted_backend": result.backend,
            "trusted_intent_id": result.intent_id,
            "trusted_status": result.status,
            "trusted_feedback_json": json.dumps(result.feedback, sort_keys=True),
            "trusted_evidence_json": json.dumps(result.evidence, sort_keys=True),
            "trusted_trace_json": json.dumps(result.trace, sort_keys=True),
            "trusted_diagnostics_json": json.dumps(result.diagnostics, sort_keys=True),
            "trusted_stage_timings_json": json.dumps(result.stage_timings_s, sort_keys=True),
        }
        if base_engine == "alphaevolve":
            if not guided:
                return metrics
            return {**metrics, **{f"__{key}": value for key, value in trusted.items()}}
        if base_engine != "openevolve":
            return metrics
        try:
            from openevolve.evaluation_result import EvaluationResult
        except ImportError:
            return metrics
        return EvaluationResult(metrics=metrics, artifacts=(trusted if guided else {}))

    def _engine_result(self, result: PipelineResult) -> Any:
        return self.format_engine_result(result)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


ChiaPipelineEvaluator = TrustedVerificationEvaluator
