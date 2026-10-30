from __future__ import annotations

from typing import Any, Mapping

from chia_trusted_verification.types import PIPELINE_SCHEMA_VERSION, VerificationIntent

from .registry import get_backend


class TrustedVerificationLayer:
    def __init__(
        self,
        *,
        backend_name: str,
        backend_settings: Mapping[str, Any],
        intent: VerificationIntent,
    ) -> None:
        self.backend = get_backend(backend_name)
        self.backend_settings = dict(backend_settings)
        self.intent = intent

    def prepare(self, candidate_source: str, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._normalize(
            "prepare",
            self.backend.prepare(
                candidate_source,
                context,
                self.intent,
                self.backend_settings,
            ),
        )

    def verify(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return self._normalize(
            "verify",
            self.backend.verify(prepared, self.intent, self.backend_settings),
        )

    def evaluate(self, verified: Mapping[str, Any]) -> dict[str, Any]:
        return self._normalize(
            "evaluate",
            self.backend.evaluate(verified, self.intent, self.backend_settings),
        )

    def feedback(self, evaluated: Mapping[str, Any]) -> dict[str, Any]:
        result = self._normalize(
            "feedback",
            self.backend.feedback(evaluated, self.intent, self.backend_settings),
        )
        result.setdefault("stage_timings_s", dict(evaluated.get("stage_timings_s", {})))
        for key in ("metrics", "feedback", "evidence", "trace"):
            if not isinstance(result.get(key, {}), Mapping):
                raise TypeError(f"backend feedback result field {key!r} must be a mapping")
        return result

    def _normalize(self, stage: str, value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result.setdefault("schema_version", PIPELINE_SCHEMA_VERSION)
        result["stage"] = stage
        result["backend"] = self.backend.name
        result["intent_id"] = self.intent.intent_id
        result.setdefault("status", "error")
        result.setdefault("diagnostics", [])
        return result
