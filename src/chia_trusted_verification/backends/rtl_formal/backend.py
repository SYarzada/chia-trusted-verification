from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chia_trusted_verification.core.backend import TrustedVerificationBackend
from chia_trusted_verification.types import (
    PIPELINE_SCHEMA_VERSION,
    SearchGuidance,
    SearchProblem,
    VerificationIntent,
)

from .collateral import CollateralValidationError, analyze_collateral, render_seed_candidate
from .formal_runner import FormalSetupError, describe_target_rtl, prepare_formal_workspace, run_formal
from .intents import FormalIntentSpec, get_intent_spec


class RtlFormalBackend(TrustedVerificationBackend):
    name = "rtl_formal"
    supported_artifact_type = "systemverilog_formal_collateral"
    supported_requirements = frozenset(
        {
            "assertions_compile",
            "assertions_non_vacuous",
            "prove_selected_intents",
            "reach_selected_covers",
            "trusted_assumption_policy",
        }
    )

    def build_search_problem(self, settings: Mapping[str, Any]) -> SearchProblem:
        intent_value = settings.get("_verification_intent")
        if isinstance(intent_value, Mapping):
            intent_id = str(intent_value.get("intent_id", ""))
        else:
            intent_id = str(settings.get("intent_id", ""))
        if not intent_id:
            raise ValueError(
                "RTL-formal search construction requires a verification intent; use the normal CHIA loop "
                "or set backend.settings.intent_id"
            )
        spec = self._spec(intent_id, settings)
        initial_source = render_seed_candidate(spec)
        signals = ", ".join(spec.signal_hints) if spec.signal_hints else "inspect the target RTL"
        source_anchor = (
            f"Source anchor: {spec.source_file}: {spec.source_anchor}. "
            if spec.source_file and spec.source_anchor
            else ""
        )
        rtl_context = describe_target_rtl(spec, settings)
        rtl_context_text = (
            "\n\nGenerated RTL context (trusted read-only context; use exact identifiers shown here):\n"
            + rtl_context
            if rtl_context
            else ""
        )
        return SearchProblem(
            initial_source=initial_source,
            file_name=f"{spec.intent_id}_formal.sv",
            metadata={"formal_intent": spec.to_dict()},
            guidance=SearchGuidance(
                system_message=(
                    "You generate untrusted SystemVerilog formal collateral for a fixed RTL design. "
                    "Modify only the CHIA agent-collateral region. Do not emit modules, endmodule, bind, "
                    "preprocessor directives, initial blocks, assignments/drives, restrict statements, file I/O, "
                    "or edits to the RTL. Agent-generated assumptions "
                    "are forbidden unless the trusted verification intent explicitly permits them. Use "
                    "simple Yosys-compatible clocked assert(...) and cover(...) statements rather than "
                    "advanced SVA syntax. The assertion must directly express the configured requirement; "
                    "the cover must establish that its meaningful triggering behavior is reachable. Use "
                    "formal counterexamples and diagnostics returned by the trusted tool to repair later "
                    "attempts. Never weaken a property merely to make the solver pass."
                ),
                problem_description=(
                    f"Verification intent {spec.intent_id} ({spec.category}): {spec.requirement}\n"
                    f"Target module: {spec.target_module}. Signal/source hints: {signals}. "
                    f"{source_anchor}Generate at least one assertion and one cover unless the intent says otherwise."
                    f"{rtl_context_text}"
                ),
            ),
        )

    def prepare(
        self,
        candidate_source: str,
        context: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_intent(intent)
        spec = self._spec(intent.intent_id, settings)
        strict_signals = bool(settings.get("strict_required_signals", False))
        try:
            analysis = analyze_collateral(
                candidate_source,
                spec,
                strict_required_signals=strict_signals,
            )
            workspace = prepare_formal_workspace(
                candidate_source=analysis.source,
                spec=spec,
                settings=settings,
            )
        except (CollateralValidationError, FormalSetupError, OSError, ValueError) as exc:
            return {
                "status": "fail",
                "representation_valid": False,
                "diagnostics": [
                    {
                        "stage": "prepare",
                        "kind": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
                "context": dict(context),
                "formal_intent": spec.to_dict(),
            }
        return {
            "status": "pass",
            "representation_valid": True,
            "candidate_source": analysis.source,
            "collateral_analysis": analysis.to_dict(),
            "workspace": workspace,
            "context": dict(context),
            "formal_intent": spec.to_dict(),
            "diagnostics": [],
        }

    def verify(
        self,
        prepared: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_intent(intent)
        if prepared.get("status") != "pass":
            return {**dict(prepared), "status": "fail", "formal_status": "not_run"}
        spec = self._spec(intent.intent_id, settings)
        try:
            result = run_formal(dict(prepared["workspace"]), spec, settings)
        except (FormalSetupError, OSError, ValueError) as exc:
            diagnostics = list(prepared.get("diagnostics", []))
            diagnostics.append(
                {"stage": "verify", "kind": type(exc).__name__, "message": str(exc)}
            )
            return {
                **dict(prepared),
                "status": "error",
                "formal_status": "tool_error",
                "diagnostics": diagnostics,
            }
        formal_status = str(result.get("status", "tool_error"))
        status = "pass" if formal_status == "proved" else (
            "error" if formal_status == "tool_error" else "fail"
        )
        return {
            **dict(prepared),
            "status": status,
            "formal_status": formal_status,
            "formal_result": result,
        }

    def evaluate(
        self,
        verified: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        del settings
        self._validate_intent(intent)
        result = dict(verified.get("formal_result", {}))
        compile_run = result.get("compile") if isinstance(result.get("compile"), Mapping) else {}
        bmc = result.get("bmc") if isinstance(result.get("bmc"), Mapping) else {}
        induction = result.get("induction") if isinstance(result.get("induction"), Mapping) else {}
        cover = result.get("cover") if isinstance(result.get("cover"), Mapping) else None
        compile_ok = bool(compile_run) and int(compile_run.get("returncode", 1)) == 0
        bmc_ok = bool(bmc) and int(bmc.get("returncode", 1)) == 0
        induction_ok = bool(induction) and int(induction.get("returncode", 1)) == 0
        cover_required = "cover" in self._spec_from_verified(verified).required_property_kinds
        cover_ok = (not cover_required) or (
            isinstance(cover, Mapping) and int(cover.get("returncode", 1)) == 0
        )
        proved = str(verified.get("formal_status")) == "proved" and compile_ok and bmc_ok and induction_ok
        accepted = proved and cover_ok
        return {
            **dict(verified),
            "status": "pass" if accepted else str(verified.get("status", "fail")),
            "compile_ok": compile_ok,
            "bounded_check_ok": bmc_ok,
            "induction_ok": induction_ok,
            "cover_required": cover_required,
            "cover_ok": cover_ok,
            "proved": proved,
            "accepted": accepted,
        }

    def feedback(
        self,
        evaluated: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        del settings
        self._validate_intent(intent)
        stage_status = str(evaluated.get("status", "error"))
        accepted = bool(evaluated.get("accepted", False))
        compile_ok = bool(evaluated.get("compile_ok", False))
        bmc_ok = bool(evaluated.get("bounded_check_ok", False))
        induction_ok = bool(evaluated.get("induction_ok", False))
        cover_ok = bool(evaluated.get("cover_ok", False))
        formal_status = str(evaluated.get("formal_status", "not_run"))
        analysis = dict(evaluated.get("collateral_analysis", {}))
        formal_result = dict(evaluated.get("formal_result", {}))
        diagnostics = list(evaluated.get("diagnostics", []))

        if accepted:
            score = 2.0
            summary = "Generated collateral compiled, proved by induction, and reached its required cover."
            action = "Retain this candidate; it satisfies the configured trusted formal checks."
        elif stage_status == "error":
            score = -100.0
            summary = "Trusted formal execution could not complete."
            action = "Fix the reported tool/setup error before changing the property semantics."
        elif not bool(evaluated.get("representation_valid", False)):
            score = -10.0
            summary = "Generated collateral violated the trusted candidate-format policy."
            action = "Repair the collateral format and required property structure."
        elif not compile_ok:
            score = -5.0
            summary = "Generated collateral did not compile/elaborate with the fixed RTL."
            action = "Repair syntax or signal references without weakening the intended property."
        elif not bmc_ok:
            score = 0.0
            summary = "The assertion has a bounded counterexample."
            action = "Use the counterexample to correct the assertion if it misstates the intent; do not hide an RTL violation."
        elif not induction_ok:
            score = 0.5
            summary = "No bounded counterexample was found, but the property was not inductively proved."
            action = "Strengthen the property structure or trusted harness only when justified by the intent."
        elif not cover_ok:
            score = 1.0
            summary = "The assertion proved, but the required cover was not reached."
            action = "Repair the cover/trigger so the checked behavior is demonstrably reachable and non-vacuous."
        else:
            score = -1.0
            summary = "Generated collateral did not satisfy all trusted formal checks."
            action = "Use the formal diagnostics to repair the next proposal."

        messages = []
        for key in ("compile", "bmc", "induction", "cover"):
            run = formal_result.get(key)
            if isinstance(run, Mapping) and run.get("output_tail"):
                messages.append(f"{key}: {str(run['output_tail'])[-1800:]}")
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "pass" if accepted else ("error" if stage_status == "error" else "fail"),
            "metrics": {
                "combined_score": score,
                "score": score,
                "representation_valid": 1.0 if evaluated.get("representation_valid") else 0.0,
                "compile_ok": 1.0 if compile_ok else 0.0,
                "bounded_check_ok": 1.0 if bmc_ok else 0.0,
                "induction_ok": 1.0 if induction_ok else 0.0,
                "cover_ok": 1.0 if cover_ok else 0.0,
                "proved": 1.0 if evaluated.get("proved") else 0.0,
                "formal_runtime_s": float(formal_result.get("runtime_s", 0.0) or 0.0),
                "assert_count": float(analysis.get("assert_count", 0) or 0),
                "cover_count": float(analysis.get("cover_count", 0) or 0),
                "assume_count": float(analysis.get("assume_count", 0) or 0),
            },
            "feedback": {
                "summary": summary,
                "suggested_action": action,
                "formal_status": formal_status,
                "messages": messages[-4:],
                "missing_required_signals": list(analysis.get("missing_required_signals", [])),
            },
            "evidence": {
                "verification_intent": intent.to_dict(),
                "formal_intent": evaluated.get("formal_intent"),
                "collateral_analysis": analysis,
                "formal_result": formal_result,
            },
            "trace": {
                "candidate_sha256": dict(evaluated.get("workspace", {})).get("candidate_sha256"),
                "workspace": dict(evaluated.get("workspace", {})).get("workspace"),
                "formal_status": formal_status,
                "terminal_stage": "evaluate",
            },
            "diagnostics": diagnostics,
        }

    def _validate_intent(self, intent: VerificationIntent) -> None:
        if intent.artifact_type != self.supported_artifact_type:
            raise ValueError(
                f"RTL-formal expects artifact_type {self.supported_artifact_type!r}, got {intent.artifact_type!r}"
            )
        unsupported = sorted(set(intent.requirements) - self.supported_requirements)
        if unsupported:
            raise ValueError(
                "RTL formal verification intent contains unsupported requirement(s): "
                + ", ".join(unsupported)
            )

    def _spec(self, intent_id: str, settings: Mapping[str, Any]) -> FormalIntentSpec:
        path = self._catalog_path(settings)
        return get_intent_spec(path, intent_id)

    def _catalog_path(self, settings: Mapping[str, Any]) -> Path:
        base = Path(str(settings.get("_config_dir", "."))).resolve()
        raw = Path(str(settings.get("intent_catalog", "../rtl_formal/gemmini/intents.yaml"))).expanduser()
        return (raw if raw.is_absolute() else base / raw).resolve()

    def _spec_from_verified(self, value: Mapping[str, Any]) -> FormalIntentSpec:
        raw = value.get("formal_intent")
        if not isinstance(raw, Mapping):
            raise ValueError("formal_intent metadata is missing from verified result")
        return FormalIntentSpec.from_dict(raw)
