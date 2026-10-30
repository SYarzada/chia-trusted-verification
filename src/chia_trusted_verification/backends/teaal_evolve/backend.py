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

from .candidate_program import CandidateProgramError, extract_candidate_payload, render_initial_program
from .executor import run_worker_stage


class TeaalEvolveBackend(TrustedVerificationBackend):
    name = "teaal_evolve"
    supported_artifact_type = "structured_accelerator_candidate"
    supported_requirements = frozenset(
        {
            "mapping_format_binding_legality",
            "local_semantics",
            "producer_consumer_interfaces",
            "structured_resource_legality",
        }
    )

    def build_search_problem(self, settings: Mapping[str, Any]) -> SearchProblem:
        worker = self._worker_settings(settings)
        catalog_result = run_worker_stage("catalog", {}, worker)
        if catalog_result.get("status") != "pass":
            raise RuntimeError(f"could not build curated TeAAL-Evolve catalog: {catalog_result}")
        source = render_initial_program(
            catalog_result["catalog"],
            min_mutations=int(worker.get("min_mutations", 1)),
            max_mutations=int(worker.get("max_mutations", 2)),
        )
        min_mutations = int(worker.get("min_mutations", 1))
        max_mutations = int(worker.get("max_mutations", 2))
        return SearchProblem(
            initial_source=source,
            file_name="initial_program.py",
            metadata={
                "catalog_sha256": catalog_result.get("catalog_sha256"),
                "catalog_size": len(catalog_result.get("catalog", [])),
            },
            guidance=SearchGuidance(
                system_message=(
                    "Optimize a structured TeAAL accelerator candidate by editing only the literal "
                    "CANDIDATE value inside EVOLVE-BLOCK. Select between "
                    f"{min_mutations} and {max_mutations} unique mutation_ids from MUTATION_CATALOG. "
                    "Do not modify MUTATION_CATALOG, VALID_MUTATION_IDS, helper functions, imports, "
                    "or executable scaffolding. Candidates are accepted only after trusted legality "
                    "certification and objective evaluation. Lower legal objective values are better. "
                    "Use failed trusted checks and certificate-backed diagnostics to guide repairs."
                ),
                evaluator_system_message=(
                    "Preserve trusted evaluator metrics, certificates, diagnostics, and feedback "
                    "without subjective rescoring."
                ),
                problem_description=(
                    "Search over bounded combinations of curated TeAAL mapping, format, binding, "
                    "placement, loop-order, spacetime, and related structured accelerator mutations. "
                    "Each proposal is materialized from the fixed legal seed, certified by the "
                    "TeAAL-Evolve trusted backend, and scored only if legal. The immutable "
                    "MUTATION_CATALOG lists every allowed mutation ID and MIN_MUTATIONS/MAX_MUTATIONS "
                    "define the exact proposal-size bounds. The seed with zero mutations is the only "
                    "zero-mutation exception. Every generated candidate must keep exactly one "
                    "top-level CANDIDATE assignment inside EVOLVE-BLOCK. CANDIDATE must remain a "
                    "Python literal dictionary parseable by ast.literal_eval with keys label and "
                    "mutation_ids. mutation_ids must contain only unique string IDs copied verbatim "
                    "from MUTATION_CATALOG (for example m006, never numeric 6 or 0). Do not use "
                    "variables, function calls, comprehensions, concatenation, or other executable "
                    "expressions inside CANDIDATE, and do not edit immutable code outside EVOLVE-BLOCK. "
                    "Invalid candidates receive graded negative combined_score values so that "
                    "candidates with fewer failed trusted obligations rank above candidates with "
                    "more failures. Representation or pipeline errors receive much larger negative "
                    "penalties. Use trusted verification feedback and failed-check insights to "
                    "repair later proposals rather than repeating a known-failing mutation set."
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
        try:
            proposal = extract_candidate_payload(candidate_source)
        except CandidateProgramError as exc:
            proposal = {
                "_adapter_error": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            }
        return run_worker_stage(
            "prepare",
            {
                "proposal": proposal,
                "allow_seed_candidate": bool(context.get("allow_seed_candidate", False)),
            },
            self._worker_settings(settings),
        )

    def verify(
        self,
        prepared: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_intent(intent)
        result = run_worker_stage("verify", prepared, self._worker_settings(settings))
        result["verification_intent"] = intent.to_dict()
        return result

    def evaluate(
        self,
        verified: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_intent(intent)
        return run_worker_stage("evaluate", verified, self._worker_settings(settings))

    def feedback(
        self,
        evaluated: Mapping[str, Any],
        intent: VerificationIntent,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        del settings
        self._validate_intent(intent)
        stage_status = str(evaluated.get("status", "error"))
        legal = bool(evaluated.get("legal", False))
        objective = dict(evaluated.get("objective", {}))
        value = objective.get("value")
        objective_ok = objective.get("status") == "ok" and value is not None
        accepted = stage_status == "pass" and legal and objective_ok
        numeric_value = float(value) if objective_ok else 0.0
        checks = list(evaluated.get("checks", []))
        failed_checks = [
            check for check in checks if check.get("status") in {"fail", "error"}
        ]
        if accepted:
            # Legal designs score above 1. Lower objectives get higher scores.
            score = 1.0 + 1.0 / (1.0 + max(0.0, numeric_value))
        elif stage_status == "error":
            # Execution errors receive the lowest score.
            score = -100.0
        elif not legal:
            # Fewer failed checks give illegal designs a higher score.
            score = -float(max(1, len(failed_checks)))
        else:
            # Legal designs without a valid objective receive -50.
            score = -50.0
        diagnostics = list(evaluated.get("diagnostics", []))
        messages = [str(item.get("message", "")) for item in diagnostics]
        messages.extend(str(check.get("message", "")) for check in failed_checks[:8])
        if accepted:
            summary = "Candidate passed trusted verification and objective evaluation."
            action = "Retain this candidate as a legal objective-scored parent."
        elif stage_status == "error":
            summary = "Candidate could not complete the trusted evaluation pipeline."
            action = (
                "Repair the proposal or pipeline error reported in diagnostics; do not repeat "
                "this exact candidate unchanged."
            )
        elif not legal:
            summary = "Candidate failed trusted TeAAL-Evolve verification."
            action = (
                "Do not repeat this exact mutation set unchanged. Choose a different curated "
                "mutation set that avoids the failed trusted obligations."
            )
        else:
            summary = "Candidate could not complete the trusted evaluation pipeline."
            action = (
                "Repair the objective-evaluation failure reported in diagnostics; do not repeat "
                "this exact candidate unchanged."
            )
        proposal = dict(evaluated.get("proposal", {}))
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "pass" if accepted else ("error" if stage_status == "error" else "fail"),
            "metrics": {
                "combined_score": score,
                "score": score,
                "legal": 1.0 if legal else 0.0,
                "certification_passed": 1.0 if legal else 0.0,
                "objective_available": 1.0 if objective_ok else 0.0,
                "objective_value": numeric_value,
                "failed_checks": float(len(failed_checks)),
                "runs_successfully": 0.0 if stage_status == "error" else 1.0,
            },
            "feedback": {
                "summary": summary,
                "suggested_action": action,
                "mutation_ids": list(proposal.get("mutation_ids", [])),
                "failed_check_ids": [str(check.get("check_id")) for check in failed_checks],
                "failed_categories": sorted({str(check.get("category")) for check in failed_checks}),
                "messages": [message for message in messages if message][:12],
                "objective": objective,
            },
            "evidence": {
                "certificate": evaluated.get("certificate"),
                "certificate_size_bytes": evaluated.get("certificate_size_bytes"),
                "certificate_validation": evaluated.get("certificate_validation"),
                "certificate_validation_time_s": evaluated.get("certificate_validation_time_s"),
                "checks": checks,
                "verification_intent": intent.to_dict(),
            },
            "trace": {
                "proposal": proposal,
                "candidate_sha256": evaluated.get("candidate_sha256"),
                "catalog_sha256": evaluated.get("catalog_sha256"),
                "terminal_stage": evaluated.get("stage"),
                "terminal_status": stage_status,
            },
        }

    def _validate_intent(self, intent: VerificationIntent) -> None:
        if intent.artifact_type != self.supported_artifact_type:
            raise ValueError(
                f"TeAAL-Evolve expects artifact_type {self.supported_artifact_type!r}, "
                f"got {intent.artifact_type!r}"
            )
        unsupported = sorted(set(intent.requirements) - self.supported_requirements)
        if unsupported:
            raise ValueError(
                "TeAAL-Evolve verification intent contains unsupported requirement(s): "
                + ", ".join(unsupported)
            )

    def _worker_settings(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        base = Path(str(settings.get("_config_dir", "."))).resolve()

        def resolve(value: Any) -> str:
            path = Path(str(value)).expanduser()
            return str((path if path.is_absolute() else base / path).resolve())

        return {
            "teaal_root": resolve(settings.get("root", "../teaal_evolve")),
            "kernel_label": str(settings.get("kernel_label", "attention")),
            "kernel_file": resolve(
                settings.get("kernel_file", "../teaal_evolve/example/kernel/nki/attention.yaml")
            ),
            "architecture_file": resolve(
                settings.get("architecture_file", "../teaal_evolve/example/arch/trainium1.yaml")
            ),
            "checker_backend": str(settings.get("checker_backend", "hierarchical")),
            "objective_backend": str(settings.get("objective_backend", "monolithic_bottleneck")),
            "knob_space": str(settings.get("knob_space", "rich")),
            "min_mutations": int(settings.get("min_mutations", 1)),
            "max_mutations": int(settings.get("max_mutations", 2)),
            "catalog_seed": int(settings.get("catalog_seed", 0)),
            "objective_shapes": {
                str(key): int(value)
                for key, value in dict(settings.get("objective_shapes", {})).items()
            },
            "node_timeout_s": int(settings.get("node_timeout_s", 300)),
        }
