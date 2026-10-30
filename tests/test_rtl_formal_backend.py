from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chia_trusted_verification.chia.evaluator import TrustedVerificationEvaluator
from chia_trusted_verification.core import backend_names, get_backend
from chia_trusted_verification.backends.rtl_formal.collateral import (
    CollateralValidationError,
    analyze_collateral,
)
from chia_trusted_verification.backends.rtl_formal.intents import (
    get_intent_spec,
    load_intent_catalog,
)
from chia_trusted_verification.types import VerificationIntent


ROOT = Path(__file__).resolve().parents[1]


class RtlFormalBackendTests(unittest.TestCase):
    def test_registered_builtin(self) -> None:
        self.assertIn("rtl_formal", backend_names())
        self.assertEqual(get_backend("rtl_formal").name, "rtl_formal")

    def test_gemmini_catalog_has_balanced_twenty_four_intents(self) -> None:
        _, specs = load_intent_catalog(ROOT / "rtl_formal/gemmini/intents.yaml")
        self.assertEqual(len(specs), 24)
        self.assertEqual(
            Counter(spec.category for spec in specs.values()),
            Counter(
                {
                    "queue_dependency": 6,
                    "control": 6,
                    "memory_dma": 6,
                    "interface_protocol": 6,
                }
            ),
        )

    def test_agent_assumptions_are_rejected_by_default(self) -> None:
        spec = get_intent_spec(ROOT / "rtl_formal/smoke/intents.yaml", "smoke_queue_issue")
        candidate = """
        always @(posedge clock) begin
          if (!reset) begin
            assume(allocated);
            assert(!issue_valid || deps == 0);
            cover(issue_valid);
          end
        end
        """
        with self.assertRaises(CollateralValidationError):
            analyze_collateral(candidate, spec, strict_required_signals=False)

    def test_agent_cannot_drive_fixed_rtl(self) -> None:
        spec = get_intent_spec(ROOT / "rtl_formal/smoke/intents.yaml", "smoke_queue_issue")
        for mutation in (
            "assign issue_valid = 1'b0;\nassert(1'b1);\ncover(1'b1);",
            "always @(posedge clock) begin issued <= 1'b0; assert(1'b1); cover(1'b1); end",
            "restrict(allocated);\nassert(1'b1);\ncover(1'b1);",
        ):
            with self.subTest(mutation=mutation), self.assertRaises(CollateralValidationError):
                analyze_collateral(mutation, spec, strict_required_signals=False)

    def test_smoke_candidate_runs_through_generic_pipeline_with_fake_solver(self) -> None:
        backend_settings = {
            "_config_dir": str(ROOT / "configs"),
            "intent_catalog": "../rtl_formal/smoke/intents.yaml",
            "rtl_files": ["../rtl_formal/smoke/gemmini_formal_smoke.sv"],
            "target_file": "../rtl_formal/smoke/gemmini_formal_smoke.sv",
            "top_module": "GemminiFormalSmoke",
            "strict_required_signals": True,
        }
        intent = VerificationIntent(
            "smoke_queue_issue",
            "systemverilog_formal_collateral",
            (
                "assertions_compile",
                "assertions_non_vacuous",
                "prove_selected_intents",
                "reach_selected_covers",
                "trusted_assumption_policy",
            ),
        )
        candidate = (ROOT / "rtl_formal/smoke/known_good_collateral.sv").read_text()
        fake_result = {
            "status": "proved",
            "formal_engine": "smtbmc",
            "compile": {"returncode": 0, "output_tail": "compile ok"},
            "bmc": {"returncode": 0, "output_tail": "BMC pass"},
            "induction": {"returncode": 0, "output_tail": "induction pass"},
            "cover": {"returncode": 0, "output_tail": "cover reached"},
            "runtime_s": 0.01,
            "artifacts": {},
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "chia_trusted_verification.backends.rtl_formal.backend.run_formal",
            return_value=fake_result,
        ):
            settings = dict(backend_settings)
            settings["workspace_root"] = str(Path(tmp) / "workspaces")
            evaluator = TrustedVerificationEvaluator(
                backend_name="rtl_formal",
                backend_settings=settings,
                backend_resources={},
                intent=intent,
                engine="llm_agent",
                initial_source="seed",
                result_log=Path(tmp) / "results.jsonl",
                use_chia=False,
            )
            result = evaluator.evaluate_source(candidate)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.metrics["proved"], 1.0)
        self.assertEqual(result.metrics["cover_ok"], 1.0)
        self.assertEqual(result.metrics["assume_count"], 0.0)

    def test_search_problem_uses_verification_intent(self) -> None:
        backend = get_backend("rtl_formal")
        settings = {
            "_config_dir": str(ROOT / "configs"),
            "intent_catalog": "../rtl_formal/smoke/intents.yaml",
            "rtl_files": ["../rtl_formal/smoke/gemmini_formal_smoke.sv"],
            "target_file": "../rtl_formal/smoke/gemmini_formal_smoke.sv",
            "top_module": "GemminiFormalSmoke",
            "_verification_intent": {"intent_id": "smoke_queue_issue"},
        }
        problem = backend.build_search_problem(settings)
        description = problem.guidance.problem_description or ""
        self.assertIn("smoke_queue_issue", description)
        self.assertIn("Generated RTL context", description)
        self.assertIn("issue_valid", description)
        self.assertIn("CHIA-BEGIN-AGENT-COLLATERAL", problem.initial_source)


if __name__ == "__main__":
    unittest.main()
