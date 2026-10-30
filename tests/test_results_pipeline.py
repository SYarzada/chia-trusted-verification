from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chia_trusted_verification.results import (
    append_jsonl_sequenced,
    build_cost_summary,
    candidate_record,
    reconcile_search_result,
)
from chia_trusted_verification.types import PipelineResult, SearchResult


class ResultsPipelineTests(unittest.TestCase):
    def test_cost_estimation_uses_explicit_rates(self) -> None:
        result = build_cost_summary(
            raw_metrics={"usage": {"input_tokens": 2_000_000, "output_tokens": 500_000}},
            cost_tracking={
                "input_per_million_tokens_usd": 1.0,
                "output_per_million_tokens_usd": 4.0,
                "currency": "USD",
            },
            wall_time_s=10.0,
        )
        self.assertEqual(result["estimated_llm_cost_usd"], 4.0)
        self.assertEqual(result["total_recorded_cost_usd"], 4.0)
        self.assertEqual(result["llm_cost_source"], "estimated_from_tokens")


    def test_cloud_credit_is_reported_separately_from_cost(self) -> None:
        result = build_cost_summary(
            raw_metrics={},
            cost_tracking={
                "manual_llm_cost_usd": 7.5,
                "manual_cloud_credit_used_usd": 7.5,
                "billing_source": "GCP credits",
                "pricing_as_of": "2026-09-21",
            },
            wall_time_s=10.0,
        )
        self.assertEqual(result["total_recorded_cost_usd"], 7.5)
        self.assertEqual(result["manual_cloud_credit_used_usd"], 7.5)
        self.assertEqual(result["billing_source"], "GCP credits")

    def test_sequenced_jsonl_assigns_global_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trusted_results.jsonl"
            first = append_jsonl_sequenced(path, {"evaluation_id": "a", "evaluation_index": 1})
            second = append_jsonl_sequenced(path, {"evaluation_id": "b", "evaluation_index": 1})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(first["evaluation_sequence"], 1)
        self.assertEqual(second["evaluation_sequence"], 2)
        self.assertEqual([row["evaluation_index"] for row in rows], [1, 2])

    def test_candidate_record_uses_backend_candidate_identity_not_program_hash(self) -> None:
        pipeline = PipelineResult(
            backend="toy",
            intent_id="intent",
            status="pass",
            metrics={"legal": 1.0},
            feedback={},
            trace={"candidate_sha256": "candidate", "catalog_sha256": "catalog"},
        )
        record = candidate_record(
            run_metadata={"run_id": "run"},
            evaluation_index=1,
            role="proposal",
            evaluation_origin="trusted_evaluator",
            result=pipeline,
            pipeline_wall_time_s=0.1,
            elapsed_since_run_start_s=1.0,
        )
        self.assertEqual(record["candidate_sha256"], "candidate")
        self.assertEqual(record["catalog_sha256"], "catalog")
        self.assertEqual(record["candidate_id"], "cand_candidate")
        self.assertTrue(str(record["evaluation_id"]).startswith("eval_"))
        self.assertEqual(record["evaluation_local_index"], 1)
        self.assertNotIn("program_sha256", record)


    def test_reconcile_search_result_uses_trusted_best_and_convergence_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "trusted_results.jsonl"
            programs = root / "candidate_programs"
            programs.mkdir()
            seed_source = 'CANDIDATE = {"label": "seed", "mutation_ids": []}\n'
            worse_source = 'CANDIDATE = {"label": "worse", "mutation_ids": ["m011"]}\n'
            best_source = 'CANDIDATE = {"label": "best", "mutation_ids": ["m006", "m007"]}\n'
            (programs / "seed.py").write_text(seed_source, encoding="utf-8")
            (programs / "worse.py").write_text(worse_source, encoding="utf-8")
            (programs / "best.py").write_text(best_source, encoding="utf-8")

            base = {"record_type": "candidate_evaluation", "run_instance_id": "run_test"}
            append_jsonl_sequenced(log, {
                **base,
                "role": "seed_baseline",
                "status": "pass",
                "mutation_ids": [],
                "elapsed_since_run_start_s": 2.0,
                "candidate_id": "seed",
                "candidate_source_file": "candidate_programs/seed.py",
                "candidate_source_sha256": "seedsha",
                "metrics": {"legal": 1.0, "objective_available": 1.0, "objective_value": 5.0, "combined_score": 1.1},
            })
            append_jsonl_sequenced(log, {
                **base,
                "role": "proposal",
                "status": "pass",
                "mutation_ids": ["m011"],
                "elapsed_since_run_start_s": 8.0,
                "candidate_id": "worse",
                "candidate_source_file": "candidate_programs/worse.py",
                "candidate_source_sha256": "worsesha",
                "metrics": {"legal": 1.0, "objective_available": 1.0, "objective_value": 4.0, "combined_score": 1.5},
            })
            append_jsonl_sequenced(log, {
                **base,
                "role": "proposal",
                "status": "fail",
                "mutation_ids": ["m016"],
                "elapsed_since_run_start_s": 10.0,
                "candidate_id": "bad",
                "metrics": {"legal": 0.0, "objective_available": 0.0, "objective_value": 0.0, "combined_score": -1.0},
            })
            append_jsonl_sequenced(log, {
                **base,
                "role": "proposal",
                "status": "pass",
                "mutation_ids": ["m006", "m007"],
                "elapsed_since_run_start_s": 14.5,
                "candidate_id": "best",
                "candidate_source_file": "candidate_programs/best.py",
                "candidate_source_sha256": "bestsha",
                "metrics": {"legal": 1.0, "objective_available": 1.0, "objective_value": 3.0, "combined_score": 1.9},
            })

            engine_result = SearchResult(
                engine="alphaevolve",
                best_score=1.5,
                best_solution=worse_source,
                metrics={"search_iteration_count": 3},
                output_dir=str(root),
            )
            reconciled = reconcile_search_result(
                engine_result,
                result_log=log,
                run_instance_id="run_test",
            )

        self.assertEqual(reconciled.best_score, 1.9)
        self.assertEqual(reconciled.best_solution, best_source)
        self.assertEqual(reconciled.metrics["best_trusted_mutation_ids"], ["m006", "m007"])
        self.assertEqual(reconciled.metrics["best_trusted_proposal_index"], 3)
        self.assertEqual(reconciled.metrics["best_trusted_evaluation_count"], 4)
        self.assertEqual(reconciled.metrics["best_trusted_time_to_best_s"], 14.5)
        self.assertEqual(reconciled.metrics["best_trusted_time_from_first_proposal_s"], 6.5)
        self.assertEqual(reconciled.metrics["first_legal_proposal_index"], 1)
        self.assertEqual(reconciled.metrics["search_proposal_evaluations"], 3)
        self.assertEqual(reconciled.metrics["search_total_candidate_evaluations"], 4)
        self.assertEqual(reconciled.metrics["engine_reported_best_score"], 1.5)
        self.assertFalse(reconciled.metrics["engine_reported_best_matches_trusted_best"])



if __name__ == "__main__":
    unittest.main()
