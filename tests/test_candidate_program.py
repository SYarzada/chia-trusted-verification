from __future__ import annotations

import unittest

from chia_trusted_verification.backends.teaal_evolve.candidate_program import (
    CandidateProgramError,
    extract_candidate_payload,
    render_initial_program,
)


class CandidateProgramTests(unittest.TestCase):
    def test_literal_candidate_is_read(self) -> None:
        source = "CANDIDATE = {'label': 'x', 'mutation_ids': ['m001']}\n"
        self.assertEqual(
            extract_candidate_payload(source),
            {"label": "x", "mutation_ids": ["m001"]},
        )

    def test_candidate_code_is_not_executed(self) -> None:
        with self.assertRaises(CandidateProgramError):
            extract_candidate_payload("CANDIDATE = dangerous_call()\n")

    def test_exactly_one_assignment_is_required(self) -> None:
        with self.assertRaises(CandidateProgramError):
            extract_candidate_payload("CANDIDATE = {}\nCANDIDATE = {}\n")

    def test_rendered_program_round_trips(self) -> None:
        source = render_initial_program(
            [{"mutation_id": "m000", "family": "mapping", "stage": "S"}],
            min_mutations=1,
            max_mutations=2,
        )
        self.assertEqual(extract_candidate_payload(source)["mutation_ids"], [])
        self.assertIn("EVOLVE-BLOCK-START", source)

    def test_search_catalog_does_not_leak_characterization_labels(self) -> None:
        source = render_initial_program(
            [
                {
                    "mutation_id": "m000",
                    "knob_id": "mapping.loop_order::S",
                    "knob_spec": "mapping.loop_order::S=[M,N]",
                    "family": "mapping",
                    "stage": "S",
                    "expected_legality": "legal",
                    "objective_direction": "improving",
                }
            ],
            min_mutations=1,
            max_mutations=2,
        )
        self.assertIn("knob_spec", source)
        self.assertNotIn("expected_legality", source)
        self.assertNotIn("objective_direction", source)


if __name__ == "__main__":
    unittest.main()
