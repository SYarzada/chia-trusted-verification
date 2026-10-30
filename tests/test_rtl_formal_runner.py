from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.configure_gemmini_generated_rtl import MODULE_CONFIG_NAMES, MODULE_FILE_SETS, write_module_configs
from chia_trusted_verification.backends.rtl_formal.formal_runner import prepare_formal_workspace, run_formal
from chia_trusted_verification.backends.rtl_formal.intents import get_intent_spec


ROOT = Path(__file__).resolve().parents[1]


class RtlFormalRunnerTests(unittest.TestCase):
    def test_local_module_configs_keep_each_rtl_source_set(self) -> None:
        base = yaml.safe_load((ROOT / "configs/rtl_formal_gemmini.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "generated-src"
            generated.mkdir()
            for name in {item for files in MODULE_FILE_SETS.values() for item in files}:
                (generated / name).write_text(f"module {Path(name).stem}; endmodule\n")
            self.assertEqual(write_module_configs(generated, base, Path(tmp)), [])
            for module, filenames in MODULE_FILE_SETS.items():
                config = yaml.safe_load((Path(tmp) / MODULE_CONFIG_NAMES[module]).read_text())
                settings = config["backend"]["settings"]
                self.assertEqual([Path(path).name for path in settings["rtl_files"]], filenames)
                self.assertEqual(settings["formal_timeout_s"], 60)

    def test_prepare_instruments_only_target_module(self) -> None:
        spec = get_intent_spec(ROOT / "rtl_formal/smoke/intents.yaml", "smoke_queue_issue")
        candidate = """
        always @(posedge clock) begin
          if (!reset) begin
            assert(!issue_valid || (allocated && !issued && deps == 0));
            cover(issue_valid);
          end
        end
        """
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_formal_workspace(
                candidate_source=candidate,
                spec=spec,
                settings={
                    "_config_dir": str(ROOT / "configs"),
                    "workspace_root": tmp,
                    "rtl_files": ["../rtl_formal/smoke/gemmini_formal_smoke.sv"],
                    "target_file": "../rtl_formal/smoke/gemmini_formal_smoke.sv",
                    "top_module": "GemminiFormalSmoke",
                },
            )
            text = Path(prepared["instrumented_file"]).read_text()
        self.assertIn("assert(!issue_valid", text)
        self.assertNotIn("CHIA_FORMAL_COLLATERAL", text)
        self.assertEqual(prepared["top_module"], "GemminiFormalSmoke")
        self.assertIn("FORMAL", prepared["defines"])
        self.assertIn("SYNTHESIS", prepared["defines"])

    def test_prepare_keeps_trusted_harness_separate_from_agent_collateral(self) -> None:
        spec = get_intent_spec(ROOT / "rtl_formal/smoke/intents.yaml", "smoke_queue_issue")
        candidate = "always @(posedge clock) begin assert(!issue_valid || allocated); cover(issue_valid); end\n"
        with tempfile.TemporaryDirectory() as tmp:
            trusted = Path(tmp) / "trusted.sv"
            trusted.write_text("always @(posedge clock) if ($initstate) assume(reset);\n")
            prepared = prepare_formal_workspace(
                candidate_source=candidate,
                spec=spec,
                settings={
                    "_config_dir": str(ROOT / "configs"),
                    "workspace_root": str(Path(tmp) / "workspaces"),
                    "rtl_files": ["../rtl_formal/smoke/gemmini_formal_smoke.sv"],
                    "target_file": "../rtl_formal/smoke/gemmini_formal_smoke.sv",
                    "top_module": "GemminiFormalSmoke",
                    "trusted_collateral_files": [str(trusted)],
                },
            )
            text = Path(prepared["instrumented_file"]).read_text()
        self.assertIn("CHIA-BEGIN-TRUSTED-HARNESS", text)
        self.assertIn("assume(reset)", text)
        self.assertIn("assert(!issue_valid", text)
        self.assertEqual(prepared["trusted_collateral_files"], [str(trusted.resolve())])

    def test_direct_smtbmc_command_path_with_fake_tools(self) -> None:
        spec = get_intent_spec(ROOT / "rtl_formal/smoke/intents.yaml", "smoke_queue_issue")
        candidate = (ROOT / "rtl_formal/smoke/known_good_collateral.sv").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            yosys = bin_dir / "yosys"
            smtbmc = bin_dir / "yosys-smtbmc"
            z3 = bin_dir / "z3"
            yosys.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' "
                "'; yosys-smt2-assert 0 fake_assert' "
                "'; yosys-smt2-cover 0 fake_cover' "
                "> design.smt2\n"
                "exit 0\n"
            )
            smtbmc.write_text("#!/bin/sh\nexit 0\n")
            z3.write_text("#!/bin/sh\nexit 0\n")
            for executable in (yosys, smtbmc, z3):
                executable.chmod(0o755)

            settings = {
                "_config_dir": str(ROOT / "configs"),
                "workspace_root": str(tmp_path / "workspaces"),
                "rtl_files": ["../rtl_formal/smoke/gemmini_formal_smoke.sv"],
                "target_file": "../rtl_formal/smoke/gemmini_formal_smoke.sv",
                "top_module": "GemminiFormalSmoke",
                "yosys_bin": str(yosys),
                "yosys_smtbmc_bin": str(smtbmc),
                "solver": str(z3),
                "formal_engine": "smtbmc",
                "proof_depth": 8,
                "formal_timeout_s": 10,
            }
            prepared = prepare_formal_workspace(
                candidate_source=candidate,
                spec=spec,
                settings=settings,
            )
            result = run_formal(prepared, spec, settings)
        self.assertEqual(result["status"], "proved")
        self.assertEqual(result["compile"]["returncode"], 0)
        self.assertEqual(result["bmc"]["returncode"], 0)
        self.assertEqual(result["induction"]["returncode"], 0)
        self.assertEqual(result["cover"]["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
