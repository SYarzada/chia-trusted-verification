from __future__ import annotations

from pathlib import Path

import yaml

from experiments.run_teaal_experiments import _base_model_name, _models_for


ROOT = Path(__file__).resolve().parents[1]


def test_matched_model_sweep_derives_provider_aliases():
    run_settings = {"model_sweep": "gemini-2.5-flash", "models": {}}
    assert _models_for("openevolve_structured", run_settings) == ["google/gemini-2.5-flash"]
    assert _models_for("openevolve_guided", run_settings) == ["google/gemini-2.5-flash"]
    assert _models_for("alphaevolve_structured", run_settings) == ["gemini-2.5-flash"]
    assert _models_for("alphaevolve_guided", run_settings) == ["gemini-2.5-flash"]
    assert _models_for("llm_agent_structured", run_settings) == ["google-vertex/gemini-2.5-flash"]
    assert _models_for("llm_agent_guided", run_settings) == ["google-vertex/gemini-2.5-flash"]


def test_non_llm_baselines_ignore_model_sweep():
    run_settings = {
        "model_sweep": "gemini-2.5-flash",
        "models": {"structured_random": [None], "structured_walk": [None]},
    }
    assert _models_for("structured_random", run_settings) == [None]
    assert _models_for("structured_walk", run_settings) == [None]


def test_base_model_name_removes_provider_prefixes():
    assert _base_model_name("google/gemini-2.5-flash") == "gemini-2.5-flash"
    assert _base_model_name("google-vertex/gemini-2.5-flash") == "gemini-2.5-flash"
    assert _base_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


def test_checked_in_experiments_is_gemini_35_flash_and_five_seeds():
    raw = yaml.safe_load((ROOT / "configs" / "teaal_experiments.yaml").read_text())
    run_settings = raw["experiment_set"]
    assert run_settings["model_sweep"] == "gemini-3.5-flash"
    assert run_settings["iterations"] == 30
    assert run_settings["seeds"] == [0, 1, 2, 3, 4]
    assert len(raw["kernels"]) == 8
