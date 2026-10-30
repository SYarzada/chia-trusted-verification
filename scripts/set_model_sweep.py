
#!/usr/bin/env python3
"""Set the Gemini model for TeAAL-Evolve experiments.

Updates model names and pricing settings in the experiment and search configs.
Does not change kernels, seeds, search methods, or proposal counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = ROOT / "configs" / "teaal_experiments.yaml"

# Token prices in USD per million input and output tokens.
VERTEX_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# Combined Gemini and AlphaEvolve agent token pricing.
ALPHAEVOLVE_TOTAL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (4.50, 27.00),
    "gemini-3.1-pro-preview": (6.00, 36.00),
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Gemini model, e.g. gemini-2.5-flash")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENTS))
    args = parser.parse_args()

    model = str(args.model).strip()
    for prefix in ("google-vertex/", "google/", "gemini/"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break
    if not model.startswith("gemini-"):
        raise SystemExit(f"Expected a Gemini model ID. Got {model!r}")

    config_path = Path(args.config).expanduser().resolve()
    run_set = _load(config_path)
    run_set.setdefault("experiment_set" if "experiment_set" in run_set else "campaign", {})["model_sweep"] = model

    costs = run_set.setdefault("cost_tracking", {})
    by_model = costs.setdefault("by_model", {})

    if model in VERTEX_PRICING:
        inp, out = VERTEX_PRICING[model]
        record = {
            "input_per_million_tokens_usd": inp,
            "output_per_million_tokens_usd": out,
        }
        by_model[f"google/{model}"] = dict(record)
        by_model[f"google-vertex/{model}"] = dict(record)

    if model in ALPHAEVOLVE_TOTAL_PRICING:
        inp, out = ALPHAEVOLVE_TOTAL_PRICING[model]
        by_model[model] = {
            "input_per_million_tokens_usd": inp,
            "output_per_million_tokens_usd": out,
            "note": "Includes Gemini and AlphaEvolve agent token pricing.",
        }
    else:
        by_model[model] = {
            "input_per_million_tokens_usd": None,
            "output_per_million_tokens_usd": None,
            "note": (
                "Token usage is saved. Use Cloud Billing to confirm "
                "the final AlphaEvolve cost."
            ),
        }

    _write(config_path, run_set)

    openevolve = _load(ROOT / "configs" / "search" / "openevolve.yaml")
    llm = openevolve.setdefault("llm", {})
    existing = llm.get("models") or [{}]
    base = dict(existing[0]) if isinstance(existing[0], dict) else {}
    base.update({"name": f"google/{model}", "weight": 1.0})
    llm["models"] = [base]
    _write(ROOT / "configs" / "search" / "openevolve.yaml", openevolve)

    alpha = _load(ROOT / "configs" / "search" / "alphaevolve.yaml")
    generation = alpha.setdefault("alphaevolve", {}).setdefault("generation_settings", {})
    generation["models"] = [{"name": model, "weight": 1.0}]
    _write(ROOT / "configs" / "search" / "alphaevolve.yaml", alpha)

    agent = _load(ROOT / "configs" / "search" / "llm_agent.yaml")
    agent.setdefault("agent", {})["model"] = f"google-vertex/{model}"
    _write(ROOT / "configs" / "search" / "llm_agent.yaml", agent)

    print(f"Configured model sweep: {model}")
    print(f"  OpenEvolve: google/{model}")
    print(f"  AlphaEvolve: {model}")
    print(f"  CHIA LLM agent: google-vertex/{model}")


if __name__ == "__main__":
    main()
