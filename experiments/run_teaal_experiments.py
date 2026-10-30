from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from chia_trusted_verification.backends.teaal_evolve.replay import replay_run
from chia_trusted_verification.config import load_experiment_config
from chia_trusted_verification.loop import run_chia_loop


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_") or "value"


def _resolve(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("experiment list must be a YAML mapping")
    return dict(raw)


_AGENTIC_ENGINES = frozenset({
    "openevolve_structured",
    "openevolve_guided",
    "alphaevolve_structured",
    "alphaevolve_guided",
    "llm_agent_structured",
    "llm_agent_guided",
})


def _base_model_name(value: str) -> str:
    model = str(value).strip()
    for prefix in ("google-vertex/", "google/", "gemini/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    if not model:
        raise ValueError("model_sweep must be non-empty")
    return model


def _model_alias(engine: str, selected_model: str) -> str:
    if engine.startswith("openevolve_"):
        return f"google/{selected_model}"
    if engine.startswith("alphaevolve_"):
        return selected_model
    if engine.startswith("llm_agent_"):
        return f"google-vertex/{selected_model}"
    raise ValueError(f"no model alias is defined for engine {engine!r}")


def _models_for(engine: str, run_settings: Mapping[str, Any]) -> list[str | None]:
    sweep = run_settings.get("model_sweep")
    if engine in _AGENTIC_ENGINES and sweep is not None:
        return [_model_alias(engine, _base_model_name(str(sweep)))]

    value = dict(run_settings.get("models", {})).get(engine)
    if value is None:
        return [None]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [None if item is None else str(item) for item in value]
    raise TypeError(f"models.{engine} must be a string or list")


def _cost_for(engine: str, model: str | None, raw: Mapping[str, Any]) -> dict[str, Any]:
    common = dict(raw.get("cost_tracking", {}).get("default", {}))
    by_engine = dict(raw.get("cost_tracking", {}).get("by_engine", {}))
    if isinstance(by_engine.get(engine), Mapping):
        common.update(dict(by_engine[engine]))
    by_model = dict(raw.get("cost_tracking", {}).get("by_model", {}))
    if model is not None and isinstance(by_model.get(model), Mapping):
        common.update(dict(by_model[model]))
    return common


def build_runs(experiment_path: Path, *, engine_filter: set[str], kernel_filter: set[str]):
    raw = _load(experiment_path)
    run_settings = dict(raw.get("experiment_set", raw.get("campaign", {})))
    base = experiment_path.parent
    base_config = _resolve(base, str(run_settings.get("base_experiment_config", "experiment.yaml")))
    sweep = run_settings.get("model_sweep")
    selected_model = _base_model_name(str(sweep)) if sweep is not None else None
    model_slug = _slug(selected_model or "mixed_models").replace("-", "").replace(".", "")
    name_template = str(run_settings.get("name", "teaal_chia_experiments"))
    output_template = str(run_settings.get("output_root", "../runs/teaal_experiments"))
    name = name_template.format(model=selected_model or "mixed_models", model_slug=model_slug)
    output_root = Path(_resolve(base, output_template.format(model=selected_model or "mixed_models", model_slug=model_slug)))
    iterations = int(run_settings.get("iterations", 30))
    seeds = [int(item) for item in run_settings.get("seeds", [0, 1, 2, 3, 4])]
    engines = [str(item) for item in run_settings.get("engines", ["openevolve"])]
    kernels = dict(raw.get("kernels", {}))
    if not kernels:
        raise ValueError("experiment list must define at least one kernel")

    for engine in engines:
        if engine_filter and engine not in engine_filter:
            continue
        for model in _models_for(engine, run_settings):
            for kernel_name, kernel_value in kernels.items():
                if kernel_filter and kernel_name not in kernel_filter:
                    continue
                if not isinstance(kernel_value, Mapping):
                    raise TypeError(f"kernel {kernel_name!r} must be a mapping")
                for seed in seeds:
                    config = load_experiment_config(base_config, engine_override=engine)
                    settings = dict(config.backend_settings)
                    settings["kernel_label"] = str(kernel_name)
                    settings["kernel_file"] = _resolve(base, str(kernel_value["kernel_file"]))
                    if kernel_value.get("objective_shapes") is not None:
                        settings["objective_shapes"] = dict(kernel_value["objective_shapes"])
                    model_slug = _slug(model or "no_model")
                    run_dir = output_root / _slug(engine) / model_slug / _slug(kernel_name) / f"seed_{seed}"
                    run_id = f"{name}-{kernel_name}-{engine}-{model_slug}-seed{seed}"
                    yield replace(
                        config,
                        experiment_name=name,
                        run_id=run_id,
                        seed=seed,
                        iterations=iterations,
                        model=model if model is not None else config.model,
                        output_dir=str(run_dir.resolve()),
                        result_log=str((run_dir / "trusted_results.jsonl").resolve()),
                        backend_settings=settings,
                        cost_tracking=_cost_for(engine, model, raw),
                    )


def _completed(run_dir: Path, *, require_replay: bool) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    search_path = run_dir / "search_result.json"
    if not manifest_path.exists() or not search_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("status") != "completed":
        return False
    if require_replay:
        replay_path = run_dir / "certification_replay.jsonl"
        return replay_path.exists() and replay_path.stat().st_size > 0
    return True


def _shard(runs: list[Any], *, num_shards: int, shard_index: int) -> list[Any]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [config for index, config in enumerate(runs) if index % num_shards == shard_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teaal_experiments.yaml")
    parser.add_argument("--engine", action="append", default=[])
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-certification-replay",
        action="store_true",
        help="Skip matched Full/Hierarchical/Delta-Syntactic/Delta-Rewrite replay after each run.",
    )
    parser.add_argument(
        "--force-certification-replay",
        action="store_true",
        help="Regenerate certification_replay.jsonl even if it already exists.",
    )
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    all_runs = list(
        build_runs(
            path,
            engine_filter=set(args.engine),
            kernel_filter=set(args.kernel),
        )
    )
    runs = _shard(all_runs, num_shards=args.num_shards, shard_index=args.shard_index)
    replay_enabled = not args.skip_certification_replay
    print(
        f"planned runs: {len(all_runs)} total; shard {args.shard_index}/{args.num_shards} "
        f"contains {len(runs)} run(s)"
    )
    for index, config in enumerate(runs, start=1):
        run_dir = Path(config.output_dir)
        print(
            f"[{index}/{len(runs)}] {config.run_id} "
            f"kernel={config.backend_settings.get('kernel_label')} "
            f"engine={config.engine} seed={config.seed} model={config.model}"
        )
        if args.dry_run:
            continue
        if args.resume and _completed(run_dir, require_replay=replay_enabled):
            print("  skip: completed run and certification replay already present")
            continue
        if not _completed(run_dir, require_replay=False):
            run_chia_loop(config)
        elif args.resume:
            print("  search already completed. Only running missing certification replay.")
        else:
            run_chia_loop(config)
        if replay_enabled:
            output = replay_run(run_dir, force=args.force_certification_replay)
            print(f"  certification replay: {output}")


if __name__ == "__main__":
    main()
