#!/usr/bin/env python3
"""Choose legal starting designs and rebuild TeAAL mutation lists.

The final experiment should not cherry-pick stochastic random seeds.  Instead,
this script deterministically inspects the bounded, curated TeAAL completion
space for each kernel.  If the original legal seed already has meaningful
objective headroom, it is preserved.  Otherwise the script selects a legal,
moderately suboptimal design that still has a legal one-step improvement after
rebuilding the curated mutation catalog around that design.

Outputs:
  teaal_evolve/example/kernel/nki/selected_seeds/*.yaml
  results/selected_seed_results/selected_seed_summary.csv
  results/selected_seed_results/selected_seed_manifest.json
  results/selected_seed_results/rebased_catalog_characterization.csv
  configs/teaal_selected_seeds_gemini35flash.yaml
  configs/teaal_selected_seeds_smoke.yaml

The search-space characterization uses the same Candidate, mutation, Full
checker, and objective implementations as the trusted backend itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from chia_trusted_verification.backends.teaal_evolve.worker import (
    _build_catalog,
    _load_api,
    _seed_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = ROOT / "configs" / "teaal_experiments.yaml"
DEFAULT_OUT = ROOT / "results" / "selected_seed_results"
DEFAULT_SELECTED_DIR = ROOT / "teaal_evolve" / "example" / "kernel" / "nki" / "selected_seeds"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping")
    return dict(value)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _objective_value(
    api: Mapping[str, Any],
    candidate: Any,
    *,
    knob_space: str,
    objective_backend: str,
    objective_shapes: Mapping[str, int] | None,
) -> tuple[bool, float | None, str]:
    checker = api["check_candidate_with_backend"](
        candidate,
        backend_name="full",
        include_bottleneck_objective=False,
        knob_space=knob_space,
    )
    if not checker.candidate_report.is_legal:
        return False, None, "illegal"
    objective = api["compute_objective_with_backend"](
        candidate,
        backend_name=objective_backend,
        shapes=dict(objective_shapes or {}) or None,
    )
    if objective.status != "ok" or objective.value is None:
        return True, None, f"objective_{objective.status}"
    return True, float(objective.value), "ok"


def _compatible_combinations(
    catalog: Sequence[Mapping[str, Any]],
    max_mutations: int,
) -> Iterable[tuple[Mapping[str, Any], ...]]:
    for size in range(1, max_mutations + 1):
        for combo in itertools.combinations(catalog, size):
            knob_ids = [str(row["knob_id"]) for row in combo]
            if len(knob_ids) != len(set(knob_ids)):
                continue
            yield combo


def _apply(api: Mapping[str, Any], seed: Any, combo: Sequence[Mapping[str, Any]], *, knob_space: str, label: str):
    return api["apply_knob_specs_atomic"](
        seed,
        [str(row["knob_spec"]) for row in combo],
        label=label,
        knob_space=knob_space,
    )


def _kernel_only_yaml(original_kernel: Path, candidate: Any) -> dict[str, Any]:
    """Serialize only sections originally owned by the kernel file.

    Candidate.from_files merges kernel + architecture.  Writing that merged
    object as a new kernel would duplicate architecture data when the normal
    architecture file is loaded again.  Keep exactly the top-level sections
    that were present in the original kernel and replace their values with the
    candidate's mutated values.
    """
    original = _load_yaml(original_kernel)
    result: dict[str, Any] = {}
    for key in original:
        if key not in candidate.spec_dict:
            raise KeyError(f"mutated candidate lost kernel section {key!r}")
        result[key] = candidate.spec_dict[key]
    return result


def _worker_config(
    experiment_path: Path,
    run_set: Mapping[str, Any],
    kernel_name: str,
    kernel_entry: Mapping[str, Any],
) -> dict[str, Any]:
    config_dir = experiment_path.parent
    set_config = dict(run_set.get("experiment_set", run_set.get("campaign", {})))
    experiment_path = _resolve(
        config_dir,
        str(set_config.get("base_experiment_config", "experiment.yaml")),
    )
    experiment = _load_yaml(experiment_path)
    backend = dict(experiment.get("backend", {}))
    settings = dict(backend.get("settings", {}))
    config_base = experiment_path.parent

    def setting_path(name: str, default: str) -> str:
        return str(_resolve(config_base, str(settings.get(name, default))))

    return {
        "teaal_root": setting_path("root", "../teaal_evolve"),
        "kernel_label": kernel_name,
        "kernel_file": str(_resolve(config_dir, str(kernel_entry["kernel_file"]))),
        "architecture_file": setting_path(
            "architecture_file", "../teaal_evolve/example/arch/trainium1.yaml"
        ),
        "checker_backend": str(settings.get("checker_backend", "hierarchical")),
        "objective_backend": str(settings.get("objective_backend", "monolithic_bottleneck")),
        "knob_space": str(settings.get("knob_space", "rich")),
        "min_mutations": int(settings.get("min_mutations", 1)),
        "max_mutations": int(settings.get("max_mutations", 2)),
        "catalog_seed": int(settings.get("catalog_seed", 0)),
        "objective_shapes": dict(
            kernel_entry.get("objective_shapes", settings.get("objective_shapes", {})) or {}
        ),
        "node_timeout_s": int(settings.get("node_timeout_s", 300)),
    }


def _rebase_characterization(
    *,
    api: Mapping[str, Any],
    selected_candidate: Any,
    selected_kernel_yaml: Path,
    worker: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_worker = dict(worker)
    selected_worker["kernel_file"] = str(selected_kernel_yaml.resolve())
    rebased_seed = _seed_candidate(api, selected_worker)
    catalog = _build_catalog(api, selected_worker)
    knob_space = str(worker["knob_space"])
    objective_backend = str(worker["objective_backend"])
    shapes = dict(worker.get("objective_shapes", {}))

    legal, seed_obj, seed_status = _objective_value(
        api,
        rebased_seed,
        knob_space=knob_space,
        objective_backend=objective_backend,
        objective_shapes=shapes,
    )
    if not legal or seed_obj is None:
        raise RuntimeError(
            f"selected starting design failed after serialization: {seed_status}"
        )

    rows: list[dict[str, Any]] = []
    improving: list[dict[str, Any]] = []
    for row in catalog:
        mutation_id = str(row["mutation_id"])
        try:
            child = api["apply_knob_specs_atomic"](
                rebased_seed,
                [str(row["knob_spec"])],
                label=f"rebase-{mutation_id}",
                knob_space=knob_space,
            )
            child_legal, child_obj, status = _objective_value(
                api,
                child,
                knob_space=knob_space,
                objective_backend=objective_backend,
                objective_shapes=shapes,
            )
        except Exception as exc:
            child_legal, child_obj, status = False, None, f"error:{type(exc).__name__}:{exc}"
        rel = None
        if child_obj is not None and seed_obj > 0:
            rel = (seed_obj - child_obj) / seed_obj
        out = {
            **dict(row),
            "selected_seed_objective": seed_obj,
            "legal": bool(child_legal),
            "objective_status": status,
            "objective_value": child_obj,
            "relative_improvement_from_selected_seed": rel,
        }
        rows.append(out)
        if child_legal and child_obj is not None and child_obj < seed_obj:
            improving.append(out)
    return rows, improving


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_final_configs(
    experiment_path: Path,
    raw: Mapping[str, Any],
    selected_kernel_paths: Mapping[str, Path],
) -> tuple[Path, Path]:
    final = json.loads(json.dumps(raw))
    cfg = final.setdefault("experiment_set" if "experiment_set" in final else "campaign", {})
    cfg["name"] = "teaal_chia_challenge_{model_slug}"
    cfg["output_root"] = "../runs/teaal_chia_challenge_{model_slug}"
    cfg["iterations"] = 30
    cfg["seeds"] = [0, 1, 2, 3, 4]
    cfg["engines"] = [
        "structured_random",
        "structured_walk",
        "openevolve_structured",
        "openevolve_guided",
        "alphaevolve_structured",
        "alphaevolve_guided",
        "llm_agent_structured",
        "llm_agent_guided",
    ]
    cfg["model_sweep"] = "gemini-3.5-flash"

    config_dir = experiment_path.parent
    final_kernels = final.setdefault("kernels", {})
    for kernel, path in selected_kernel_paths.items():
        entry = dict(final_kernels[kernel])
        entry["kernel_file"] = str(path.resolve().relative_to(ROOT)).replace("\\", "/")
        # Config paths are resolved relative to configs/, so prefix ../.
        entry["kernel_file"] = "../" + entry["kernel_file"]
        final_kernels[kernel] = entry

    final_path = config_dir / "teaal_selected_seeds_gemini35flash.yaml"
    final_path.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")

    smoke = json.loads(json.dumps(final))
    smoke["experiment_set" if "experiment_set" in smoke else "campaign"]["name"] = "teaal_chia_challenge_smoke_{model_slug}"
    smoke["experiment_set" if "experiment_set" in smoke else "campaign"]["output_root"] = "../runs/teaal_chia_challenge_smoke_{model_slug}"
    smoke["experiment_set" if "experiment_set" in smoke else "campaign"]["engines"] = ["alphaevolve_structured"]
    smoke["experiment_set" if "experiment_set" in smoke else "campaign"]["seeds"] = [0]
    # Attention already has useful objective headroom in the original artifact
    # and exercises a nontrivial kernel while remaining a single smoke run.
    smoke["kernels"] = {"attention": smoke["kernels"]["attention"]}
    smoke_path = config_dir / "teaal_selected_seeds_smoke.yaml"
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
    return final_path, smoke_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", "--campaign", dest="experiments", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--selected-dir", "--challenge-dir", dest="selected_dir", default=str(DEFAULT_SELECTED_DIR))
    parser.add_argument(
        "--keep-headroom",
        type=float,
        default=0.05,
        help="Keep the original seed when observed bounded-space headroom is at least this fraction.",
    )
    parser.add_argument(
        "--target-headroom",
        type=float,
        default=0.12,
        help="Target recoverable improvement for a deliberately suboptimal starting design.",
    )
    parser.add_argument("--min-challenge-headroom", type=float, default=0.05)
    parser.add_argument("--max-challenge-headroom", type=float, default=0.30)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the fixed starting-design plan instead of exhaustive 1-2 mutation enumeration.",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiments).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    selected_dir = Path(args.selected_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_yaml(experiment_path)
    kernels = dict(raw.get("kernels", {}))
    if not kernels:
        raise SystemExit("experiment list defines no kernels")

    # Fixed starting-design choices for the final experiment.  These choices
    # were fixed before the final model sweep and are validated below using
    # the same Full checker and objective as normal candidates.
    fast_plan = {
        "unary": [],
        "unary_reduce_one_tile": [],
        "vector_scalar_overlap": [],
        "attention": [],
        "one_pass_attention": ["m000", "m002"],
        "one_pass_attention_no_pipelining": ["m000", "m002"],
        "matmul": [],
        "mamba": [],
    }

    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    all_rebase_rows: list[dict[str, Any]] = []
    selected_paths: dict[str, Path] = {}

    for kernel_name, kernel_entry_any in kernels.items():
        if not isinstance(kernel_entry_any, Mapping):
            raise TypeError(f"kernel {kernel_name!r} must be a mapping")
        kernel_entry = dict(kernel_entry_any)
        worker = _worker_config(experiment_path, raw, kernel_name, kernel_entry)
        api = _load_api(worker)
        seed = _seed_candidate(api, worker)
        catalog = _build_catalog(api, worker)
        knob_space = str(worker["knob_space"])
        objective_backend = str(worker["objective_backend"])
        shapes = dict(worker.get("objective_shapes", {}))

        seed_legal, seed_obj, seed_status = _objective_value(
            api,
            seed,
            knob_space=knob_space,
            objective_backend=objective_backend,
            objective_shapes=shapes,
        )
        if not seed_legal or seed_obj is None:
            raise RuntimeError(f"original seed for {kernel_name} is not objective-scored legal: {seed_status}")

        evaluated: list[dict[str, Any]] = [
            {
                "mutation_ids": [],
                "candidate": seed,
                "objective": seed_obj,
                "legal": True,
            }
        ]

        if args.fast:
            requested_ids = fast_plan.get(kernel_name, [])
            if requested_ids:
                by_id = {str(row["mutation_id"]): row for row in catalog}
                missing = [mid for mid in requested_ids if mid not in by_id]
                if missing:
                    raise RuntimeError(
                        f"{kernel_name}: predetermined mutation IDs not in catalog: {missing}"
                    )

                combo = [by_id[mid] for mid in requested_ids]
                candidate = _apply(
                    api,
                    seed,
                    combo,
                    knob_space=knob_space,
                    label=f"challenge-fast-{'-'.join(requested_ids)}",
                )
                legal, obj, status = _objective_value(
                    api,
                    candidate,
                    knob_space=knob_space,
                    objective_backend=objective_backend,
                    objective_shapes=shapes,
                )
                if not legal or obj is None:
                    raise RuntimeError(
                        f"{kernel_name}: predetermined challenge is not objective-scored legal: {status}"
                    )
                evaluated.append(
                    {
                        "mutation_ids": requested_ids,
                        "candidate": candidate,
                        "objective": obj,
                        "legal": True,
                    }
                )

            # In fast mode the original headroom for kernels kept unchanged is
            # supplied by the already-characterized final search space.
            known_headroom = {
                "unary": 0.0,
                "unary_reduce_one_tile": 0.0,
                "vector_scalar_overlap": 0.11111,
                "attention": 0.15923,
                "matmul": 0.0,
            }
            if requested_ids:
                candidate_obj = float(evaluated[-1]["objective"])
                best_obj = seed_obj
                original_headroom = (
                    (seed_obj - best_obj) / seed_obj if seed_obj > 0 else 0.0
                )
            else:
                best_obj = seed_obj * (1.0 - known_headroom.get(kernel_name, 0.0))
                original_headroom = known_headroom.get(kernel_name, 0.0)

        else:
            seen_sha = {_sha(seed.spec_dict)}
            for combo in _compatible_combinations(catalog, int(worker["max_mutations"])):
                ids = [str(row["mutation_id"]) for row in combo]
                try:
                    candidate = _apply(
                        api,
                        seed,
                        combo,
                        knob_space=knob_space,
                        label=f"challenge-enum-{'-'.join(ids)}",
                    )
                except Exception:
                    continue
                candidate_sha = _sha(candidate.spec_dict)
                if candidate_sha in seen_sha:
                    continue
                seen_sha.add(candidate_sha)
                legal, obj, status = _objective_value(
                    api,
                    candidate,
                    knob_space=knob_space,
                    objective_backend=objective_backend,
                    objective_shapes=shapes,
                )
                if legal and obj is not None:
                    evaluated.append(
                        {
                            "mutation_ids": ids,
                            "candidate": candidate,
                            "objective": obj,
                            "legal": True,
                        }
                    )

            best_obj = min(float(row["objective"]) for row in evaluated)
            original_headroom = (
                (seed_obj - best_obj) / seed_obj if seed_obj > 0 else 0.0
            )

        original_kernel_path = Path(str(worker["kernel_file"])).resolve()
        selected = evaluated[0]
        reason = "original_seed_has_headroom"

        if args.fast:
            requested_ids = fast_plan.get(kernel_name, [])
            if requested_ids:
                selected = evaluated[-1]
                reason = "predetermined_validated_challenge"
            elif original_headroom < args.keep_headroom:
                reason = "no_reachable_moderate_challenge_keep_original"

        elif original_headroom < args.keep_headroom:
            candidates = []
            for row in evaluated[1:]:
                obj = float(row["objective"])
                if obj <= seed_obj:
                    continue
                headroom = (obj - best_obj) / obj if obj > 0 else 0.0
                if not (args.min_challenge_headroom <= headroom <= args.max_challenge_headroom):
                    continue
                candidates.append((abs(headroom - args.target_headroom), len(row["mutation_ids"]), tuple(row["mutation_ids"]), row))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))

            # Try candidates in deterministic target-distance order.  A valid
            # The selected design must remain legal after saving/loading and have
            # at least one legal improving one-step mutation from that new seed.
            reason = "no_reachable_moderate_challenge_keep_original"
            for _, _, _, row in candidates:
                candidate = row["candidate"]
                temp_path = selected_dir / f".{kernel_name}.candidate.tmp.yaml"
                temp_payload = _kernel_only_yaml(original_kernel_path, candidate)
                temp_path.write_text(yaml.safe_dump(temp_payload, sort_keys=False), encoding="utf-8")
                try:
                    rebase_rows, improving = _rebase_characterization(
                        api=api,
                        selected_candidate=candidate,
                        selected_kernel_yaml=temp_path,
                        worker=worker,
                    )
                finally:
                    temp_path.unlink(missing_ok=True)
                if improving:
                    selected = row
                    reason = "deterministic_moderately_suboptimal_challenge"
                    break

        selected_candidate = selected["candidate"]
        selected_obj = float(selected["objective"])
        selected_yaml = selected_dir / f"{kernel_name}.yaml"
        selected_yaml.write_text(
            yaml.safe_dump(_kernel_only_yaml(original_kernel_path, selected_candidate), sort_keys=False),
            encoding="utf-8",
        )
        selected_paths[kernel_name] = selected_yaml

        rebase_rows, improving = _rebase_characterization(
            api=api,
            selected_candidate=selected_candidate,
            selected_kernel_yaml=selected_yaml,
            worker=worker,
        )
        for row in rebase_rows:
            all_rebase_rows.append({"kernel": kernel_name, **row})
        best_rebased = min(
            [selected_obj]
            + [float(row["objective_value"]) for row in improving if row.get("objective_value") is not None]
        )
        selected_headroom = (
            (selected_obj - best_rebased) / selected_obj if selected_obj > 0 else 0.0
        )

        summary = {
            "kernel": kernel_name,
            "selection_reason": reason,
            "original_seed_objective": seed_obj,
            "best_objective_observed_from_original_bounded_space": best_obj,
            "original_seed_progress_headroom": original_headroom,
            "selected_seed_objective": selected_obj,
            "selected_seed_best_reachable_single_objective": best_rebased,
            "selected_seed_single_step_progress_headroom": selected_headroom,
            "challenge_mutation_ids_from_original": json.dumps(selected["mutation_ids"]),
            "original_catalog_size": len(catalog),
            "rebased_catalog_size": len(rebase_rows),
            "rebased_legal_single_mutations": sum(bool(row["legal"]) for row in rebase_rows),
            "rebased_improving_single_mutations": len(improving),
            "selected_kernel_file": str(selected_yaml),
        }
        summary_rows.append(summary)
        manifest_rows.append(
            {
                **summary,
                "selected_candidate_sha256": _sha(selected_candidate.spec_dict),
                "original_kernel_file": str(original_kernel_path),
                "architecture_file": str(worker["architecture_file"]),
                "knob_space": knob_space,
                "objective_backend": objective_backend,
            }
        )
        print(
            f"{kernel_name:36s} {reason:48s} "
            f"old_headroom={100*original_headroom:6.2f}% "
            f"new_headroom={100*selected_headroom:6.2f}% "
            f"mutations={selected['mutation_ids']}"
        )

    _write_csv(out_dir / "selected_seed_summary.csv", summary_rows)
    _write_csv(out_dir / "rebased_catalog_characterization.csv", all_rebase_rows)
    (out_dir / "selected_seed_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_path, smoke_path = _write_final_configs(experiment_path, raw, selected_paths)

    print(f"wrote: {out_dir / 'selected_seed_summary.csv'}")
    print(f"wrote: {out_dir / 'rebased_catalog_characterization.csv'}")
    print(f"wrote: {out_dir / 'selected_seed_manifest.json'}")
    print(f"wrote: {final_path}")
    print(f"wrote: {smoke_path}")
    print("final experiment set: 8 kernels x 8 methods x 5 seeds = 320 runs")


if __name__ == "__main__":
    main()
