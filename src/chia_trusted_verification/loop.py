from __future__ import annotations

from pathlib import Path

from .chia.evaluator import TrustedVerificationEvaluator
from .config import ExperimentConfig
from .core import get_backend
from .results import RunRecorder, reconcile_search_result
from .search import build_search_strategy
from .types import PipelineResult, SearchProblem, SearchResult


def build_search_problem(config: ExperimentConfig) -> tuple[SearchProblem, Path]:
    backend = get_backend(config.backend_name)
    search_settings = dict(config.backend_settings)
    search_settings["_verification_intent"] = config.intent.to_dict()
    problem = backend.build_search_problem(search_settings)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / problem.file_name
    path.write_text(problem.initial_source, encoding="utf-8")
    return problem, path


def build_initial_program(config: ExperimentConfig) -> tuple[Path, str]:
    problem, path = build_search_problem(config)
    return path, problem.initial_source


def run_chia_loop(config: ExperimentConfig) -> SearchResult:
    problem, initial_program = build_search_problem(config)
    recorder = RunRecorder(config, problem)
    evaluator = TrustedVerificationEvaluator(
        backend_name=config.backend_name,
        backend_settings=config.backend_settings,
        backend_resources=config.backend_resources,
        intent=config.intent,
        engine=config.engine,
        initial_source=problem.initial_source,
        result_log=config.result_log,
        run_metadata=recorder.evaluator_metadata(),
        use_chia=True,
    )
    try:
        evaluator.evaluate_source(problem.initial_source, role="seed_baseline")
        strategy = build_search_strategy(config.engine)
        result = strategy.run(
            problem=problem,
            initial_program=initial_program,
            evaluator=evaluator,
            config=config,
        )
        result = reconcile_search_result(
            result,
            result_log=config.result_log,
            run_instance_id=str(recorder.metadata["run_instance_id"]),
        )
        recorder.finish(result)
        return result
    except BaseException as exc:
        recorder.fail(exc)
        raise


def evaluate_program_locally(
    program: str | Path,
    config: ExperimentConfig,
) -> PipelineResult:
    problem, _ = build_search_problem(config)
    evaluator = TrustedVerificationEvaluator(
        backend_name=config.backend_name,
        backend_settings=config.backend_settings,
        backend_resources=config.backend_resources,
        intent=config.intent,
        engine=config.engine,
        initial_source=problem.initial_source,
        result_log=config.result_log,
        run_metadata={
            "experiment_name": config.experiment_name,
            "run_id": config.run_id,
            "kernel": config.backend_settings.get("kernel_label"),
            "search_strategy": config.engine,
            "seed": config.seed,
            "model": config.model,
        },
        use_chia=False,
    )
    source = Path(program).read_text(encoding="utf-8")
    return evaluator.evaluate_source(source, role="local_evaluation")
