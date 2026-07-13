"""Validate and summarize a paired trajectory horizon-stress experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.provenance import validate_artifact_records
from utils import file_sha256, validate_code_revision


METHODS = ("cem", "gc_idm", "larc")
METHOD_LABELS = {"cem": "CEM", "gc_idm": "GC-IDM", "larc": "LARC"}
METHOD_COLORS = {"cem": "#4C78A8", "gc_idm": "#F58518", "larc": "#54A24B"}
REQUIRED_ARTIFACTS = {
    "cem": {
        "latent_cache_metadata",
        "released_lewm_config",
        "released_lewm_weights",
    },
    "gc_idm": {
        "latent_cache_metadata",
        "world_model_config",
        "world_model_weights",
        "world_model_metadata",
        "policy_config",
        "policy_weights",
        "policy_metadata",
    },
    "larc": {
        "latent_cache_metadata",
        "world_model_config",
        "world_model_weights",
        "world_model_metadata",
        "policy_config",
        "policy_weights",
        "policy_metadata",
    },
}
OPEN_LOOP_ARTIFACTS = {
    "latent_cache_metadata",
    "world_model_config",
    "world_model_weights",
    "world_model_metadata",
}


def _nonnegative_number(mapping: dict, key: str) -> float:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{key} must be a finite non-negative number")
    return float(value)


def _integer_list(mapping: dict, key: str, expected_length: int) -> list[int]:
    values = mapping.get(key)
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError(f"{key} must contain {expected_length} non-negative integers")
    return values


def _boolean_list(mapping: dict, key: str) -> list[bool]:
    values = mapping.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError(f"{key} must be a non-empty boolean list")
    return values


def _validate_manifest(path: Path, seed: int) -> tuple[dict, tuple[int, ...], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    goal_offsets = manifest.get("goal_offsets")
    if (
        manifest.get("version") != 2
        or manifest.get("evaluation_seed") != seed
        or not isinstance(goal_offsets, list)
        or not goal_offsets
        or any(type(value) is not int or value < 1 for value in goal_offsets)
        or goal_offsets != sorted(set(goal_offsets))
        or manifest.get("max_goal_offset") != goal_offsets[-1]
    ):
        raise ValueError("paired evaluation manifest has an invalid protocol")
    num_eval = manifest.get("num_eval")
    if type(num_eval) is not int or num_eval < 1:
        raise ValueError("paired evaluation manifest has an invalid num_eval")
    episodes = _integer_list(manifest, "episode_indices", num_eval)
    _integer_list(manifest, "start_steps", num_eval)
    if len(set(episodes)) != num_eval:
        raise ValueError("paired evaluation manifest episodes must be unique")
    return manifest, tuple(goal_offsets), file_sha256(path)


def _load_result(
    path: Path,
    *,
    method: str,
    seed: int,
    goal_offset: int,
    goal_offsets: tuple[int, ...],
    manifest_num_eval: int,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("method") != method:
        raise ValueError(
            f"{path.name} declares method {payload.get('method')!r}, expected {method!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("method") != method:
        raise ValueError(f"{path.name} has an incompatible resolved method")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path.name} has no resolved evaluation protocol")
    if (
        protocol.get("seed") != seed
        or protocol.get("goal_offset") != goal_offset
        or protocol.get("goal_offsets") != list(goal_offsets)
    ):
        raise ValueError(f"{path.name} has an incompatible evaluation protocol")
    if protocol.get("num_eval") != manifest_num_eval:
        raise ValueError(f"{path.name} num_eval disagrees with the paired manifest")
    multiplier = protocol.get("budget_multiplier")
    if (
        type(multiplier) is not int
        or multiplier < 1
        or protocol.get("eval_budget") != multiplier * goal_offset
    ):
        raise ValueError(f"{path.name} has an incompatible evaluation budget")
    configured_manifest = protocol.get("manifest_path")
    if (
        not isinstance(payload.get("manifest"), str)
        or not isinstance(configured_manifest, str)
        or Path(payload["manifest"]).resolve() != manifest_path
        or Path(configured_manifest).resolve() != manifest_path
        or payload.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"{path.name} does not reference the paired manifest")

    validate_code_revision(payload.get("code_revision"))
    if config.get("code_revision") != payload["code_revision"]:
        raise ValueError(f"{path.name} result and config declare different code revisions")
    artifacts = payload.get("artifacts")
    validate_artifact_records(artifacts, verify_files=True)
    missing = REQUIRED_ARTIFACTS[method].difference(artifacts)
    if missing:
        raise ValueError(f"{path.name} is missing evaluation artifacts: {sorted(missing)}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path.name} does not contain metrics")
    successes = _boolean_list(metrics, "episode_successes")
    if protocol.get("num_eval") != len(successes):
        raise ValueError(f"{path.name} num_eval disagrees with episode_successes")
    expected_rate = 100.0 * sum(successes) / len(successes)
    rate = metrics.get("success_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isclose(float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{path.name} success_rate disagrees with episode_successes")

    environment_steps = _integer_list(metrics, "episode_environment_steps", len(successes))
    if any(step < 1 or step > protocol["eval_budget"] for step in environment_steps):
        raise ValueError(f"{path.name} contains an invalid episode environment-step count")
    planner_calls = _integer_list(metrics, "episode_planner_calls", len(successes))
    row_calls = metrics.get("planning_row_calls")
    batch_calls = metrics.get("planning_batch_calls")
    if (
        type(row_calls) is not int
        or row_calls != sum(planner_calls)
        or type(batch_calls) is not int
        or batch_calls < 1
        or row_calls < batch_calls
    ):
        raise ValueError(f"{path.name} contains inconsistent planner-call counts")
    planning_seconds = _nonnegative_number(metrics, "planning_elapsed_seconds")
    goal_seconds = _nonnegative_number(metrics, "goal_encoding_seconds")
    mean_batch = _nonnegative_number(metrics, "mean_planning_seconds_per_batch")
    mean_row = _nonnegative_number(metrics, "mean_planning_seconds_per_row")
    if not math.isclose(
        mean_batch,
        planning_seconds / batch_calls,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        mean_row,
        planning_seconds / row_calls,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{path.name} contains inconsistent planning-time averages")
    del goal_seconds
    _nonnegative_number(payload, "elapsed_seconds")
    return payload


def _record_fingerprint(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def _single_record(results: list[dict], artifact_name: str) -> dict:
    records = {_record_fingerprint(result["artifacts"][artifact_name]) for result in results}
    if len(records) != 1:
        raise ValueError(f"results do not share one {artifact_name}")
    return results[0]["artifacts"][artifact_name]


def _shared_task_config(results: list[dict]) -> dict:
    tasks = [result.get("config", {}).get("task") for result in results]
    if any(not isinstance(task, dict) for task in tasks):
        raise ValueError("closed-loop results do not declare resolved task configuration")
    if len({_record_fingerprint(task) for task in tasks}) != 1:
        raise ValueError("closed-loop result files do not share one task configuration")
    task = tasks[0]
    name = task.get("name")
    label = task.get("label")
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
        raise ValueError("task name must be a safe lowercase output identifier")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("task label must be a non-empty string")
    return task


def _validate_open_loop(
    open_loop_dir: Path,
    *,
    code_revision: str,
    seed: int,
    max_goal_offset: int,
    action_block: int,
    shared_artifacts: dict[str, dict],
    task_config: dict,
) -> dict:
    metrics_path = open_loop_dir / "open_loop_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text())
    if payload.get("config", {}).get("task") != task_config:
        raise ValueError("open-loop and closed-loop results use different task configuration")
    if payload.get("seed") != seed:
        raise ValueError("open-loop metrics use a different evaluation seed")
    if payload.get("code_revision") != code_revision:
        raise ValueError("open-loop and closed-loop results use different code revisions")
    artifacts = payload.get("artifacts")
    validate_artifact_records(artifacts, verify_files=True)
    if not OPEN_LOOP_ARTIFACTS.issubset(artifacts):
        raise ValueError("open-loop metrics are missing evaluated artifacts")
    for name, record in shared_artifacts.items():
        if artifacts[name] != record:
            raise ValueError(f"open-loop and closed-loop results do not share {name}")

    expected_steps = list(range(action_block, max_goal_offset + 1, action_block))
    if payload.get("environment_steps") != expected_steps:
        raise ValueError("open-loop metrics do not cover every H50 prefix")
    sample_count = payload.get("num_samples")
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("open-loop metrics have an invalid sample count")
    for key in ("fast_lewm_mse", "persistence_mse"):
        values = payload.get(key)
        if (
            not isinstance(values, list)
            or len(values) != len(expected_steps)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            )
        ):
            raise ValueError(f"open-loop {key} is invalid")
    figure_path = open_loop_dir / "open_loop_curve.png"
    if not figure_path.is_file():
        raise FileNotFoundError(figure_path)
    return payload


def _wilson_interval(successes: int, total: int, z_score: float = 1.96) -> tuple[float, float]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("success count must lie inside a non-empty evaluation set")
    probability = successes / total
    denominator = 1.0 + z_score**2 / total
    center = (probability + z_score**2 / (2 * total)) / denominator
    radius = (
        z_score
        * math.sqrt(probability * (1.0 - probability) / total + z_score**2 / (4 * total**2))
        / denominator
    )
    return 100.0 * (center - radius), 100.0 * (center + radius)


def _plot_results(
    results: dict[int, dict[str, dict]],
    goal_offsets: tuple[int, ...],
    output_dir: Path,
    task_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in METHODS:
        rates = [results[offset][method]["metrics"]["success_rate"] for offset in goal_offsets]
        axis.plot(
            goal_offsets,
            rates,
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Goal offset (environment steps)")
    axis.set_ylabel("Success rate (%)")
    axis.set_xticks(goal_offsets)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{task_name}_horizon_success.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method in METHODS:
        seconds = [
            results[offset][method]["elapsed_seconds"]
            / len(results[offset][method]["metrics"]["episode_successes"])
            for offset in goal_offsets
        ]
        axis.plot(
            goal_offsets,
            seconds,
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Goal offset (environment steps)")
    axis.set_ylabel("Amortized end-to-end time (s/task, log scale)")
    axis.set_yscale("log")
    axis.set_xticks(goal_offsets)
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{task_name}_horizon_time.png", dpi=180)
    plt.close(figure)


def summarize(
    run_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    open_loop_dir: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    open_loop_dir = (
        Path(open_loop_dir) if open_loop_dir is not None else run_dir.parent / "open_loop"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (run_dir / f"paired_manifest_seed_{seed}.json").resolve()
    manifest, goal_offsets, manifest_sha256 = _validate_manifest(manifest_path, seed)
    results = {
        offset: {
            method: _load_result(
                run_dir / f"offset_{offset}" / f"{method}_seed_{seed}.json",
                method=method,
                seed=seed,
                goal_offset=offset,
                goal_offsets=goal_offsets,
                manifest_num_eval=manifest["num_eval"],
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )
            for method in METHODS
        }
        for offset in goal_offsets
    }
    flat_results = [results[offset][method] for offset in goal_offsets for method in METHODS]
    task = _shared_task_config(flat_results)

    code_revisions = {result["code_revision"] for result in flat_results}
    if len(code_revisions) != 1:
        raise ValueError("closed-loop result files do not share one code revision")
    code_revision = next(iter(code_revisions))
    cache_record = _single_record(flat_results, "latent_cache_metadata")
    released_config = _single_record(
        [results[offset]["cem"] for offset in goal_offsets],
        "released_lewm_config",
    )
    released_weights = _single_record(
        [results[offset]["cem"] for offset in goal_offsets],
        "released_lewm_weights",
    )
    learned_results = [
        results[offset][method] for offset in goal_offsets for method in ("gc_idm", "larc")
    ]
    world_records = {
        name: _single_record(learned_results, name)
        for name in (
            "world_model_config",
            "world_model_weights",
            "world_model_metadata",
        )
    }
    policy_records = {
        method: {
            name: _single_record([results[offset][method] for offset in goal_offsets], name)
            for name in ("policy_config", "policy_weights", "policy_metadata")
        }
        for method in ("gc_idm", "larc")
    }

    cache_metadata = json.loads(Path(cache_record["path"]).read_text())
    if (
        cache_metadata.get("source_model_config_sha256") != released_config["sha256"]
        or cache_metadata.get("source_weights_sha256") != released_weights["sha256"]
    ):
        raise ValueError("CEM artifacts differ from the shared latent-cache source")

    base_protocol = None
    for offset in goal_offsets:
        protocols = [results[offset][method]["config"]["protocol"] for method in METHODS]
        if any(protocol != protocols[0] for protocol in protocols[1:]):
            raise ValueError(f"offset {offset} result files do not share one evaluation protocol")
        normalized = {
            key: value
            for key, value in protocols[0].items()
            if key not in {"goal_offset", "eval_budget"}
        }
        if base_protocol is None:
            base_protocol = normalized
        elif normalized != base_protocol:
            raise ValueError("goal offsets do not share one paired evaluation protocol")

    action_blocks = {result["config"]["cem"]["action_block"] for result in flat_results}
    if len(action_blocks) != 1:
        raise ValueError("closed-loop result files do not share one action block")
    action_block = next(iter(action_blocks))
    if type(action_block) is not int or action_block < 1:
        raise ValueError("closed-loop action block is invalid")

    open_loop = _validate_open_loop(
        open_loop_dir,
        code_revision=code_revision,
        seed=seed,
        max_goal_offset=goal_offsets[-1],
        action_block=action_block,
        shared_artifacts={
            "latent_cache_metadata": cache_record,
            **world_records,
        },
        task_config=task,
    )
    _plot_results(results, goal_offsets, output_dir, task["name"])
    open_loop_figure_name = f"{task['name']}_open_loop_curve.png"
    shutil.copy2(open_loop_dir / "open_loop_curve.png", output_dir / open_loop_figure_name)

    rows = [
        f"# {task['label']} horizon-stress results",
        "",
        f"Evaluation seed: `{seed}`. The same `{manifest['num_eval']}` held-out "
        "episode/start pairs are reused at every goal offset and by every method.",
        "",
        "| Goal offset | Method | Successes | Success rate | 95% Wilson CI | "
        "End-to-end s/task | Planning s/replan | Replans/task |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for offset in goal_offsets:
        for method in METHODS:
            result = results[offset][method]
            metrics = result["metrics"]
            episode_count = len(metrics["episode_successes"])
            success_count = sum(metrics["episode_successes"])
            rate = 100.0 * success_count / episode_count
            interval = _wilson_interval(success_count, episode_count)
            rows.append(
                f"| {offset} | {METHOD_LABELS[method]} | "
                f"{success_count}/{episode_count} | {rate:.1f}% | "
                f"[{interval[0]:.1f}%, {interval[1]:.1f}%] | "
                f"{result['elapsed_seconds'] / episode_count:.4f} | "
                f"{metrics['mean_planning_seconds_per_row']:.4f} | "
                f"{metrics['planning_row_calls'] / episode_count:.2f} |"
            )
    rows.extend(
        [
            "",
            "## Paired success differences",
            "",
            "Positive differences favor the first method. Discordant counts use the "
            "shared manifest order.",
            "",
            "| Goal offset | Comparison | Difference | First only | Second only |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    comparisons = (("larc", "cem"), ("larc", "gc_idm"), ("gc_idm", "cem"))
    for offset in goal_offsets:
        episode_count = manifest["num_eval"]
        for first, second in comparisons:
            first_values = results[offset][first]["metrics"]["episode_successes"]
            second_values = results[offset][second]["metrics"]["episode_successes"]
            first_only = sum(
                first_success and not second_success
                for first_success, second_success in zip(first_values, second_values)
            )
            second_only = sum(
                second_success and not first_success
                for first_success, second_success in zip(first_values, second_values)
            )
            difference = 100.0 * (sum(first_values) - sum(second_values)) / episode_count
            rows.append(
                f"| {offset} | {METHOD_LABELS[first]} - {METHOD_LABELS[second]} | "
                f"{difference:+.1f} pp | {first_only} | {second_only} |"
            )
    rows.extend(
        [
            "",
            f"![Success versus goal offset]({task['name']}_horizon_success.png)",
            "",
            f"![End-to-end time versus goal offset]({task['name']}_horizon_time.png)",
            "",
            "## Fast-LeWM open-loop validation",
            "",
            f"Held-out validation clips: `{open_loop['num_samples']}`.",
            "",
            "| Environment steps | Fast-LeWM MSE | Persistence MSE |",
            "|---:|---:|---:|",
        ]
    )
    rows.extend(
        f"| {steps} | {model_mse:.6f} | {baseline_mse:.6f} |"
        for steps, model_mse, baseline_mse in zip(
            open_loop["environment_steps"],
            open_loop["fast_lewm_mse"],
            open_loop["persistence_mse"],
        )
    )
    rows.extend(
        [
            "",
            f"![Fast-LeWM open-loop curve]({open_loop_figure_name})",
            "",
            "## Evaluation provenance",
            "",
            f"Code revision: `{code_revision}`.",
            f"Paired manifest SHA-256: `{manifest_sha256}`.",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
            f"| Latent cache metadata | {cache_record['sha256']} |",
            f"| Released LeWM config | {released_config['sha256']} |",
            f"| Released LeWM weights | {released_weights['sha256']} |",
            f"| Fast-LeWM config | {world_records['world_model_config']['sha256']} |",
            f"| Fast-LeWM weights | {world_records['world_model_weights']['sha256']} |",
            f"| GC-IDM config | {policy_records['gc_idm']['policy_config']['sha256']} |",
            f"| GC-IDM weights | {policy_records['gc_idm']['policy_weights']['sha256']} |",
            f"| LARC config | {policy_records['larc']['policy_config']['sha256']} |",
            f"| LARC weights | {policy_records['larc']['policy_weights']['sha256']} |",
            "",
            "Raw per-task metrics, resolved configurations, and the paired manifest "
            "remain in the external run directory.",
            "",
        ]
    )
    destination = output_dir / f"RESULTS_{task['name']}_horizon.md"
    destination.write_text("\n".join(rows))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open-loop-dir")
    arguments = parser.parse_args()
    print(
        summarize(
            arguments.run_dir,
            arguments.output_dir,
            arguments.seed,
            arguments.open_loop_dir,
        )
    )


if __name__ == "__main__":
    main()
