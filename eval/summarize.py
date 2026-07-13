"""Create versionable Push-T result tables and figures from external run files."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.provenance import (
    validate_artifact_records,
    validate_code_revision,
)

from utils import file_sha256, is_sha256


METHODS = ("cem", "gc_idm", "larc")

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


def _load_result(path: Path, *, method: str, seed: int) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("method") != method:
        raise ValueError(
            f"{path.name} declares method {payload.get('method')!r}, expected {method!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("method") != method:
        raise ValueError(f"{path.name} has an incompatible resolved method")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("seed") != seed:
        raise ValueError(f"{path.name} has an incompatible evaluation seed")
    manifest = payload.get("manifest")
    configured_manifest = protocol.get("manifest_path")
    if not isinstance(manifest, str) or not isinstance(configured_manifest, str):
        raise ValueError(f"{path.name} does not declare its evaluation manifest")
    if Path(manifest).resolve() != Path(configured_manifest).resolve():
        raise ValueError(f"{path.name} result and config reference different manifests")
    manifest_hash = payload.get("manifest_sha256")
    if not is_sha256(manifest_hash):
        raise ValueError(f"{path.name} does not declare a valid manifest SHA-256")
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
    episode_successes = metrics.get("episode_successes")
    if (
        not isinstance(episode_successes, list)
        or not episode_successes
        or any(type(value) is not bool for value in episode_successes)
    ):
        raise ValueError(f"{path.name} episode_successes must be a non-empty boolean list")
    expected_rate = 100.0 * sum(episode_successes) / len(episode_successes)
    rate = metrics.get("success_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isclose(float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{path.name} success_rate disagrees with episode_successes")
    if protocol.get("num_eval") != len(episode_successes):
        raise ValueError(f"{path.name} protocol num_eval disagrees with episode_successes")
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError(f"{path.name} has an invalid evaluation time")
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
    results = {
        method: _load_result(
            run_dir / f"{method}_seed_{seed}.json",
            method=method,
            seed=seed,
        )
        for method in METHODS
    }
    code_revisions = {results[method]["code_revision"] for method in METHODS}
    if len(code_revisions) != 1:
        raise ValueError("closed-loop result files do not share one code revision")
    code_revision = next(iter(code_revisions))
    cache_records = {
        json.dumps(results[method]["artifacts"]["latent_cache_metadata"], sort_keys=True)
        for method in METHODS
    }
    if len(cache_records) != 1:
        raise ValueError("closed-loop result files do not share one latent cache")
    for name in ("world_model_config", "world_model_weights", "world_model_metadata"):
        if results["gc_idm"]["artifacts"][name] != results["larc"]["artifacts"][name]:
            raise ValueError(f"learned-policy results do not share {name}")

    cache_metadata_path = Path(results["cem"]["artifacts"]["latent_cache_metadata"]["path"])
    cache_metadata = json.loads(cache_metadata_path.read_text())
    for metadata_field, artifact_name in (
        ("source_model_config_sha256", "released_lewm_config"),
        ("source_weights_sha256", "released_lewm_weights"),
    ):
        if (
            cache_metadata.get(metadata_field)
            != results["cem"]["artifacts"][artifact_name]["sha256"]
        ):
            raise ValueError(f"CEM {artifact_name} differs from the shared latent-cache source")
    manifest_paths = {str(Path(results[method]["manifest"]).resolve()) for method in METHODS}
    manifest_hashes = {results[method]["manifest_sha256"] for method in METHODS}
    if len(manifest_paths) != 1 or len(manifest_hashes) != 1:
        raise ValueError("closed-loop result files do not share one manifest")
    protocols = [results[method]["config"]["protocol"] for method in METHODS]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("closed-loop result files do not share one evaluation protocol")
    manifest_path = Path(next(iter(manifest_paths)))
    if not manifest_path.is_file() or file_sha256(manifest_path) != next(iter(manifest_hashes)):
        raise ValueError("closed-loop manifest file is missing or its SHA-256 changed")
    manifest = json.loads(manifest_path.read_text())
    elapsed = [results[method]["elapsed_seconds"] for method in METHODS]
    episode_counts = [len(results[method]["metrics"]["episode_successes"]) for method in METHODS]
    if len(set(episode_counts)) != 1:
        raise ValueError("closed-loop result files contain different episode counts")
    episode_count = episode_counts[0]
    if (
        manifest.get("evaluation_seed") != seed
        or manifest.get("num_eval") != episode_count
        or manifest.get("goal_offset") != protocols[0].get("goal_offset")
    ):
        raise ValueError("closed-loop manifest and resolved protocol are incompatible")
    successes = [sum(results[method]["metrics"]["episode_successes"]) for method in METHODS]
    success = [100.0 * count / episode_count for count in successes]
    labels = ["CEM", "GC-IDM", "LARC"]

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    bars = axis.bar(labels, success, color=("#4C78A8", "#F58518", "#54A24B"))
    axis.set_ylim(0, 100)
    axis.set_ylabel("Success rate (%)")
    confidence_intervals = [_wilson_interval(count, episode_count) for count in successes]
    axis.set_title(f"Push-T closed-loop evaluation (n={episode_count})")
    axis.bar_label(bars, fmt="%.1f%%")
    figure.tight_layout()
    figure.savefig(output_dir / "pusht_success_rates.png", dpi=180)
    plt.close(figure)

    rows = [
        "# Push-T results",
        "",
        f"Fixed evaluation seed: `{seed}`. All methods use the same episode/start manifest.",
        "The episodes are held out from newly trained dynamics/policies; the released "
        "LeWM/CEM checkpoint retains its upstream clip-level split provenance. See "
        "[the protocol](../docs/pusht_protocol.md).",
        "",
        "| Method | Successes | Success rate | 95% Wilson CI | Evaluation time |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {label} | {count}/{episode_count} | {rate:.1f}% | "
        f"[{interval[0]:.1f}%, {interval[1]:.1f}%] | {seconds:.1f} s |"
        for label, count, rate, interval, seconds in zip(
            labels, successes, success, confidence_intervals, elapsed
        )
    )
    rows.extend(
        [
            "",
            "![Push-T success rates](pusht_success_rates.png)",
            "",
        ]
    )
    provenance_records = (
        (
            "Latent cache metadata",
            results["cem"]["artifacts"]["latent_cache_metadata"],
        ),
        (
            "Released LeWM config",
            results["cem"]["artifacts"]["released_lewm_config"],
        ),
        (
            "Released LeWM weights",
            results["cem"]["artifacts"]["released_lewm_weights"],
        ),
        (
            "Fast-LeWM config",
            results["gc_idm"]["artifacts"]["world_model_config"],
        ),
        (
            "Fast-LeWM weights",
            results["gc_idm"]["artifacts"]["world_model_weights"],
        ),
        ("GC-IDM config", results["gc_idm"]["artifacts"]["policy_config"]),
        ("GC-IDM weights", results["gc_idm"]["artifacts"]["policy_weights"]),
        ("LARC config", results["larc"]["artifacts"]["policy_config"]),
        ("LARC weights", results["larc"]["artifacts"]["policy_weights"]),
    )
    rows.extend(
        [
            "## Evaluation provenance",
            "",
            f"Code revision: {code_revision}.",
            f"Manifest SHA-256: {next(iter(manifest_hashes))}.",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
        ]
    )
    rows.extend(f"| {label} | {record['sha256']} |" for label, record in provenance_records)
    rows.append("")
    open_loop_metrics_path = open_loop_dir / "open_loop_metrics.json"
    if open_loop_metrics_path.is_file():
        open_loop = json.loads(open_loop_metrics_path.read_text())
        rows.extend(
            [
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
        source_figure = open_loop_dir / "open_loop_curve.png"
        if source_figure.is_file():
            shutil.copy2(source_figure, output_dir / "open_loop_curve.png")
            rows.extend(["", "![Fast-LeWM open-loop curve](open_loop_curve.png)", ""])
    rows.extend(
        [
            "Raw metrics, per-episode successes, resolved configs, and the fixed protocol "
            "manifest remain in the external run directory and are not committed.",
            "",
        ]
    )
    destination = output_dir / "RESULTS_pusht.md"
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
