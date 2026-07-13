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


METHODS = ("cem", "gc_idm", "larc")


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
        method: json.loads((run_dir / f"{method}_seed_{seed}.json").read_text())
        for method in METHODS
    }
    success = [results[method]["metrics"]["success_rate"] for method in METHODS]
    elapsed = [results[method]["elapsed_seconds"] for method in METHODS]
    episode_counts = [len(results[method]["metrics"]["episode_successes"]) for method in METHODS]
    if len(set(episode_counts)) != 1:
        raise ValueError("closed-loop result files contain different episode counts")
    episode_count = episode_counts[0]
    successes = [sum(results[method]["metrics"]["episode_successes"]) for method in METHODS]
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
