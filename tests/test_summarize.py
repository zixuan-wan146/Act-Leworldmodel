import json

import pytest

from eval.provenance import artifact_record
from eval.summarize import summarize
from utils import file_sha256


METHODS = ("cem", "gc_idm", "larc")
GOAL_OFFSETS = (25, 35, 50)
CODE_REVISION = "a" * 40


def _write_results(tmp_path):
    run_dir = tmp_path / "eval"
    open_loop_dir = tmp_path / "open_loop"
    output_dir = tmp_path / "results"
    run_dir.mkdir()
    open_loop_dir.mkdir()
    manifest_path = run_dir / "paired_manifest_seed_42.json"
    manifest = {
        "version": 2,
        "evaluation_seed": 42,
        "num_eval": 2,
        "goal_offsets": list(GOAL_OFFSETS),
        "max_goal_offset": max(GOAL_OFFSETS),
        "episode_indices": [7, 9],
        "start_steps": [1, 3],
    }
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = file_sha256(manifest_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    def make_artifact(name):
        path = artifacts_dir / name
        path.write_bytes(name.encode())
        return artifact_record(path)

    released_config = make_artifact("released_lewm_config.yaml")
    released_weights = make_artifact("released_lewm_weights.pt")
    cache_metadata_path = artifacts_dir / "cache_metadata.json"
    cache_metadata_path.write_text(
        json.dumps(
            {
                "source_model_config_sha256": released_config["sha256"],
                "source_weights_sha256": released_weights["sha256"],
            }
        )
    )
    cache_metadata = artifact_record(cache_metadata_path)
    world_artifacts = {
        "world_model_config": make_artifact("world_model_config.yaml"),
        "world_model_weights": make_artifact("world_model_weights.pt"),
        "world_model_metadata": make_artifact("world_model_metadata.json"),
    }
    policy_artifacts = {
        method: {
            "policy_config": make_artifact(f"{method}_policy_config.yaml"),
            "policy_weights": make_artifact(f"{method}_policy_weights.pt"),
            "policy_metadata": make_artifact(f"{method}_policy_metadata.json"),
        }
        for method in ("gc_idm", "larc")
    }
    method_artifacts = {
        "cem": {
            "latent_cache_metadata": cache_metadata,
            "released_lewm_config": released_config,
            "released_lewm_weights": released_weights,
        },
        **{
            method: {
                "latent_cache_metadata": cache_metadata,
                **world_artifacts,
                **policy_artifacts[method],
            }
            for method in ("gc_idm", "larc")
        },
    }
    success_by_offset = {
        25: ([False, True], [True, True], [True, True]),
        35: ([False, False], [False, True], [True, True]),
        50: ([False, False], [False, True], [False, True]),
    }
    for goal_offset in GOAL_OFFSETS:
        offset_dir = run_dir / f"offset_{goal_offset}"
        offset_dir.mkdir()
        budget = 2 * goal_offset
        for index, method in enumerate(METHODS):
            successes = success_by_offset[goal_offset][index]
            planner_calls = [budget // 5, budget // 5] if method != "gc_idm" else [budget, budget]
            row_calls = sum(planner_calls)
            batch_calls = max(planner_calls)
            planning_seconds = row_calls * 0.01
            protocol = {
                "seed": 42,
                "training_seed": 3072,
                "num_eval": 2,
                "goal_offset": goal_offset,
                "goal_offsets": list(GOAL_OFFSETS),
                "budget_multiplier": 2,
                "eval_budget": budget,
                "manifest_path": str(manifest_path.resolve()),
                "video": False,
            }
            payload = {
                "method": method,
                "elapsed_seconds": float(goal_offset + index),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": manifest_hash,
                "code_revision": CODE_REVISION,
                "artifacts": method_artifacts[method],
                "config": {
                    "method": method,
                    "code_revision": CODE_REVISION,
                    "cem": {"action_block": 5},
                    "protocol": protocol,
                },
                "metrics": {
                    "success_rate": 100.0 * sum(successes) / len(successes),
                    "episode_successes": successes,
                    "episode_environment_steps": [budget, budget],
                    "episode_planner_calls": planner_calls,
                    "planning_batch_calls": batch_calls,
                    "planning_row_calls": row_calls,
                    "planning_elapsed_seconds": planning_seconds,
                    "mean_planning_seconds_per_batch": planning_seconds / batch_calls,
                    "mean_planning_seconds_per_row": planning_seconds / row_calls,
                    "goal_encoding_seconds": 0.02,
                },
            }
            (offset_dir / f"{method}_seed_42.json").write_text(json.dumps(payload))

    open_loop = {
        "seed": 42,
        "num_samples": 100,
        "environment_steps": list(range(5, 51, 5)),
        "fast_lewm_mse": [0.01 * index for index in range(1, 11)],
        "persistence_mse": [0.05 * index for index in range(1, 11)],
        "code_revision": CODE_REVISION,
        "artifacts": {
            "latent_cache_metadata": cache_metadata,
            **world_artifacts,
        },
        "config": {},
    }
    (open_loop_dir / "open_loop_metrics.json").write_text(json.dumps(open_loop))
    (open_loop_dir / "open_loop_curve.png").write_bytes(b"figure")
    return run_dir, open_loop_dir, output_dir


def test_summary_includes_all_offsets_and_open_loop_results(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)

    destination = summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)
    report = destination.read_text()
    assert destination.name == "RESULTS_pusht_horizon.md"
    assert "| 25 | CEM |" in report
    assert "| 35 | GC-IDM |" in report
    assert "| 50 | LARC |" in report
    assert "95% Wilson CI" in report
    assert "Evaluation provenance" in report
    assert CODE_REVISION in report
    assert "Fast-LeWM open-loop validation" in report
    assert "Paired success differences" in report
    assert "0.010000" in report
    assert (output_dir / "open_loop_curve.png").read_bytes() == b"figure"
    assert (output_dir / "pusht_horizon_success.png").is_file()
    assert (output_dir / "pusht_horizon_time.png").is_file()


def test_summary_rejects_inconsistent_success_rate(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    path = run_dir / "offset_50" / "larc_seed_42.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["success_rate"] = 0.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="success_rate"):
        summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)


def test_summary_rejects_cross_protocol_mixing(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    path = run_dir / "offset_35" / "gc_idm_seed_42.json"
    payload = json.loads(path.read_text())
    payload["config"]["protocol"]["eval_budget"] = 71
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evaluation budget"):
        summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)


def test_summary_rejects_manifest_task_count_mismatch(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    path = run_dir / "offset_25" / "cem_seed_42.json"
    payload = json.loads(path.read_text())
    payload["config"]["protocol"]["num_eval"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="num_eval disagrees with the paired manifest"):
        summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)


def test_summary_rejects_changed_evaluated_artifact(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    result = json.loads((run_dir / "offset_25" / "larc_seed_42.json").read_text())
    weights_path = result["artifacts"]["policy_weights"]["path"]
    with open(weights_path, "ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="policy_weights content changed"):
        summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)


def test_summary_rejects_cross_revision_mixing(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    path = run_dir / "offset_50" / "larc_seed_42.json"
    payload = json.loads(path.read_text())
    payload["code_revision"] = "b" * 40
    payload["config"]["code_revision"] = "b" * 40
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="one code revision"):
        summarize(run_dir, output_dir, seed=42, open_loop_dir=open_loop_dir)
