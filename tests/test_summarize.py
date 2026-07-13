import json

import pytest

from eval.provenance import artifact_record
from eval.summarize import summarize
from utils import file_sha256


METHODS = ("cem", "gc_idm", "larc")
CODE_REVISION = "a" * 40


def _write_results(tmp_path):
    run_dir = tmp_path / "eval"
    open_loop_dir = tmp_path / "open_loop"
    output_dir = tmp_path / "results"
    run_dir.mkdir()
    open_loop_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "evaluation_seed": 42,
        "num_eval": 2,
        "goal_offset": 25,
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
    success_lists = ([False, True], [True, True], [True, True])
    for index, method in enumerate(METHODS):
        successes = success_lists[index]
        protocol = {
            "seed": 42,
            "training_seed": 3072,
            "num_eval": 2,
            "goal_offset": 25,
            "eval_budget": 50,
            "manifest_path": str(manifest_path.resolve()),
        }
        payload = {
            "method": method,
            "elapsed_seconds": float(index + 1),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_hash,
            "code_revision": CODE_REVISION,
            "artifacts": method_artifacts[method],
            "config": {"method": method, "code_revision": CODE_REVISION, "protocol": protocol},
            "metrics": {
                "success_rate": 100.0 * sum(successes) / len(successes),
                "episode_successes": successes,
            },
        }
        (run_dir / f"{method}_seed_42.json").write_text(json.dumps(payload))
    return run_dir, open_loop_dir, output_dir


def test_summary_includes_closed_and_open_loop_results(tmp_path):
    run_dir, open_loop_dir, output_dir = _write_results(tmp_path)
    open_loop = {
        "num_samples": 100,
        "environment_steps": [5, 10],
        "fast_lewm_mse": [0.1, 0.2],
        "persistence_mse": [0.5, 0.8],
    }
    (open_loop_dir / "open_loop_metrics.json").write_text(json.dumps(open_loop))
    (open_loop_dir / "open_loop_curve.png").write_bytes(b"figure")

    destination = summarize(run_dir, output_dir, seed=42)
    report = destination.read_text()
    assert "2/2" in report
    assert "95% Wilson CI" in report
    assert "Evaluation provenance" in report
    assert CODE_REVISION in report
    assert "Fast-LeWM open-loop validation" in report
    assert "0.100000" in report
    assert (output_dir / "open_loop_curve.png").read_bytes() == b"figure"


def test_summary_rejects_inconsistent_success_rate(tmp_path):
    run_dir, _, output_dir = _write_results(tmp_path)
    path = run_dir / "larc_seed_42.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["success_rate"] = 0.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="success_rate"):
        summarize(run_dir, output_dir, seed=42)


def test_summary_rejects_cross_protocol_mixing(tmp_path):
    run_dir, _, output_dir = _write_results(tmp_path)
    path = run_dir / "gc_idm_seed_42.json"
    payload = json.loads(path.read_text())
    payload["config"]["protocol"]["eval_budget"] = 51
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="one evaluation protocol"):
        summarize(run_dir, output_dir, seed=42)


def test_summary_rejects_changed_evaluated_artifact(tmp_path):
    run_dir, _, output_dir = _write_results(tmp_path)
    result = json.loads((run_dir / "larc_seed_42.json").read_text())
    weights_path = result["artifacts"]["policy_weights"]["path"]
    with open(weights_path, "ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="policy_weights content changed"):
        summarize(run_dir, output_dir, seed=42)


def test_summary_rejects_cross_revision_mixing(tmp_path):
    run_dir, _, output_dir = _write_results(tmp_path)
    path = run_dir / "larc_seed_42.json"
    payload = json.loads(path.read_text())
    payload["code_revision"] = "b" * 40
    payload["config"]["code_revision"] = "b" * 40
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="one code revision"):
        summarize(run_dir, output_dir, seed=42)
