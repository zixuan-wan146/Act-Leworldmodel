import json

from eval.summarize import summarize


def test_summary_includes_closed_and_open_loop_results(tmp_path):
    run_dir = tmp_path / "eval"
    open_loop_dir = tmp_path / "open_loop"
    output_dir = tmp_path / "results"
    run_dir.mkdir()
    open_loop_dir.mkdir()
    for index, method in enumerate(("cem", "gc_idm", "larc")):
        payload = {
            "elapsed_seconds": float(index + 1),
            "metrics": {
                "success_rate": float(20 * (index + 1)),
                "episode_successes": [index > 0, True],
            },
        }
        (run_dir / f"{method}_seed_42.json").write_text(json.dumps(payload))
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
    assert "Fast-LeWM open-loop validation" in report
    assert "0.100000" in report
    assert (output_dir / "open_loop_curve.png").read_bytes() == b"figure"
