from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


CONFIG_NAMES = (
    "cache_latents",
    "train_world_model",
    "train_gc_idm",
    "train_larc",
    "eval_open_loop",
    "eval_closed_loop",
)


@pytest.mark.parametrize(
    ("task_name", "dataset_name", "state_key", "environment_target"),
    (
        ("pusht", "pusht.h5", "state", "eval.pusht_env.PushTEnv"),
        ("tworoom", "tworoom.h5", "proprio", "eval.tworoom_env.TwoRoomEnv"),
    ),
)
def test_every_entrypoint_composes_from_one_task_config(
    tmp_path,
    monkeypatch,
    task_name,
    dataset_name,
    state_key,
    environment_target,
):
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "runs"
    monkeypatch.setenv("ACT_LEWM_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("ACT_LEWM_RUN_ROOT", str(run_root))
    monkeypatch.setenv("ACT_LEWM_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("PUSHT_DATASET_PATH", str(tmp_path / "pusht.h5"))
    monkeypatch.setenv("PUSHT_LEWM_WEIGHTS", str(tmp_path / "pusht.pt"))
    monkeypatch.setenv("TWOROOM_DATASET_PATH", str(tmp_path / "tworoom.h5"))
    monkeypatch.setenv("TWOROOM_LEWM_WEIGHTS", str(tmp_path / "tworoom.pt"))

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    resolved = {}
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for config_name in CONFIG_NAMES:
            cfg = compose(config_name=config_name, overrides=[f"task={task_name}"])
            resolved[config_name] = OmegaConf.to_container(cfg, resolve=True)

    for config in resolved.values():
        assert config["task"]["name"] == task_name
        assert config["task"]["state_key"] == state_key
        assert config["task"]["dataset_path"].endswith(dataset_name)
        assert config["task"]["latent_cache_dir"] == str(cache_root / task_name / "frame_latents")

    closed_loop = resolved["eval_closed_loop"]
    assert closed_loop["environment"]["_target_"] == environment_target
    assert closed_loop["dataset_path"].endswith(dataset_name)
    assert closed_loop["experiment_dir"] == str(run_root / task_name / "horizon_h50")
    assert closed_loop["cem"]["weights_path"].endswith(f"{task_name}.pt")


def test_unknown_task_config_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ACT_LEWM_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ACT_LEWM_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("ACT_LEWM_CODE_REVISION", "a" * 40)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        with pytest.raises(Exception, match="task"):
            compose(config_name="cache_latents", overrides=["task=unknown"])
