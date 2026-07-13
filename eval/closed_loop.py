"""Fixed-seed Push-T closed-loop evaluation for CEM, GC-IDM, and LARC."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import hydra
import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from controllers import Controller
from controllers.learned import GCIDMController, LARCController
from data import (
    ActionBlockTransform,
    ActionStatistics,
    ZScoreActionTransform,
    calculate_action_statistics,
    file_sha256,
    load_latent_metadata,
    preprocess_pusht_pixels,
)
from eval.protocol import create_or_load_manifest
from models.world_model import load_frozen_world_model
from train.reproducibility import configure_reproducibility


PUSHT_CALLABLES = [
    {"method": "_set_state", "args": {"state": {"value": "state"}}},
    {
        "method": "_set_goal_state",
        "args": {"goal_state": {"value": "goal_state"}},
    },
]


class LearnedPushTPolicy:
    """Adapt learned latent policies to stable-worldmodel's vectorized API."""

    def __init__(
        self,
        *,
        controller: Controller,
        goal_offset: int,
        minimum_horizon: int,
        device: str,
    ) -> None:
        if goal_offset < 1 or minimum_horizon < 1:
            raise ValueError("goal offset and minimum horizon must be positive")
        self.controller = controller.to(device).eval().requires_grad_(False)
        self.goal_offset = int(goal_offset)
        self.minimum_horizon = int(minimum_horizon)
        self.device = torch.device(device)
        self.env = None
        self._has_goal = False
        self._elapsed = None
        self._buffers = None

    def set_env(self, env) -> None:
        self.env = env
        self._has_goal = False
        self._elapsed = np.zeros(env.num_envs, dtype=np.int64)
        self._buffers = [deque() for _ in range(env.num_envs)]

    def _preprocess(self, pixels) -> torch.Tensor:
        array = np.asarray(pixels)
        if array.ndim == 5:
            array = array[:, -1]
        return preprocess_pusht_pixels(array, self.device)

    def _reset_rows(self, rows: np.ndarray) -> None:
        if not rows.any():
            return
        self._elapsed[rows] = 0
        # A vector controller caches one goal batch. If any vector row resets,
        # refresh that batch and discard every stale open-loop prefix.
        for index in range(self.env.num_envs):
            self._buffers[index].clear()
        self._has_goal = False

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("policy must be attached to an environment")
        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(self.env.num_envs)), dtype=bool
        ).reshape(-1)
        self._reset_rows(needs_flush)
        if not self._has_goal:
            with torch.inference_mode():
                self.controller.reset(self._preprocess(info_dict["goal"]))
            self._has_goal = True
        terminated = np.asarray(
            info_dict.get("terminated", np.zeros(self.env.num_envs)), dtype=bool
        ).reshape(-1)
        action_dim = self.env.single_action_space.shape[-1]
        output = np.full((self.env.num_envs, action_dim), np.nan, dtype=np.float32)
        alive = ~terminated

        with torch.inference_mode():
            replan_rows = np.asarray(
                [alive[index] and not self._buffers[index] for index in range(self.env.num_envs)]
            )
            rows = np.flatnonzero(replan_rows)
            if len(rows):
                remaining_values = np.maximum(
                    self.goal_offset - self._elapsed,
                    self.minimum_horizon,
                )
                remaining = torch.from_numpy(remaining_values).to(self.device)
                command = self.controller.act(self._preprocess(info_dict["pixels"]), remaining)
                chunks = command.actions.float().cpu().numpy()
                for env_index in rows:
                    self._buffers[env_index].extend(chunks[env_index, : command.replan_after])
            for env_index in np.flatnonzero(alive):
                if not self._buffers[env_index]:
                    raise RuntimeError("learned-controller action buffer unexpectedly empty")
                output[env_index] = self._buffers[env_index].popleft()

        low = np.asarray(self.env.single_action_space.low)
        high = np.asarray(self.env.single_action_space.high)
        output[alive] = np.clip(output[alive], low, high)
        self._elapsed[alive] += 1
        return output.reshape(self.env.action_space.shape)


def _image_transform(image: torch.Tensor) -> torch.Tensor:
    return preprocess_pusht_pixels(image.unsqueeze(0), image.device)[0]


def _validate_learned_artifacts(
    policy_metadata: dict,
    world_metadata: dict,
    *,
    method: str,
    training_seed: int,
    goal_offset: int,
) -> None:
    if policy_metadata.get("method") != method:
        raise ValueError("policy metadata method does not match requested evaluation method")
    for field in (
        "source_checkpoint_sha256",
        "seed",
        "train_fraction",
        "frameskip",
        "max_horizon",
        "latent_dim",
        "action_statistics",
    ):
        if policy_metadata.get(field) != world_metadata.get(field):
            raise ValueError(f"policy and world model have incompatible {field}")
    if policy_metadata.get("training_seed") != training_seed:
        raise ValueError("policy training seed does not match the evaluation protocol")
    if world_metadata.get("training_seed") != training_seed:
        raise ValueError("world-model training seed does not match the evaluation protocol")
    if goal_offset > int(policy_metadata["max_goal_offset"]):
        raise ValueError("evaluation goal offset exceeds the policy training horizon")


def _load_learned_policy(cfg: DictConfig, method: str):
    world_model = load_frozen_world_model(cfg.world_model.config_path, cfg.world_model.weights_path)
    policy_dir = Path(cfg[method].directory)
    policy_config = OmegaConf.load(policy_dir / "policy_config.yaml")
    policy = hydra.utils.instantiate(policy_config)
    state = torch.load(policy_dir / cfg[method].weights, map_location="cpu", weights_only=True)
    policy.load_state_dict(state, strict=True)
    metadata = json.loads((policy_dir / "policy_metadata.json").read_text())
    world_metadata = json.loads(
        Path(cfg.world_model.config_path).with_name("model_metadata.json").read_text()
    )
    _validate_learned_artifacts(
        metadata,
        world_metadata,
        method=method,
        training_seed=cfg.protocol.training_seed,
        goal_offset=cfg.protocol.goal_offset,
    )
    stats = ActionStatistics.from_mapping(metadata["action_statistics"])
    mean = torch.tensor(stats.mean)
    std = torch.tensor(stats.std)
    if method == "gc_idm":
        transform = ZScoreActionTransform(mean, std)
        controller = GCIDMController(world_model, policy, transform)
        minimum_horizon = 1
    else:
        transform = ActionBlockTransform(mean, std, frameskip=metadata["frameskip"])
        controller = LARCController(
            world_model,
            policy,
            commit_steps=cfg[method].commit_steps,
            action_transform=transform,
        )
        minimum_horizon = metadata["frameskip"]
    return LearnedPushTPolicy(
        controller=controller,
        goal_offset=cfg.protocol.goal_offset,
        minimum_horizon=minimum_horizon,
        device=cfg.device,
    )


def _load_cem_policy(cfg: DictConfig):
    model = torch.load(cfg.cem.checkpoint, map_location="cpu", weights_only=False)
    model = model.to(cfg.device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    # The released LeWM checkpoint was trained with a scaler fit to the full
    # official Push-T training file. Reproduce that scaler exactly; our new
    # train/validation split statistics belong only to Fast-LeWM and policies.
    with h5py.File(cfg.dataset_path, "r", swmr=True) as dataset:
        stats = calculate_action_statistics(dataset["action"][:])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(stats.mean, dtype=np.float64)
    scaler.scale_ = np.asarray(stats.std, dtype=np.float64)
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = len(stats.mean)
    scaler.n_samples_seen_ = 1
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=cfg.cem.batch_size,
        num_samples=cfg.cem.num_samples,
        var_scale=cfg.cem.var_scale,
        n_steps=cfg.cem.n_steps,
        topk=cfg.cem.topk,
        device=cfg.device,
        seed=cfg.protocol.seed,
    )
    plan_config = swm.PlanConfig(
        horizon=cfg.cem.horizon,
        receding_horizon=cfg.cem.receding_horizon,
        history_len=cfg.cem.history_len,
        action_block=cfg.cem.action_block,
        warm_start=cfg.cem.warm_start,
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process={"action": scaler},
        transform={"pixels": _image_transform, "goal": _image_transform},
    )


def _jsonable_metrics(metrics: dict) -> dict:
    output = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def run(cfg: DictConfig) -> dict:
    configure_reproducibility(cfg.protocol.seed)
    method = str(cfg.method)
    if method not in ("cem", "gc_idm", "larc"):
        raise ValueError("method must be cem, gc_idm, or larc")
    cache_metadata = load_latent_metadata(cfg.latent_cache_dir)
    if (
        method == "cem"
        and file_sha256(cfg.cem.checkpoint) != cache_metadata["source_checkpoint_sha256"]
    ):
        raise ValueError("CEM checkpoint differs from the released LeWM cache source")
    manifest = create_or_load_manifest(
        dataset_path=cfg.dataset_path,
        latent_cache_dir=cfg.latent_cache_dir,
        output_path=cfg.protocol.manifest_path,
        seed=cfg.protocol.seed,
        num_eval=cfg.protocol.num_eval,
        goal_offset=cfg.protocol.goal_offset,
    )
    dataset = swm.data.HDF5Dataset(
        path=cfg.dataset_path,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
    )
    world = swm.World(
        env_name=cfg.environment.name,
        num_envs=cfg.protocol.num_eval,
        image_shape=(cfg.environment.image_size, cfg.environment.image_size),
        max_episode_steps=2 * cfg.protocol.eval_budget,
    )
    policy = _load_cem_policy(cfg) if method == "cem" else _load_learned_policy(cfg, method)
    world.set_policy(policy)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / f"{method}_videos" if cfg.protocol.video else None
    started = time.time()
    try:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=manifest["start_steps"],
            goal_offset=cfg.protocol.goal_offset,
            eval_budget=cfg.protocol.eval_budget,
            episodes_idx=manifest["episode_indices"],
            callables=PUSHT_CALLABLES,
            video=video_dir,
        )
    finally:
        world.close()
    result = {
        "method": method,
        "metrics": _jsonable_metrics(metrics),
        "elapsed_seconds": time.time() - started,
        "manifest": str(Path(cfg.protocol.manifest_path).resolve()),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    destination = output_dir / f"{method}_seed_{cfg.protocol.seed}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="eval_pusht")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
