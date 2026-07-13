"""Fixed-seed Push-T evaluation using only project-owned runtime components."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from omegaconf import DictConfig, OmegaConf

from controllers import Controller
from controllers.baselines import CEMController, CEMPlanner
from controllers.learned import GCIDMController, LARCController
from data import (
    ActionBlockTransform,
    ActionStatistics,
    PushTEvaluationDataset,
    ZScoreActionTransform,
    calculate_action_statistics,
    load_latent_metadata,
    preprocess_pusht_pixels,
)
from eval.protocol import create_or_load_manifest
from eval.provenance import (
    artifact_record,
    validate_artifact_records,
    validate_code_revision,
)
from eval.pusht_env import PushTEnv
from models.world_model import load_frozen_world_model, load_released_lewm
from models.world_model.artifacts import load_tensor_state_dict
from train.reproducibility import configure_reproducibility
from utils import file_sha256, is_sha256


FRAME_LINEAGE_FIELDS = (
    "dataset_path",
    "dataset_size",
    "dataset_mtime_ns",
    "source_checkpoint_sha256",
    "seed",
    "train_fraction",
    "train_episode_ids",
    "validation_episode_ids",
    "episode_lengths",
    "episode_offsets",
    "frame_count",
    "latent_dim",
    "latent_dtype",
    "raw_action_dim",
    "action_statistics",
)
HORIZON_VIEW_FIELDS = ("frameskip", "max_horizon", "max_goal_offset")
# Public compatibility alias used by artifact-construction tests and tools.
LINEAGE_FIELDS = FRAME_LINEAGE_FIELDS + HORIZON_VIEW_FIELDS


def _evaluation_artifacts(cfg: DictConfig, method: str) -> dict[str, dict[str, str]]:
    cache_metadata = Path(cfg.latent_cache_dir) / "metadata.json"
    records = {"latent_cache_metadata": artifact_record(cache_metadata)}
    if method == "cem":
        records.update(
            {
                "released_lewm_config": artifact_record(cfg.cem.config_path),
                "released_lewm_weights": artifact_record(cfg.cem.weights_path),
            }
        )
        return records

    policy_dir = Path(cfg[method].directory)
    world_config = Path(cfg.world_model.config_path)
    records.update(
        {
            "world_model_config": artifact_record(world_config),
            "world_model_weights": artifact_record(cfg.world_model.weights_path),
            "world_model_metadata": artifact_record(world_config.with_name("model_metadata.json")),
            "policy_config": artifact_record(policy_dir / "policy_config.yaml"),
            "policy_weights": artifact_record(policy_dir / cfg[method].weights),
            "policy_metadata": artifact_record(policy_dir / "policy_metadata.json"),
        }
    )
    return records


def _validate_cem_source_artifacts(cache_metadata: dict, cfg: DictConfig) -> None:
    expected_files = (
        (
            "source_model_config_sha256",
            cfg.cem.config_path,
            "CEM model config",
        ),
        (
            "source_weights_sha256",
            cfg.cem.weights_path,
            "CEM model weights",
        ),
    )
    for metadata_field, path, label in expected_files:
        expected = cache_metadata.get(metadata_field)
        if not is_sha256(expected):
            raise ValueError(f"latent cache is missing {metadata_field}")
        if file_sha256(path) != expected:
            raise ValueError(f"{label} differ from the latent-cache representation source")


class LearnedPushTPolicy:
    """Adapt a project controller to the batched evaluation loop."""

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
        for buffer in self._buffers:
            buffer.clear()
        self._has_goal = False

    def get_action(self, info_dict: dict) -> np.ndarray:
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
            replan = np.asarray(
                [alive[index] and not self._buffers[index] for index in range(self.env.num_envs)]
            )
            rows = np.flatnonzero(replan).astype(np.int64, copy=False)
            if len(rows):
                remaining_values = np.maximum(
                    self.goal_offset - self._elapsed,
                    self.minimum_horizon,
                )
                indices = torch.from_numpy(rows).to(self.device)
                remaining = torch.from_numpy(remaining_values[rows]).to(self.device)
                command = self.controller.act(
                    self._preprocess(np.asarray(info_dict["pixels"])[rows]),
                    remaining,
                    batch_indices=indices,
                )
                chunks = command.actions.float().cpu().numpy()
                for chunk_index, env_index in enumerate(rows):
                    self._buffers[env_index].extend(chunks[chunk_index, : command.replan_after])
            for env_index in np.flatnonzero(alive):
                if not self._buffers[env_index]:
                    raise RuntimeError("controller action buffer unexpectedly empty")
                output[env_index] = self._buffers[env_index].popleft()

        output[alive] = np.clip(
            output[alive],
            self.env.single_action_space.low,
            self.env.single_action_space.high,
        )
        self._elapsed[alive] += 1
        return output


class PushTEvaluationPool:
    """Small selective-step pool for one fixed evaluation manifest."""

    def __init__(self, count: int, resolution: int) -> None:
        if count < 1:
            raise ValueError("evaluation pool must contain at least one environment")
        self.envs = [PushTEnv(resolution=resolution) for _ in range(count)]
        self.single_action_space = self.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, count)

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def initialize(self, states: np.ndarray, goal_states: np.ndarray) -> None:
        if len(states) != self.num_envs or len(goal_states) != self.num_envs:
            raise ValueError("state batches differ from the evaluation pool")
        for env, state, goal_state in zip(self.envs, states, goal_states):
            env.reset(options={"state": state, "goal_state": goal_state})

    def step(
        self,
        actions: np.ndarray,
        alive: np.ndarray,
        previous_pixels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels = previous_pixels.copy()
        terminated = np.zeros(self.num_envs, dtype=bool)
        for index in np.flatnonzero(alive):
            _, _, done, _, info = self.envs[index].step(actions[index])
            pixels[index] = info["pixels"]
            terminated[index] = done
        return pixels, terminated

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _validate_learned_artifacts(
    policy_metadata: dict,
    world_metadata: dict,
    cache_metadata: dict,
    *,
    method: str,
    training_seed: int,
    goal_offset: int,
) -> None:
    if policy_metadata.get("method") != method:
        raise ValueError("policy metadata method does not match requested evaluation method")
    for field in FRAME_LINEAGE_FIELDS:
        if field not in cache_metadata:
            raise ValueError(f"latent cache metadata is missing lineage field {field}")
    for artifact_name, artifact in (
        ("policy", policy_metadata),
        ("world model", world_metadata),
    ):
        for field in FRAME_LINEAGE_FIELDS:
            if field not in artifact:
                raise ValueError(f"{artifact_name} metadata is missing lineage field {field}")
            if artifact[field] != cache_metadata[field]:
                raise ValueError(f"{artifact_name} and latent cache have incompatible {field}")
        for field in HORIZON_VIEW_FIELDS:
            if field not in artifact:
                raise ValueError(f"{artifact_name} metadata is missing horizon field {field}")
            if artifact[field] != policy_metadata[field]:
                raise ValueError(f"{artifact_name} and policy have incompatible {field}")
    if policy_metadata.get("training_seed") != training_seed:
        raise ValueError("policy training seed does not match the evaluation protocol")
    if world_metadata.get("training_seed") != training_seed:
        raise ValueError("world-model training seed does not match the evaluation protocol")
    if goal_offset > int(policy_metadata["max_goal_offset"]):
        raise ValueError("evaluation goal offset exceeds the policy training horizon")


def _load_learned_policy(cfg: DictConfig, method: str, cache_metadata: dict):
    world_model = load_frozen_world_model(cfg.world_model.config_path, cfg.world_model.weights_path)
    policy_dir = Path(cfg[method].directory)
    policy_config = OmegaConf.load(policy_dir / "policy_config.yaml")
    policy = hydra.utils.instantiate(policy_config)
    policy.load_state_dict(
        load_tensor_state_dict(policy_dir / cfg[method].weights),
        strict=True,
    )
    metadata = json.loads((policy_dir / "policy_metadata.json").read_text())
    world_metadata = json.loads(
        Path(cfg.world_model.config_path).with_name("model_metadata.json").read_text()
    )
    _validate_learned_artifacts(
        metadata,
        world_metadata,
        cache_metadata,
        method=method,
        training_seed=cfg.protocol.training_seed,
        goal_offset=cfg.protocol.goal_offset,
    )
    stats = ActionStatistics.from_mapping(metadata["action_statistics"])
    mean, std = torch.tensor(stats.mean), torch.tensor(stats.std)
    if method == "gc_idm":
        controller = GCIDMController(world_model, policy, ZScoreActionTransform(mean, std))
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


def _load_cem_policy(
    cfg: DictConfig,
    cache_metadata: dict,
    dataset: PushTEvaluationDataset,
):
    model, artifact_metadata = load_released_lewm(cfg.cem.config_path, cfg.cem.weights_path)
    if artifact_metadata["source_checkpoint_sha256"] != cache_metadata["source_checkpoint_sha256"]:
        raise ValueError("CEM weights differ from the latent-cache representation source")
    stats = calculate_action_statistics(dataset.action_array())
    transform = ActionBlockTransform(
        torch.tensor(stats.mean),
        torch.tensor(stats.std),
        frameskip=cfg.cem.action_block,
    )
    planner = CEMPlanner(
        model=model,
        action_transform=transform,
        horizon=cfg.cem.horizon,
        receding_horizon=cfg.cem.receding_horizon,
        batch_size=cfg.cem.batch_size,
        num_samples=cfg.cem.num_samples,
        var_scale=cfg.cem.var_scale,
        n_steps=cfg.cem.n_steps,
        topk=cfg.cem.topk,
        device=cfg.device,
        seed=cfg.protocol.seed,
        warm_start=cfg.cem.warm_start,
    )
    return LearnedPushTPolicy(
        controller=CEMController(planner, commit_steps=planner.commit_steps),
        goal_offset=cfg.protocol.goal_offset,
        minimum_horizon=cfg.cem.action_block,
        device=cfg.device,
    )


def _evaluate(
    *,
    policy: LearnedPushTPolicy,
    pool: PushTEvaluationPool,
    rows: dict[str, np.ndarray],
    eval_budget: int,
    video_dir: Path | None,
) -> dict:
    pool.initialize(rows["state"], rows["goal_state"])
    policy.set_env(pool)
    pixels = np.asarray(rows["pixels"]).copy()
    goals = np.asarray(rows["goal_pixels"]).copy()
    alive = np.ones(pool.num_envs, dtype=bool)
    successes = np.zeros(pool.num_envs, dtype=bool)
    frames = [[pixels[index].copy()] for index in range(pool.num_envs)] if video_dir else None
    info = {
        "pixels": pixels[:, None],
        "goal": goals[:, None],
        "terminated": ~alive,
        "_needs_flush": np.ones(pool.num_envs, dtype=bool),
    }
    for _ in range(eval_budget):
        actions = policy.get_action(info)
        pixels, terminated = pool.step(actions, alive, pixels)
        successes |= terminated
        alive &= ~terminated
        if frames is not None:
            for index in range(pool.num_envs):
                frames[index].append(pixels[index].copy())
        info = {
            "pixels": pixels[:, None],
            "goal": goals[:, None],
            "terminated": ~alive,
            "_needs_flush": np.zeros(pool.num_envs, dtype=bool),
        }
        if not alive.any():
            break
    if frames is not None:
        import imageio.v3 as imageio

        video_dir.mkdir(parents=True, exist_ok=True)
        for index, episode_frames in enumerate(frames):
            imageio.imwrite(video_dir / f"episode_{index:03d}.mp4", episode_frames, fps=10)
    return {
        "success_rate": float(successes.mean() * 100.0),
        "episode_successes": successes.tolist(),
        "seeds": None,
    }


def run(cfg: DictConfig) -> dict:
    configure_reproducibility(cfg.protocol.seed)
    method = str(cfg.method)
    if method not in ("cem", "gc_idm", "larc"):
        raise ValueError("method must be cem, gc_idm, or larc")
    code_revision = validate_code_revision(str(cfg.code_revision))
    cache_metadata = load_latent_metadata(cfg.latent_cache_dir)
    artifacts = _evaluation_artifacts(cfg, method)
    if method == "cem":
        _validate_cem_source_artifacts(cache_metadata, cfg)
    manifest = create_or_load_manifest(
        dataset_path=cfg.dataset_path,
        latent_cache_dir=cfg.latent_cache_dir,
        output_path=cfg.protocol.manifest_path,
        seed=cfg.protocol.seed,
        num_eval=cfg.protocol.num_eval,
        goal_offset=cfg.protocol.goal_offset,
    )
    manifest_path = Path(cfg.protocol.manifest_path).resolve()
    manifest_sha256 = file_sha256(manifest_path)
    with PushTEvaluationDataset(cfg.dataset_path) as dataset:
        rows = dataset.evaluation_rows(
            manifest["episode_indices"],
            manifest["start_steps"],
            cfg.protocol.goal_offset,
        )
        policy = (
            _load_cem_policy(cfg, cache_metadata, dataset)
            if method == "cem"
            else _load_learned_policy(cfg, method, cache_metadata)
        )
        pool = PushTEvaluationPool(cfg.protocol.num_eval, cfg.environment.image_size)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / f"{method}_videos" if cfg.protocol.video else None
        started = time.time()
        try:
            metrics = _evaluate(
                policy=policy,
                pool=pool,
                rows=rows,
                eval_budget=cfg.protocol.eval_budget,
                video_dir=video_dir,
            )
        finally:
            pool.close()
    if file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("evaluation manifest changed during execution")
    validate_artifact_records(artifacts, verify_files=True)
    result = {
        "method": method,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "code_revision": code_revision,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "artifacts": artifacts,
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
