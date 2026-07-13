"""Project-owned fixed Two-Room evaluation environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class TwoRoomEnv(gym.Env):
    """Fixed two-room point navigation used by the released dataset."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    IMAGE_SIZE = 224
    BORDER_POSITION = 14
    BORDER_LINE_WIDTH = 4
    AGENT_RADIUS = 7.0
    AGENT_SPEED = 5.0
    WALL_CENTER = 112
    WALL_THICKNESS = 10
    DOOR_CENTER = 49.0
    DOOR_HALF_EXTENT = 14.0
    DOOR_COLLISION_MARGIN = 1.75
    SUCCESS_RADIUS = 16.0

    def __init__(self, resolution: int = 224) -> None:
        super().__init__()
        if resolution != self.IMAGE_SIZE:
            raise ValueError(f"Two-Room resolution must be {self.IMAGE_SIZE}")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            0.0,
            float(self.IMAGE_SIZE),
            shape=(2,),
            dtype=np.float32,
        )
        coordinates = torch.arange(self.IMAGE_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        self.wall_mask = self._make_wall_mask()
        self.agent_position = torch.tensor((60.0, 112.0), dtype=torch.float32)
        self.goal_position = torch.tensor((164.0, 112.0), dtype=torch.float32)

    @staticmethod
    def _position(value, name: str) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError(f"Two-Room {name} must contain two finite values")
        return torch.from_numpy(array.copy())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.agent_position = self._position(options.get("state", (60.0, 112.0)), "state")
        self.goal_position = self._position(
            options.get("goal_state", (164.0, 112.0)),
            "goal state",
        )
        state = self.state()
        return state, {
            "pixels": self.render(),
            "distance_to_target": self._distance(),
        }

    def state(self) -> np.ndarray:
        return self.agent_position.numpy().copy()

    def _distance(self) -> float:
        return float(torch.linalg.vector_norm(self.agent_position - self.goal_position))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Two-Room action must contain two finite values")
        action_tensor = torch.from_numpy(action.copy()).clamp_(-1.0, 1.0)
        proposed = self.agent_position + action_tensor * self.AGENT_SPEED
        self.agent_position = self._apply_collisions(self.agent_position, proposed)
        distance = self._distance()
        terminated = distance < self.SUCCESS_RADIUS
        return (
            self.state(),
            0.0,
            terminated,
            False,
            {
                "pixels": self.render(),
                "distance_to_target": distance,
            },
        )

    def _make_wall_mask(self) -> torch.Tensor:
        half_wall = self.WALL_THICKNESS // 2
        wall_stripe = (self.grid_x >= self.WALL_CENTER - half_wall) & (
            self.grid_x <= self.WALL_CENTER + half_wall
        )
        door_span = (self.grid_y >= self.DOOR_CENTER - self.DOOR_HALF_EXTENT) & (
            self.grid_y <= self.DOOR_CENTER + self.DOOR_HALF_EXTENT
        )
        mask = wall_stripe & ~door_span
        border = self.BORDER_POSITION
        width = self.BORDER_LINE_WIDTH
        mask[:, border - width : border] = True
        mask[:, self.IMAGE_SIZE - border : self.IMAGE_SIZE - border + width] = True
        mask[border - width : border, :] = True
        mask[self.IMAGE_SIZE - border : self.IMAGE_SIZE - border + width, :] = True
        return mask

    def _inside_door(self, coordinate: float) -> bool:
        extent = self.DOOR_HALF_EXTENT + self.DOOR_COLLISION_MARGIN
        return self.DOOR_CENTER - extent <= coordinate <= self.DOOR_CENTER + extent

    def _apply_collisions(
        self,
        previous: torch.Tensor,
        proposed: torch.Tensor,
    ) -> torch.Tensor:
        lower = self.BORDER_POSITION + self.AGENT_RADIUS
        upper = self.IMAGE_SIZE - self.BORDER_POSITION - self.AGENT_RADIUS
        result = torch.tensor(
            (
                min(max(float(proposed[0]), lower), upper),
                min(max(float(proposed[1]), lower), upper),
            ),
            dtype=torch.float32,
        )
        half_wall = self.WALL_THICKNESS // 2
        left_limit = self.WALL_CENTER - half_wall - self.AGENT_RADIUS
        right_limit = self.WALL_CENTER + half_wall + self.AGENT_RADIUS
        started_left = float(previous[0]) < self.WALL_CENTER
        if (
            started_left
            and float(result[0]) > left_limit
            and not self._inside_door(float(result[1]))
        ):
            result[0] = left_limit - 0.5
        elif (
            not started_left
            and float(result[0]) < right_limit
            and not self._inside_door(float(result[1]))
        ):
            result[0] = right_limit + 0.5
        return result

    def _agent_alpha(self) -> torch.Tensor:
        dx = self.grid_x - float(self.agent_position[0])
        dy = self.grid_y - float(self.agent_position[1])
        alpha = torch.exp(-(dx.square() + dy.square()) / (2.0 * self.AGENT_RADIUS**2))
        maximum = alpha.max()
        if maximum > 0:
            alpha = alpha / maximum
        return alpha

    def render(self) -> np.ndarray:
        image = torch.full(
            (3, self.IMAGE_SIZE, self.IMAGE_SIZE),
            255,
            dtype=torch.uint8,
        )
        image[:, self.wall_mask] = 0
        alpha = self._agent_alpha().clamp_(0.0, 1.0)
        output = image.to(torch.float32)
        agent_color = (255.0, 0.0, 0.0)
        for channel, value in enumerate(agent_color):
            output[channel] = output[channel] * (1.0 - alpha) + value * alpha
        return output.to(torch.uint8).permute(1, 2, 0).numpy()

    def close(self) -> None:
        pass
