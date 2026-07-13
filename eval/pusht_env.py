"""Project-owned fixed Push-T evaluation environment."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d


def _to_surface(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def _lighter(color: SpaceDebugColor) -> SpaceDebugColor:
    values = np.minimum(1.2 * np.asarray((color.r, color.g, color.b, color.a)), 255)
    return SpaceDebugColor(*values.tolist())


class _DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        super().__init__()
        self.surface = surface

    def draw_circle(self, pos, angle, radius, outline_color, fill_color) -> None:
        point = _to_surface(pos)
        pygame.draw.circle(self.surface, fill_color.as_int(), point, round(radius))
        pygame.draw.circle(self.surface, _lighter(fill_color).as_int(), point, round(radius - 4))

    def draw_segment(self, start, end, color) -> None:
        pygame.draw.aalines(
            self.surface,
            color.as_int(),
            False,
            [_to_surface(start), _to_surface(end)],
        )

    def draw_fat_segment(self, start, end, radius, outline_color, fill_color) -> None:
        first, second = _to_surface(start), _to_surface(end)
        width = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [first, second], width)
        if width > 2:
            orthogonal = [abs(second[1] - first[1]), abs(second[0] - first[0])]
            if orthogonal[0] == 0 and orthogonal[1] == 0:
                return
            scale = radius / (orthogonal[0] ** 2 + orthogonal[1] ** 2) ** 0.5
            orthogonal = [round(value * scale) for value in orthogonal]
            points = [
                (first[0] - orthogonal[0], first[1] - orthogonal[1]),
                (first[0] + orthogonal[0], first[1] + orthogonal[1]),
                (second[0] + orthogonal[0], second[1] + orthogonal[1]),
                (second[0] - orthogonal[0], second[1] - orthogonal[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
        pygame.draw.circle(self.surface, fill_color.as_int(), first, round(radius))
        pygame.draw.circle(self.surface, fill_color.as_int(), second, round(radius))

    def draw_polygon(
        self,
        vertices: Sequence[tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        points = [_to_surface(vertex) for vertex in vertices]
        points.append(points[0])
        pygame.draw.polygon(self.surface, _lighter(fill_color).as_int(), points)
        for index, start in enumerate(vertices):
            self.draw_fat_segment(
                start,
                vertices[(index + 1) % len(vertices)],
                2,
                fill_color,
                fill_color,
            )

    def draw_dot(self, size, pos, color) -> None:
        pygame.draw.circle(self.surface, color.as_int(), _to_surface(pos), round(size))


class PushTEnv(gym.Env):
    """The fixed circle-agent/T-block environment used by this benchmark."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, resolution: int = 224) -> None:
        super().__init__()
        self.window_size = 512
        self.render_size = int(resolution)
        self.action_scale = 100.0
        self.control_hz = 10
        self.k_p = 100.0
        self.k_v = 20.0
        self.dt = 0.01
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)
        self.goal_pose = np.asarray((256.0, 256.0, np.pi / 4), dtype=np.float64)
        self.goal_state = np.zeros(7, dtype=np.float64)
        self.latest_action = None
        self._setup()

    def _setup(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0
        walls = [
            pymunk.Segment(self.space.static_body, (5, 506), (5, 5), 2),
            pymunk.Segment(self.space.static_body, (5, 5), (506, 5), 2),
            pymunk.Segment(self.space.static_body, (506, 5), (506, 506), 2),
            pymunk.Segment(self.space.static_body, (5, 506), (506, 506), 2),
        ]
        for wall in walls:
            wall.color = pygame.Color("LightGray")
        self.space.add(*walls)

        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.position = (256, 400)
        agent_shape = pymunk.Circle(self.agent, 15)
        agent_shape.color = pygame.Color("RoyalBlue")
        self.space.add(self.agent, agent_shape)

        mass = 1.0
        horizontal = [(-60, 30), (60, 30), (60, 0), (-60, 0)]
        vertical = [(-15, 30), (-15, 120), (15, 120), (15, 30)]
        # Preserve the released benchmark's inertia calculation, including
        # its repeated horizontal polygon.
        inertia = pymunk.moment_for_poly(mass, horizontal) + pymunk.moment_for_poly(
            mass, horizontal
        )
        self.block = pymunk.Body(mass, inertia)
        block_shapes = (pymunk.Poly(self.block, horizontal), pymunk.Poly(self.block, vertical))
        for shape in block_shapes:
            shape.color = pygame.Color("LightSlateGray")
            shape.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS())
        self.block.center_of_gravity = sum(
            (shape.center_of_gravity for shape in block_shapes),
            Vec2d(0, 0),
        ) / len(block_shapes)
        self.block.position = (400, 100)
        self.block.friction = 1
        self.space.add(self.block, *block_shapes)
        self.space.on_collision(0, 0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._setup()
        options = options or {}
        default = np.asarray((256, 400, 400, 100, 0, 0, 0), dtype=np.float64)
        goal_state = np.asarray(options.get("goal_state", default), dtype=np.float64)
        self.set_goal_state(goal_state)
        state = np.asarray(options.get("state", default), dtype=np.float64)
        self.set_state(state)
        return self.state(), {"pixels": self.render()}

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,):
            raise ValueError("Push-T state must contain seven values")
        self.agent.velocity = state[5:7].tolist()
        self.agent.position = state[:2].tolist()
        # Pymunk position assignment depends on the rotated center of gravity,
        # so preserve the benchmark's angle-before-position ordering.
        self.block.angle = float(state[4])
        self.block.position = state[2:4].tolist()
        self.space.step(self.dt)

    def set_goal_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,):
            raise ValueError("Push-T goal state must contain seven values")
        self.goal_state = state.copy()

    def state(self) -> np.ndarray:
        return np.asarray(
            (
                *self.agent.position,
                *self.block.position,
                self.block.angle % (2 * np.pi),
                *self.agent.velocity,
            ),
            dtype=np.float64,
        )

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (2,):
            raise ValueError("Push-T action must contain two values")
        self.latest_action = action.copy()
        target = self.agent.position + action * self.action_scale
        for _ in range(int(1 / (self.dt * self.control_hz))):
            acceleration = self.k_p * (target - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt
            self.space.step(self.dt)
        state = self.state()
        position_error = np.linalg.norm(self.goal_state[:4] - state[:4])
        angle_error = abs(self.goal_state[4] - state[4])
        angle_error = min(angle_error, 2 * np.pi - angle_error)
        terminated = bool(position_error < 20 and angle_error < np.pi / 9)
        reward = -float(np.linalg.norm(self.goal_state - state))
        return state, reward, terminated, False, {"pixels": self.render()}

    def _goal_body(self) -> pymunk.Body:
        body = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        body.position = self.goal_pose[:2].tolist()
        body.angle = float(self.goal_pose[2])
        return body

    def render(self) -> np.ndarray:
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        draw_options = _DrawOptions(canvas)
        goal_body = self._goal_body()
        for shape in self.block.shapes:
            if isinstance(shape, pymunk.Poly):
                points = [
                    _to_surface(goal_body.local_to_world(vertex)) for vertex in shape.get_vertices()
                ]
                pygame.draw.polygon(canvas, pygame.Color("LightGreen"), points)
        self.space.debug_draw(draw_options)
        pixels = np.transpose(np.asarray(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))
        return cv2.resize(pixels, (self.render_size, self.render_size))

    def close(self) -> None:
        pass
