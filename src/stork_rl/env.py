from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StorkEnv(gym.Env[np.ndarray, int]):
    """Faithful headless implementation of the SWF's frame-7 ActionScript.

    Actions:
        0: no input
        1: left arrow
        2: right arrow

    The physics update order intentionally mirrors Flash's enterFrame handler.
    Rendering is optional and never initialized when ``render_mode`` is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    LEFT_FALL_ANGLE = -110.0
    RIGHT_FALL_ANGLE = 80.0
    INITIAL_BALANCE = 1.0
    INITIAL_DIFFICULTY = 1.0
    DIFFICULTY_INCREMENT = 0.002
    SPEED_MULTIPLIER = 3.0
    PIXELS_PER_THREE_METRES = 550.0

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int | None = 18_000,
        realtime: bool = True,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.realtime = realtime

        self.action_space = spaces.Discrete(3)
        # [body angle, balance velocity, difficulty, forward speed].
        # These are raw SWF values, so the observation remains fully Markov.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 1.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.body_rotation = 0.0
        self.balance_velocity = self.INITIAL_BALANCE
        self.difficulty = self.INITIAL_DIFFICULTY
        self.forward_speed = 0.0
        self.distance_pixels = 0.0
        self.metres = 0
        self.step_count = 0
        self.fall_side: str | None = None

        self._pygame: Any = None
        self._screen: Any = None
        self._canvas: Any = None
        self._clock: Any = None
        self._background: Any = None
        self._sprites: dict[int, Any] = {}
        self._closed = False

    @property
    def outer_rotation(self) -> float:
        """Rotation of the parent stork clip in the original SWF."""
        return self.body_rotation / 3.0

    @property
    def distance_metres_exact(self) -> float:
        return self.distance_pixels / self.PIXELS_PER_THREE_METRES * 3.0

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                self.body_rotation,
                self.balance_velocity,
                self.difficulty,
                self.forward_speed,
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "metres": self.metres,
            "distance_metres_exact": self.distance_metres_exact,
            "distance_pixels": self.distance_pixels,
            "body_rotation": self.body_rotation,
            "outer_rotation": self.outer_rotation,
            "balance_velocity": self.balance_velocity,
            "difficulty": self.difficulty,
            "forward_speed": self.forward_speed,
            "fall_side": self.fall_side,
            "steps": self.step_count,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.body_rotation = 0.0
        self.balance_velocity = self.INITIAL_BALANCE
        self.difficulty = self.INITIAL_DIFFICULTY
        self.forward_speed = 0.0
        self.distance_pixels = 0.0
        self.metres = 0
        self.step_count = 0
        self.fall_side = None
        self._closed = False

        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be 0, 1, or 2; got {action!r}")

        previous_distance = self.distance_pixels

        # Exact ActionScript order:
        # _root.xx += xs; metres = floor(xx / 550 * 3)
        self.distance_pixels += self.forward_speed
        self.metres = int(np.floor(self.distance_metres_exact))

        # xs = bi * 3; bi += 0.002
        self.forward_speed = self.difficulty * self.SPEED_MULTIPLIER
        self.difficulty += self.DIFFICULTY_INCREMENT

        # Key input is applied before the fall check in the SWF.
        if action == 1:
            self.balance_velocity -= self.difficulty
        elif action == 2:
            self.balance_velocity += self.difficulty

        terminated = False
        if self.body_rotation < self.LEFT_FALL_ANGLE:
            self.fall_side = "left"
            terminated = True
        elif self.body_rotation > self.RIGHT_FALL_ANGLE:
            self.fall_side = "right"
            terminated = True

        if terminated:
            # The later `going == "no"` block zeros both values in this frame.
            self.balance_velocity = 0.0
            self.forward_speed = 0.0
        else:
            # bs += _rotation * bi / 100; body._rotation += bs
            self.balance_velocity += (
                self.outer_rotation * self.difficulty / 100.0
            )
            self.body_rotation += self.balance_velocity

        self.step_count += 1
        truncated = (
            self.max_steps is not None
            and self.step_count >= self.max_steps
            and not terminated
        )

        # Maximizing this is exactly maximizing distance. It is continuous
        # rather than the displayed integer metre counter to avoid sparse reward.
        reward = (
            (self.distance_pixels - previous_distance)
            / self.PIXELS_PER_THREE_METRES
            * 3.0
        )

        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _asset_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "assets"

    def _init_renderer(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        pygame.init()
        self._pygame = pygame
        self._canvas = pygame.Surface((550, 400), pygame.SRCALPHA)
        self._clock = pygame.time.Clock()
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((550, 400))
            pygame.display.set_caption("Walk the Stork RL")

        self._background = pygame.image.load(
            str(self._asset_root() / "background.png")
        )
        if self.render_mode == "human":
            self._background = self._background.convert_alpha()
        for frame in range(1, 26):
            sprite = pygame.image.load(
                str(self._asset_root() / "stork" / f"{frame}.png")
            )
            self._sprites[frame] = (
                sprite.convert_alpha() if self.render_mode == "human" else sprite
            )

    def _sprite_frame(self) -> int:
        if self.fall_side == "right":
            return 19
        if self.fall_side == "left":
            return 23
        # The SWF loops walking frames 1..17 with conditional skips as speed rises.
        return 1 + (self.step_count % 17)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        self._init_renderer()
        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True

        # The instruction frame contains the exact clean scenery below y=185.
        self._canvas.fill((255, 255, 255, 255))
        scenery = self._background.subsurface((0, 185, 550, 215))
        self._canvas.blit(scenery, (0, 185))

        font = pygame.font.SysFont("arial", 34)
        score = font.render(f"{self.metres}m", True, (15, 15, 15))
        self._canvas.blit(score, (38, 28))

        sprite = self._sprites[self._sprite_frame()]
        # FFDec exported the sprite at its original 387x229 registration bounds.
        # Rotating by the parent clip angle reproduces the visible lean.
        rotated = pygame.transform.rotozoom(sprite, -self.outer_rotation, 1.0)
        base_rect = sprite.get_rect(topleft=(107, 176))
        rotated_rect = rotated.get_rect(center=base_rect.center)
        self._canvas.blit(rotated, rotated_rect)

        if self.render_mode == "human":
            self._screen.blit(self._canvas, (0, 0))
            pygame.display.flip()
            if self.realtime:
                self._clock.tick(self.metadata["render_fps"])
            return None

        pixels = pygame.surfarray.array3d(self._canvas)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def poll_human_action(self) -> int:
        """Return the current held-key action for the manual simulator."""
        self._init_renderer()
        keys = self._pygame.key.get_pressed()
        if keys[self._pygame.K_LEFT]:
            return 1
        if keys[self._pygame.K_RIGHT]:
            return 2
        return 0

    @property
    def user_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._canvas = None
