from __future__ import annotations

import math

import numpy as np


def heuristic_action(observation: np.ndarray) -> int:
    """A deterministic baseline controller.

    It evaluates all three legal impulses and selects the one whose exact
    one-frame prediction places the body closest to upright.
    """

    angle, velocity, difficulty, _speed = map(float, observation)
    gravity = (angle / 3.0) * difficulty / 100.0
    candidates = (
        (abs(angle + velocity + gravity), 0),
        (abs(angle + velocity - difficulty + gravity), 1),
        (abs(angle + velocity + difficulty + gravity), 2),
    )
    return min(candidates)[1]


def pd_action(
    observation: np.ndarray,
    velocity_gain: float = 12.0,
    deadband: float = 12.0,
) -> int:
    """Rule-based PD controller with a dead zone that grows with difficulty.

    Presses the opposite arrow when ``angle + velocity_gain * velocity``
    leaves ``+-deadband * difficulty``. A key press gets stronger as the game
    speeds up (the impulse equals the difficulty), so the band widens with it.
    """

    angle, velocity, difficulty, _speed = map(float, observation)
    signal = angle + velocity_gain * velocity
    band = deadband * difficulty
    if signal > band:
        return 1
    if signal < -band:
        return 2
    return 0


def lookahead_action(
    observation: np.ndarray,
    horizon: int = 2,
    velocity_weight: float = 16.0,
    pulse_frames: int = 1,
) -> int:
    """Rule-based controller that plans over the whole decision interval.

    For each legal input it simulates the original SWF physics ``horizon``
    frames ahead (key held for ``pulse_frames``, then released) and picks the
    input that leaves ``|angle| + velocity_weight * |velocity|`` smallest.
    Unlike ``heuristic_action`` it does not assume a new decision every frame,
    so it stays stable when the real control loop runs every 2-8 frames.
    """

    angle0, velocity0, difficulty0, _speed = map(float, observation)
    best_cost = math.inf
    best_action = 0
    for action in (0, 1, 2):
        angle, velocity, difficulty = angle0, velocity0, difficulty0
        for frame in range(max(1, int(horizon))):
            difficulty += 0.002
            if frame < pulse_frames:
                if action == 1:
                    velocity -= difficulty
                elif action == 2:
                    velocity += difficulty
            velocity += (angle / 3.0) * difficulty / 100.0
            angle += velocity
        cost = abs(angle) + velocity_weight * abs(velocity)
        if cost < best_cost:
            best_cost = cost
            best_action = action
    return best_action
