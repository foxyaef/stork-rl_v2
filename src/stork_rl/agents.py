from __future__ import annotations

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
