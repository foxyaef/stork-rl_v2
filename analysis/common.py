"""Shared helpers for the analysis scripts.

Run the scripts from the repository root, e.g. ``python analysis/rule_compare.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stork_rl.agents import heuristic_action, lookahead_action, pd_action  # noqa: E402

V9_MODEL = ROOT / "models_browser_v9" / "stork_ppo.zip"
V9_STATS = ROOT / "models_browser_v9" / "vec_normalize.pkl"


def load_ppo(model: Path = V9_MODEL, stats: Path = V9_STATS):
    """Deterministic PPO policy (observation -> action) without torch.

    The numpy loader reproduces ``model.predict(normalized_obs, deterministic=True)``
    of the saved Stable-Baselines3 MlpPolicy and VecNormalize statistics.
    """

    from np_policy import NumpyPPO

    return NumpyPPO(str(model), str(stats))


def make_agent(name: str):
    """Agent names: ppo, heuristic, lookahead, pd, pd:<gain>:<deadband>."""

    if name == "ppo":
        return load_ppo()
    if name == "heuristic":
        return heuristic_action
    if name == "lookahead":
        return lambda obs: lookahead_action(obs, horizon=2, velocity_weight=16.0)
    if name.startswith("pd"):
        parts = name.split(":")
        gain, band = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (12.0, 12.0)
        return lambda obs: pd_action(obs, velocity_gain=gain, deadband=band)
    raise ValueError(f"unknown agent {name!r}")
