"""A compact implementation of Multi-Agent Unilateral Inverse TD."""

from .data import TrajectoryData, load_trajectories, split_episodes
from .model import ActionEncoder, MAUITD

__all__ = [
    "ActionEncoder",
    "MAUITD",
    "TrajectoryData",
    "load_trajectories",
    "split_episodes",
]
