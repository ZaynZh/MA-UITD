"""Canonical trajectory data and the three paper-dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrajectoryData:
    """The only transition representation consumed by MA-UITD."""

    states: torch.Tensor
    arrival_states: torch.Tensor
    actions: torch.Tensor
    episode_id: np.ndarray
    step_id: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __len__(self) -> int:
        return len(self.episode_id)


@dataclass
class EpisodeSplit:
    training: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class NStepWindows:
    step_rows: torch.Tensor
    step_weights: torch.Tensor
    bootstrap_rows: torch.Tensor
    bootstrap_discount: torch.Tensor
    eligible: np.ndarray


def _tensor(array: np.ndarray, device: torch.device, discrete: bool = False) -> torch.Tensor:
    dtype = torch.long if discrete else torch.float32
    return torch.as_tensor(array, dtype=dtype, device=device)


def _spread(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = source["agent_positions"]
    arrival = source["next_agent_positions"]
    landmarks = source["landmark_positions"]
    local_distance = np.linalg.norm(arrival[:, :, None] - landmarks[:, None], axis=-1)
    pair_distance = np.linalg.norm(arrival[:, 0] - arrival[:, 1], axis=-1)[:, None]
    reward_state = np.concatenate((local_distance.reshape(len(position), -1), pair_distance), axis=1)

    relative = (position[:, :, None] - landmarks[:, None]).reshape(len(position), -1)
    pair = position[:, 0] - position[:, 1]
    velocity = source["agent_velocities"].reshape(len(position), -1)
    state = np.concatenate((relative, pair, velocity), axis=1)
    return state, reward_state, source["actions"], source["terminated"], source["truncated"]


def _push(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    landmarks = source["landmark_positions"].reshape(len(source["episode_id"]), -1)
    state = np.concatenate(
        (
            source["agent_positions"].reshape(len(landmarks), -1),
            source["agent_velocities"].reshape(len(landmarks), -1),
            landmarks,
        ),
        axis=1,
    )
    reward_state = np.concatenate(
        (source["next_agent_positions"].reshape(len(landmarks), -1), landmarks), axis=1
    )
    return state, reward_state, source["actions"], source["terminated"], source["truncated"]


def _gems(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = source["states"][:, :4]
    reward_state = source["next_states"][:, :4]
    return state, reward_state, source["actions"], source["terminated"], source["truncated"]


def load_trajectories(
    dataset: dict,
    action_kind: str,
    action_size: int,
    device: str | torch.device,
) -> TrajectoryData:
    """Load a paper dataset and convert it to the common MA-UITD representation."""

    path = Path(dataset["path"])
    if dataset.get("sha256"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != dataset["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path}")

    with np.load(path, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}

    format_name = dataset["format"]
    if format_name == "simple_spread_discrete":
        state, arrival, actions, terminated, truncated = _spread(source)
    elif format_name == "simple_push_discrete":
        state, arrival, actions, terminated, truncated = _push(source)
    elif format_name == "gems_discrete":
        state, arrival, actions, terminated, truncated = _gems(source)
    elif format_name == "canonical":
        state = source["states"]
        arrival = source["arrival_states"]
        actions = source["actions"]
        terminated = source["terminated"]
        truncated = source["truncated"]
    else:
        raise ValueError(f"Unknown dataset format: {format_name}")

    recorded_continuous = source.get("continuous_actions")
    recorded_mode = source.get("action_mode")
    if recorded_continuous is not None or recorded_mode is not None:
        recorded_kind = (
            str(recorded_mode.item())
            if recorded_mode is not None
            else ("continuous" if bool(recorded_continuous) else "discrete")
        )
        if action_kind != recorded_kind:
            raise ValueError(
                f"Dataset contains {recorded_kind} actions but the configuration requests {action_kind}"
            )
    if action_kind == "discrete":
        if actions.ndim != 2 or actions.dtype.kind not in "iu":
            raise ValueError("Discrete actions must be an integer [transition, agent] array")
        if actions.min() < 0 or actions.max() >= action_size:
            raise ValueError(f"Discrete action IDs must lie in [0, {action_size})")
    elif actions.ndim != 3 or actions.shape[-1] != action_size:
        raise ValueError(
            f"Continuous actions must have shape [transition, agent, {action_size}]"
        )

    torch_device = torch.device(device)
    arrival_tensor = _tensor(arrival, torch_device)
    if format_name == "simple_spread_discrete":
        next_positions = _tensor(source["next_agent_positions"], torch_device)
        landmarks = _tensor(source["landmark_positions"], torch_device)
        local_distance = torch.linalg.vector_norm(
            next_positions[:, :, None] - landmarks[:, None], dim=-1
        )
        pair_distance = torch.linalg.vector_norm(
            next_positions[:, 0] - next_positions[:, 1], dim=-1
        )[:, None]
        arrival_tensor = torch.cat((local_distance.flatten(1), pair_distance), dim=1)
    return TrajectoryData(
        states=_tensor(state, torch_device),
        arrival_states=arrival_tensor,
        actions=_tensor(actions, torch_device, discrete=action_kind == "discrete"),
        episode_id=np.asarray(source["episode_id"]),
        step_id=np.asarray(source["step_id"]),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def split_episodes(
    episode_id: np.ndarray,
    training: int,
    validation: int,
    test: int,
    seed: int,
) -> EpisodeSplit:
    episodes = np.random.default_rng(seed).permutation(np.unique(episode_id))
    train_ids = episodes[:training]
    validation_ids = episodes[training : training + validation]
    test_ids = episodes[training + validation : training + validation + test]

    def rows(ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(episode_id, ids))

    return EpisodeSplit(rows(train_ids), rows(validation_ids), rows(test_ids))


def build_n_step_windows(data: TrajectoryData, n_step: int, gamma: float) -> NStepWindows:
    """Build full n-step windows without crossing an episode boundary."""

    size = len(data)
    following = np.full(size, -1, dtype=np.int64)
    for episode in np.unique(data.episode_id):
        rows = np.flatnonzero(data.episode_id == episode)
        rows = rows[np.argsort(data.step_id[rows])]
        following[rows[:-1]] = rows[1:]

    step_rows = np.zeros((size, n_step), dtype=np.int64)
    step_weights = np.zeros((size, n_step), dtype=np.float32)
    bootstrap_rows = np.zeros(size, dtype=np.int64)
    bootstrap_discount = np.zeros((size, 1), dtype=np.float32)
    eligible = np.ones(size, dtype=bool)

    for root in range(size):
        row = root
        for offset in range(n_step):
            step_rows[root, offset] = row
            step_weights[root, offset] = gamma**offset
            if data.terminated[row]:
                break
            if data.truncated[row] or following[row] < 0:
                eligible[root] = False
                break
            row = following[row]
            if offset == n_step - 1:
                bootstrap_rows[root] = row
                bootstrap_discount[root] = gamma**n_step

    device = data.states.device
    return NStepWindows(
        step_rows=torch.as_tensor(step_rows, device=device),
        step_weights=torch.as_tensor(step_weights, device=device),
        bootstrap_rows=torch.as_tensor(bootstrap_rows, device=device),
        bootstrap_discount=torch.as_tensor(bootstrap_discount, device=device),
        eligible=eligible,
    )
