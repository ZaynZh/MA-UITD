"""Closed-loop evaluation for the three MA-UITD tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch


@torch.no_grad()
def choose_action(model, states, candidates, agent, mode, generator):
    if candidates.ndim == 2:
        candidates = candidates.expand(len(states), -1, -1)
    count = candidates.shape[1]
    scores = model.q(
        states.repeat_interleave(count, dim=0),
        candidates.reshape(-1, candidates.shape[-1]),
        agent,
    ).reshape(len(states), count)
    index = scores.argmax(1) if mode == "greedy" else torch.multinomial(
        scores.softmax(1), 1, generator=generator
    ).squeeze(1)
    return candidates[torch.arange(len(states), device=states.device), index]


@torch.no_grad()
def choose_discrete_action(model, states, agent, mode, generator):
    scores = model.all_discrete_q(states, agent)
    if mode == "greedy":
        return scores.argmax(dim=1)
    return torch.multinomial(scores.softmax(dim=1), 1, generator=generator).squeeze(1)


def spread_reward(positions, landmarks):
    distance = np.linalg.norm(positions[:, :, None] - landmarks[:, None], axis=-1)
    coverage = -distance.min(axis=1).sum(axis=1)
    pair_distance = np.linalg.norm(
        positions[:, :, None] - positions[:, None, :], axis=-1
    )
    collision = np.minimum(pair_distance - .3, 0).sum(axis=(1, 2))
    return 2 * coverage + collision


def spread_reward_input(positions, landmarks):
    local = np.linalg.norm(positions[:, :, None] - landmarks[:, None], axis=-1)
    pair = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1)[:, None]
    return np.concatenate((local.reshape(len(positions), -1), pair), axis=1)


def spread_state(positions, velocities, landmarks):
    relative = (positions[:, :, None] - landmarks[:, None]).flatten(1)
    pair = positions[:, 0] - positions[:, 1]
    return torch.cat((relative, pair, velocities.flatten(1)), dim=1).float()


def spread_step(positions, velocities, raw_actions):
    applied = raw_actions.clamp(0, 1)
    force = 5 * (applied[..., [1, 3]] - applied[..., [2, 4]])
    delta = positions[:, 0] - positions[:, 1]
    distance = torch.linalg.vector_norm(delta, dim=1)
    penetration = torch.nn.functional.softplus(-(distance - .3) / .001) * .001
    contact = 100 * delta / distance.clamp_min(1e-12)[:, None] * penetration[:, None]
    collision = torch.stack((contact, -contact), dim=1)
    next_velocity = .75 * velocities + .1 * (force + collision)
    return positions + .1 * next_velocity, next_velocity


@torch.no_grad()
def rollout_spread(model, episodes, mode):
    device = model.reward_weights.device
    initial = np.stack([
        np.random.RandomState(50_000 + episode).uniform(-1, 1, (4, 2))
        for episode in range(episodes)
    ])
    positions = torch.as_tensor(initial[:, :2], dtype=torch.float64, device=device)
    velocities = torch.zeros_like(positions)
    landmarks = initial[:, 2:]
    policy = torch.Generator(device=device).manual_seed(60_000)
    records = {key: [] for key in ("truth", "inferred", "episode_id", "step_id", "next_positions", "landmarks")}
    active = np.ones(episodes, dtype=bool)
    success = np.zeros(episodes, dtype=bool)
    lengths = np.zeros(episodes, dtype=np.int64)
    returns = np.zeros(episodes)

    for step in range(50):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        selected = torch.as_tensor(ids, device=device)
        states = spread_state(
            positions[selected],
            velocities[selected],
            torch.as_tensor(landmarks[ids], device=device),
        )
        actions = torch.stack(
            [
                choose_discrete_action(model, states, agent, mode, policy)
                for agent in range(2)
            ],
            dim=1,
        )
        controls = torch.nn.functional.one_hot(actions, 5).double()
        next_positions, next_velocities = spread_step(
            positions[selected], velocities[selected], controls
        )
        positions[selected], velocities[selected] = next_positions, next_velocities
        next_numpy = next_positions.cpu().numpy()
        truth = spread_reward(next_numpy, landmarks[ids])
        reward_input = spread_reward_input(next_numpy, landmarks[ids])
        inferred = model.reward(torch.as_tensor(reward_input, dtype=torch.float32, device=device)).cpu().numpy()
        distance = np.linalg.norm(next_numpy[:, :, None] - landmarks[ids, None], axis=-1)
        covered = (distance.min(axis=1) <= .1).all(axis=1)
        records["truth"].append(np.repeat(truth[:, None], 2, axis=1))
        records["inferred"].append(inferred)
        records["episode_id"].append(ids)
        records["step_id"].append(np.full(len(ids), step))
        records["next_positions"].append(next_numpy)
        records["landmarks"].append(landmarks[ids])
        lengths[ids] += 1
        returns[ids] += truth
        success[ids] = covered
        active[ids[covered]] = False

    arrays = {key: np.concatenate(value) for key, value in records.items()}
    metrics = {
        "name": mode,
        "selection": mode,
        "episodes": episodes,
        "final_coverage_rate": float(success.mean()),
        "ever_coverage_rate": float(success.mean()),
        "mean_episode_length": float(lengths.mean()),
        "mean_true_undiscounted_return": float(returns.mean()),
    }
    return arrays, metrics


class PushExpert:
    """The categorical ACKTR actor that generated the Push dataset."""

    def __init__(self, checkpoint, checksum):
        checkpoint = Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Expert checkpoint checksum mismatch: {checkpoint}")
        self.parameters = joblib.load(checkpoint)

    def act(self, observation, agent, generator):
        offset = 12 * agent
        w1, b1, w2, b2, w3, b3 = self.parameters[offset : offset + 6]
        hidden = np.maximum(observation @ w1 + b1, 0)
        logits = np.maximum(hidden @ w2 + b2, 0) @ w3 + b3
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        action = np.array([generator.choice(5, p=row) for row in probability])
        native = np.eye(5)[action]
        return native[:, [0, 2, 1, 4, 3]]


def push_observations(positions, velocities, landmark):
    goal = landmark[:, 0]
    adversary = np.concatenate((velocities[:, 0], goal - positions[:, 0], positions[:, 1] - positions[:, 0]), axis=1)
    good = np.concatenate((
        velocities[:, 1], goal - positions[:, 1],
        np.tile((.25, .75, .25), (len(positions), 1)),
        goal - positions[:, 1], np.tile((.1, .9, .1), (len(positions), 1)),
        positions[:, 0] - positions[:, 1],
    ), axis=1)
    return adversary, good


def push_step(positions, velocities, controls, force_scales, learned_agent, learned_force):
    direction = np.stack((controls[..., 1] - controls[..., 2], controls[..., 3] - controls[..., 4]), axis=-1)
    motor_force = force_scales[None, :, None] * direction
    if learned_agent is not None:
        motor_force[:, learned_agent] = learned_force
    delta = positions[:, 0] - positions[:, 1]
    distance = np.linalg.norm(delta, axis=1)
    penetration = np.logaddexp(0, -(distance - .1) / .001) * .001
    contact = 100 * delta / np.maximum(distance[:, None], np.finfo(float).tiny) * penetration[:, None]
    force = np.stack((contact, -contact), axis=1) + motor_force
    next_velocity = .75 * velocities + .1 * force
    return positions + .1 * next_velocity, next_velocity


def push_reward(positions, landmark):
    distance = np.linalg.norm(positions - landmark[:, None], axis=-1)
    return np.stack((distance[:, 1] - distance[:, 0], -distance[:, 1]), axis=1)


@torch.no_grad()
def rollout_push_matchup(
    model, source, initial_rows, learned_agent, mode, episodes,
    expert_checkpoint, expert_checksum, action_force_directions,
    action_accelerations,
):
    device = model.reward_weights.device
    expert = PushExpert(expert_checkpoint, expert_checksum)
    positions = source["agent_positions"][initial_rows].astype(np.float64).copy()
    velocities = source["agent_velocities"][initial_rows].astype(np.float64).copy()
    landmarks = source["landmark_positions"][initial_rows].astype(np.float64).copy()
    force_directions = np.asarray(action_force_directions, dtype=np.float64)
    force_scales = np.asarray(action_accelerations, dtype=np.float64)
    policy = torch.Generator(device=device).manual_seed(600_000 + (learned_agent or 0))
    expert_generator = np.random.RandomState(700_000)
    truth, inferred = [], []

    for _step in range(25):
        observations = push_observations(positions, velocities, landmarks)
        controls = np.zeros((episodes, 2, 5), dtype=np.float64)
        learned_force = None
        for agent in range(2):
            if agent == learned_agent:
                state = torch.as_tensor(
                    np.concatenate(
                        (
                            positions.reshape(episodes, -1),
                            velocities.reshape(episodes, -1),
                            landmarks.reshape(episodes, -1),
                        ),
                        axis=1,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                action_id = choose_discrete_action(model, state, agent, mode, policy).cpu().numpy()
                learned_force = (
                    force_directions[action_id] * force_scales[agent]
                )
                controls[:, agent, 0] = 1
            else:
                controls[:, agent] = expert.act(observations[agent], agent, expert_generator)
        positions, velocities = push_step(
            positions, velocities, controls, force_scales, learned_agent, learned_force
        )
        reward = push_reward(positions, landmarks[:, 0])
        state = np.concatenate((positions.reshape(episodes, -1), landmarks.reshape(episodes, -1)), axis=1)
        predicted = model.reward(torch.as_tensor(state, dtype=torch.float32, device=device)).cpu().numpy()
        truth.append(reward)
        inferred.append(predicted)

    truth = np.stack(truth, axis=1)
    inferred = np.stack(inferred, axis=1)
    good_distance = -truth[:, :, 1]
    arrays = {
        "truth": truth.reshape(-1, 2),
        "inferred": inferred.reshape(-1, 2),
        "episode_id": np.repeat(np.arange(episodes), 25),
        "step_id": np.tile(np.arange(25), episodes),
    }
    label = "expert_expert" if learned_agent is None else ("learned_adversary" if learned_agent == 0 else "learned_good")
    metrics = {
        "name": f"{label}_{mode}",
        "selection": mode,
        "learned_agent": learned_agent,
        "episodes": episodes,
        "mean_true_return_per_agent": truth.sum(axis=1).mean(axis=0).tolist(),
        "goal_success_rate_ever": float((good_distance.min(axis=1) <= .1).mean()),
        "goal_success_rate_final": float((good_distance[:, -1] <= .1).mean()),
    }
    return arrays, metrics


@torch.no_grad()
def rollout_gems(model, episodes, mode, candidates):
    from .gems import GemsConfig, make_env
    device = model.reward_weights.device
    config = GemsConfig()
    discrete = model.action_encoder.kind == "discrete"
    env = make_env(
        config,
        num_envs=episodes,
        continuous_actions=not discrete,
        seed=50_000,
        device=device,
    )
    observation = env.reset()[0]
    generator = torch.Generator(device=device).manual_seed(60_000)
    active = np.ones(episodes, dtype=bool)
    success = np.zeros(episodes, dtype=bool)
    returns = np.zeros((episodes, 2))
    lengths = np.zeros(episodes, dtype=np.int64)
    events_total = np.zeros(3, dtype=np.int64)
    records = {key: [] for key in ("truth", "inferred", "episode_id", "step_id")}

    for step in range(config.horizon):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        selected = torch.as_tensor(ids, device=device)
        states = observation[selected, :4]
        if discrete:
            actions = torch.stack(
                [
                    choose_discrete_action(model, states, agent, mode, generator)
                    for agent in range(2)
                ],
                dim=1,
            )
            full_actions = torch.full(
                (episodes, 2), 4, dtype=torch.long, device=device
            )
        else:
            banks = [
                (2 * torch.rand(len(ids), candidates, 2, device=device, generator=generator) - 1)
                * config.action_limit
                for _ in range(2)
            ]
            actions = torch.stack(
                [
                    choose_action(model, states, banks[agent], agent, mode, generator)
                    for agent in range(2)
                ],
                dim=1,
            )
            full_actions = torch.zeros((episodes, 2, 2), device=device)
        full_actions[selected] = actions
        following, rewards, done, infos = env.step([full_actions[:, agent] for agent in range(2)])
        observation = following[0]
        truth = torch.stack(rewards, dim=1)[selected].cpu().numpy()
        inferred = model.reward(observation[selected, :4]).cpu().numpy()
        records["truth"].append(truth)
        records["inferred"].append(inferred)
        records["episode_id"].append(ids)
        records["step_id"].append(np.full(len(ids), step))
        returns[ids] += truth
        lengths[ids] += 1
        events = np.column_stack((
            infos[0]["own_collected"][selected].cpu().numpy(),
            infos[1]["own_collected"][selected].cpu().numpy(),
            infos[0]["purple_collected"][selected].cpu().numpy(),
        ))
        events_total += events.sum(axis=0)
        success[ids] = infos[0]["terminated"][selected].cpu().numpy()
        finished = done[selected].cpu().numpy()
        active[ids[finished]] = False

    arrays = {key: np.concatenate(value) for key, value in records.items()}
    metrics = {
        "name": mode,
        "selection": mode,
        "episodes": episodes,
        "success_rate": float(success.mean()),
        "mean_episode_length": float(lengths.mean()),
        "mean_true_return_per_agent": returns.mean(axis=0).tolist(),
        "collection_counts": events_total.tolist(),
        "purple_collection_fraction": float(events_total[2] / events_total.sum()) if events_total.sum() else None,
    }
    return arrays, metrics


def evaluate_policy(model, config, source, split, output):
    episodes = config["evaluation"]["episodes"]
    candidates = config["evaluation"]["candidates"]
    task = config["name"]
    datasets, metrics = {}, []

    if task == "simple_spread_discrete":
        for mode in ("greedy", "boltzmann"):
            datasets[mode], result = rollout_spread(model, episodes, mode)
            metrics.append(result)
    elif task == "simple_push_discrete":
        test_episodes = np.unique(source["episode_id"][split.test])[:episodes]
        initial_rows = np.array([np.flatnonzero(source["episode_id"] == episode)[0] for episode in test_episodes])
        expert_checkpoint = config["evaluation"]["expert_checkpoint"]
        expert_checksum = config["evaluation"]["expert_checkpoint_sha256"]
        datasets["expert_expert"], result = rollout_push_matchup(
            model, source, initial_rows, None, "boltzmann", episodes,
            expert_checkpoint, expert_checksum,
            config["evaluation"]["action_force_directions"],
            config["evaluation"]["action_accelerations"],
        )
        metrics.append(result)
        for mode in ("greedy", "boltzmann"):
            for agent in range(2):
                name = ("learned_adversary" if agent == 0 else "learned_good") + "_" + mode
                datasets[name], result = rollout_push_matchup(
                    model, source, initial_rows, agent, mode, episodes,
                    expert_checkpoint, expert_checksum,
                    config["evaluation"]["action_force_directions"],
                    config["evaluation"]["action_accelerations"],
                )
                metrics.append(result)
    else:
        for mode in ("greedy", "boltzmann"):
            datasets[mode], result = rollout_gems(model, episodes, mode, candidates)
            metrics.append(result)

    for name, arrays in datasets.items():
        np.savez_compressed(output / f"{name}_rollouts.npz", **arrays)
    return datasets, metrics
