#!/usr/bin/env python3
"""Evaluate MA-UITD rewards and recovered policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

from ..data import load_trajectories, split_episodes
from .rollouts import evaluate_policy, push_reward, spread_reward, spread_reward_input


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def association(truth: np.ndarray, inferred: np.ndarray) -> dict:
    if np.ptp(truth) < 1e-12 or np.ptp(inferred) < 1e-12:
        return {"pcc": None, "scc": None}
    return {
        "pcc": float(pearsonr(truth, inferred).statistic),
        "scc": float(spearmanr(truth, inferred).statistic),
    }


def discounted_returns(rewards, episode_id, step_id, gamma):
    result = np.empty_like(rewards, dtype=np.float64)
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(step_id[rows])]
        running = np.zeros(rewards.shape[1])
        for row in rows[::-1]:
            running = rewards[row] + gamma * running
            result[row] = running
    return result


@torch.no_grad()
def predict(model, values: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    device = model.reward_weights.device
    batches = []
    for start in range(0, len(values), batch_size):
        x = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32, device=device)
        batches.append(model.reward(x).cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def dataset_record(model, data, source, rows, gamma):
    truth = np.asarray(source["rewards"][rows], dtype=np.float64)
    reward_input = data.arrival_states[rows].detach().cpu().numpy()
    inferred = predict(model, reward_input)
    episode_id = source["episode_id"][rows]
    step_id = source["step_id"][rows]
    true_rtg = discounted_returns(truth, episode_id, step_id, gamma)
    learned_rtg = discounted_returns(inferred, episode_id, step_id, gamma)
    arrays = {
        "truth": truth,
        "inferred": inferred,
        "true_rtg": true_rtg,
        "learned_rtg": learned_rtg,
        "reward_input": reward_input,
        "source_rows": rows,
        "episode_id": episode_id,
        "step_id": step_id,
    }
    return arrays


def binned(x: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    centers = np.array([x[group].mean() for group in groups])
    means = np.stack([y[group].mean(axis=0) for group in groups])
    counts = np.array([len(group) for group in groups])
    return centers, means, counts


def equal_width_binned(
    distance: np.ndarray,
    truth: np.ndarray,
    learned: np.ndarray,
    bins: int = 40,
) -> dict[str, np.ndarray]:
    """Mean true and learned rewards in occupied equal-width distance bins."""

    edges = np.linspace(distance.min(), distance.max(), bins + 1)
    bin_id = np.clip(np.searchsorted(edges, distance, side="right") - 1, 0, bins - 1)
    counts = np.bincount(bin_id, minlength=bins)
    occupied = counts > 0

    result = {
        "edges": edges,
        "occupied_bins": np.flatnonzero(occupied),
        "counts": counts[occupied],
    }
    for name, values in (
        ("distance", distance),
        ("true_reward", truth),
        ("learned_reward", learned),
    ):
        result[name] = (
            np.bincount(bin_id, weights=values, minlength=bins)[occupied]
            / counts[occupied]
        )
    return result


def push_reward_distance_figure(source, test_rows, test):
    """Build the three task-relevant Simple Push test-distance panels."""

    positions = source["next_agent_positions"][test_rows]
    landmark = source["landmark_positions"][test_rows, 0]
    distances = np.linalg.norm(positions - landmark[:, None], axis=-1)
    panels = (
        ("adversary_vs_good", 0, 1, "Adversary reward", "Good–landmark distance"),
        ("adversary_vs_adversary", 0, 0, "Adversary reward", "Adversary–landmark distance"),
        ("good_vs_good", 1, 1, "Good reward", "Good–landmark distance"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.subplots_adjust(left=.07, right=.99, bottom=.24, top=.90, wspace=.34)
    arrays = {
        "source_rows": test_rows,
        "episode_id": source["episode_id"][test_rows],
        "step_id": source["step_id"][test_rows],
        "distances": distances,
        "truth": test["truth"],
        "inferred": test["inferred"],
    }
    legend_handles = None
    for axis, (name, agent, distance_agent, title, xlabel) in zip(axes, panels):
        values = equal_width_binned(
            distances[:, distance_agent],
            test["truth"][:, agent],
            test["inferred"][:, agent],
        )
        arrays.update({f"{name}_{key}": value for key, value in values.items()})
        true_line, = axis.plot(values["distance"], values["true_reward"], color="black", label="True reward")
        learned_line, = axis.plot(values["distance"], values["learned_reward"], color="#2878B5", label="Learned reward")
        legend_handles = (true_line, learned_line)
        axis.set(title=title, xlabel=xlabel, ylabel="Mean raw reward")
        axis.grid(alpha=.2)
    fig.legend(
        legend_handles,
        ("True reward", "Learned reward"),
        loc="lower center",
        bbox_to_anchor=(.5, .01),
        ncol=2,
    )
    return fig, arrays


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def task_distances(config, source, rows):
    task = config["name"]
    if task == "simple_spread_discrete":
        values = spread_reward_input(source["next_agent_positions"][rows], source["landmark_positions"][rows])
        names = ("A1-L1", "A1-L2", "A2-L1", "A2-L2", "A1-A2")
    elif task == "simple_push_discrete":
        positions = source["next_agent_positions"][rows]
        landmark = source["landmark_positions"][rows, 0]
        values = np.column_stack((
            np.linalg.norm(positions[:, 0] - landmark, axis=1),
            np.linalg.norm(positions[:, 1] - landmark, axis=1),
            np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1),
        ))
        names = ("Adversary-target", "Good-target", "Adversary-good")
    else:
        states = source["next_states"][rows]
        agents = states[:, :4].reshape(-1, 2, 2)
        gems = states[:, 4:10].reshape(-1, 3, 2)
        values = np.linalg.norm(agents[:, :, None] - gems[:, None], axis=-1).reshape(-1, 6)
        names = ("Red-red", "Red-blue", "Red-purple", "Blue-red", "Blue-blue", "Blue-purple")
    return values, names


def spatial_values(model, config, source, row):
    task = config["name"]
    if task == "gems_discrete":
        axis = np.linspace(0, 1, 65)
        base = source["next_states"][row]
        positions = base[:4].reshape(2, 2)
        gems = base[4:10].reshape(3, 2)
        radius = .04
    else:
        axis = np.linspace(-1.3, 1.3, 65)
        positions = source["next_agent_positions"][row]
        gems = source["landmark_positions"][row]
    xx, yy = np.meshgrid(axis, axis)
    results = []
    moving_agents = (0,) if task == "gems_discrete" else range(2)
    for moving in moving_agents:
        moved = np.broadcast_to(positions, (xx.size, 2, 2)).copy()
        if task == "gems_discrete":
            moved[:, 1] = gems[2]
        moved[:, moving] = np.column_stack((xx.ravel(), yy.ravel()))
        fixed_gems = np.broadcast_to(gems, (xx.size, *gems.shape)).copy()
        if task == "simple_spread_discrete":
            scalar = spread_reward(moved, fixed_gems)
            truth = np.repeat(scalar[:, None], 2, axis=1)
            reward_input = spread_reward_input(moved, fixed_gems)
        elif task == "simple_push_discrete":
            truth = push_reward(moved, fixed_gems[:, 0])
            reward_input = np.concatenate((moved.reshape(xx.size, -1), fixed_gems.reshape(xx.size, -1)), axis=1)
        else:
            distance = np.linalg.norm(moved[:, :, None] - fixed_gems[:, None], axis=-1)
            private = np.column_stack((distance[:, 0, 0] <= radius, distance[:, 1, 1] <= radius))
            purple = (distance[:, :, 2] <= radius).all(axis=1)
            truth = private + 6 * purple[:, None]
            reward_input = moved.reshape(xx.size, -1)
        results.append((moving, moved, fixed_gems, truth, predict(model, reward_input)))
    return xx, yy, results


def gems_counterfactual_spatial_figure(xx, yy, slices):
    """Recovered rewards while Red moves and Blue remains at the purple gem."""

    moving, positions, gems, truth, inferred = slices[0]
    assert moving == 0
    assert np.allclose(positions[:, 1], gems[:, 2])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), layout="constrained")
    gem_colors = ("#D64B40", "#3F78B5", "#8E5AA7")
    titles = ("Red-agent reward", "Blue-agent reward")
    for agent, (axis, title) in enumerate(zip(axes, titles)):
        image = axis.imshow(
            inferred[:, agent].reshape(xx.shape),
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap="coolwarm",
        )
        for center, color in zip(gems[0], gem_colors):
            axis.add_patch(plt.Circle(
                center, 0.04, facecolor="none", edgecolor="white", linewidth=2.4,
            ))
            axis.add_patch(plt.Circle(
                center, 0.04, facecolor="none", edgecolor=color, linewidth=1.4,
            ))
        axis.set(
            title=title,
            xlabel="x",
            ylabel="y",
            xlim=(xx.min(), xx.max()),
            ylim=(yy.min(), yy.max()),
            aspect="equal",
        )
        axis.text(
            -0.12, 1.04, chr(ord("a") + agent),
            transform=axis.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom",
        )
        colorbar = fig.colorbar(image, ax=axis, shrink=0.82, pad=0.03)
        colorbar.set_label("Raw reward")
    fig.suptitle("Move Red; Blue fixed at the purple gem", fontsize=9)

    plot_data = {
        "x": xx,
        "y": yy,
        "moving_agent": np.asarray(0),
        "fixed_agent": np.asarray(1),
        "fixed_agent_position": gems[0, 2],
        "gem_positions": gems[0],
        "true_reward": truth,
        "learned_reward": inferred,
    }
    return fig, plot_data


def make_plots(model, history, datasets, source, test_rows, config, output):
    colors = {"truth": "black", "learned": "#2878B5"}

    test = datasets["test"]
    if config["name"] == "simple_push_discrete":
        fig, plot_data = push_reward_distance_figure(source, test_rows, test)
        save(fig, output / "reward_vs_distance.png")
        np.savez_compressed(output / "test_reward_distance_bins.npz", **plot_data)
    elif config["name"] in ("simple_spread_discrete", "gems_discrete"):
        distances, names = task_distances(config, source, test_rows)
        scalar_data = {}
        for agent in range(2):
            rows = int(np.ceil(len(names) / 3))
            fig, axes = plt.subplots(rows, 3, figsize=(11, 3.1 * rows), squeeze=False, layout="constrained")
            for axis in axes.flat[len(names):]:
                axis.axis("off")
            for coordinate, axis in enumerate(axes.flat[:len(names)]):
                x, y, count = binned(distances[:, coordinate], np.column_stack((test["truth"][:, agent], test["inferred"][:, agent])))
                axis.plot(x, y[:, 0], color=colors["truth"], label="True")
                axis.plot(x, y[:, 1], color=colors["learned"], label="Learned")
                axis.set(xlabel=names[coordinate], ylabel=f"Agent {agent + 1} reward")
                axis.grid(alpha=.2)
                scalar_data[f"agent{agent + 1}_distance{coordinate + 1}"] = np.column_stack((x, y, count))
            axes.flat[0].legend()
            save(fig, output / f"reward_vs_scalar_distances_agent{agent + 1}.png")

            if config["name"] == "simple_spread_discrete":
                nearest = distances[:, 2 * agent : 2 * agent + 2].min(axis=1)
                for composite in (False, True):
                    values = (np.column_stack((test["truth"].sum(axis=1), test["inferred"].sum(axis=1))) if composite
                              else np.column_stack((test["truth"][:, agent], test["inferred"][:, agent])))
                    x, y, count = binned(nearest, values)
                    fig, axis = plt.subplots(figsize=(5, 3.7), layout="constrained")
                    axis.plot(x, y[:, 0], color=colors["truth"], label="True")
                    axis.plot(x, y[:, 1], color=colors["learned"], label="Learned")
                    axis.set(xlabel=f"Agent {agent + 1} distance to closest landmark", ylabel="Summed reward" if composite else f"Agent {agent + 1} reward")
                    axis.grid(alpha=.2); axis.legend()
                    stem = (f"reward_vs_agent_to_closest_landmark_composite_distance_agent{agent + 1}" if composite
                            else f"reward_vs_agent_to_closest_landmark_agent{agent + 1}")
                    save(fig, output / f"{stem}.png")
                    np.savez_compressed(output / f"{stem}.npz", distance=nearest, bin_distance=x, bin_reward=y, bin_counts=count)
        np.savez_compressed(output / "scalar_distance_plot_data.npz", **scalar_data)

    row = int(test_rows[0])
    xx, yy, slices = spatial_values(model, config, source, row)
    if config["name"] == "gems_discrete":
        fig, spatial = gems_counterfactual_spatial_figure(xx, yy, slices)
    else:
        spatial = {}
        columns = 3 if config["name"] == "simple_spread_discrete" else 4
        fig, axes = plt.subplots(2, columns, figsize=(3.6 * columns, 7), layout="constrained")
        for moving, positions, landmarks, truth, inferred in slices:
            spatial[f"move_agent{moving + 1}_truth"] = truth
            spatial[f"move_agent{moving + 1}_inferred"] = inferred
            panels = ([truth[:, 0], inferred[:, 0], inferred[:, 1]] if columns == 3 else
                      [truth[:, 0], inferred[:, 0], truth[:, 1], inferred[:, 1]])
            titles = (["True reward", "Agent 1 learned reward", "Agent 2 learned reward"] if columns == 3 else
                      ["Agent 1 true reward", "Agent 1 learned reward", "Agent 2 true reward", "Agent 2 learned reward"])
            for column, values in enumerate(panels):
                image = axes[moving, column].imshow(
                    values.reshape(xx.shape), origin="lower",
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap="coolwarm"
                )
                axes[moving, column].scatter(landmarks[0, :, 0], landmarks[0, :, 1], marker="*", s=45, color="white", edgecolor="black")
                fixed = 1 - moving
                axes[moving, column].scatter(positions[0, fixed, 0], positions[0, fixed, 1], s=22, color="black")
                axes[moving, column].set(title=titles[column], xlabel="x", ylabel="y")
                fig.colorbar(image, ax=axes[moving, column], shrink=.72)
        spatial.update({"x": xx, "y": yy})
    save(fig, output / "reward_spatial_slices.png")
    np.savez_compressed(output / "spatial_slice_values.npz", **spatial)

    steps = [entry["step"] for entry in history]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for axis, key, title in zip(
        axes,
        ("behavioral_loss", "validation_behavioral_loss", "inverse_td_loss"),
        ("Training NCE", "Validation NCE", "Inverse TD"),
    ):
        axis.plot(steps, [entry[key] for entry in history])
        axis.set(title=title, xlabel="Update", ylabel="Loss"); axis.grid(alpha=.2)
    save(fig, output.parent / "training_history.png")


def evaluate_run(model, data, split, config, history, run):
    run = Path(run)
    model.eval()
    with np.load(config["dataset"]["path"], allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}

    diagnostic_dir = run / "reward_diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    datasets = {}
    for name, rows in (("training", split.training), ("validation", split.validation), ("test", split.test)):
        datasets[name] = dataset_record(model, data, source, rows, config["gamma"])
        np.savez_compressed(diagnostic_dir / f"{name}_values.npz", **datasets[name])

    rollout_datasets, _ = evaluate_policy(model, config, source, split, run)
    for name, values in rollout_datasets.items():
        values["true_rtg"] = discounted_returns(values["truth"], values["episode_id"], values["step_id"], config["gamma"])
        values["learned_rtg"] = discounted_returns(values["inferred"], values["episode_id"], values["step_id"], config["gamma"])
        datasets[name] = values
        np.savez_compressed(run / f"{name}_rollouts.npz", **values)

    make_plots(model, history, datasets, source, split.test, config, diagnostic_dir)
    test = datasets["test"]
    metrics = {
        "test_immediate_reward": {
            f"agent{agent + 1}": association(
                test["truth"][:, agent], test["inferred"][:, agent]
            )
            for agent in range(2)
        }
    }
    write_json(diagnostic_dir / "metrics.json", metrics)
    write_json(
        run / "evaluation.json",
        {"status": "completed", "metrics": "reward_diagnostics/metrics.json"},
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    config.setdefault("evaluation", {"episodes": 500, "candidates": 256})
    history = json.loads((run / "history.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location=config["device"], weights_only=True)
    from ..train import build_model
    model = build_model(config).to(config["device"])
    model.load_state_dict(checkpoint["model"])
    data = load_trajectories(
        config["dataset"],
        config["action"]["kind"],
        config["action"]["size"],
        config["device"],
    )
    split = split_episodes(data.episode_id, **config["split"], seed=config["seed"])
    metrics = evaluate_run(model, data, split, config, history, run)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
