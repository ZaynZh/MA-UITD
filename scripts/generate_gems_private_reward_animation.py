#!/usr/bin/env python3
"""Render a held-out Gems private-goal trajectory and raw recovered rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # Optional publication-QA helper, supplied outside the package.
    require_matplotlib_panel_alignment = None


RED = "#E7603B"
BLUE = "#2D7FB8"
PURPLE = "#7A5195"
BLACK = "#202124"
GRID = "#DDE3E8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-values", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=2521)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--qa-pdf", type=Path)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--alignment-overlay", type=Path)
    parser.add_argument(
        "--selection-note",
        default="held-out episode selected by explicit episode ID",
    )
    return parser.parse_args()


def load_episode(dataset_path: Path, values_path: Path, episode_id: int) -> dict:
    with np.load(dataset_path, allow_pickle=False) as dataset, np.load(
        values_path, allow_pickle=False
    ) as values:
        source_rows = values["source_rows"]
        selected = dataset["episode_id"][source_rows] == episode_id
        rows = source_rows[selected]
        if not len(rows):
            raise ValueError(f"Episode {episode_id} is absent from the held-out values")

        episode_rows = np.flatnonzero(dataset["episode_id"] == episode_id)
        if not np.array_equal(rows, episode_rows):
            raise ValueError("Held-out values do not contain the complete requested episode")

        truth = dataset["rewards"][rows].astype(np.float64)
        inferred = values["inferred"][selected].astype(np.float64)
        if truth.shape != inferred.shape or truth.shape[1] != 2:
            raise ValueError("Expected aligned two-agent true and inferred rewards")
        if not np.isfinite(inferred).all():
            raise ValueError("Inferred rewards contain non-finite values")

        terminal = truth[-1]
        private = (terminal > 0) & (terminal <= 1.0 + 1e-6)
        if private.sum() != 1 or np.any(terminal > 1.0 + 1e-6):
            raise ValueError("Requested episode does not end at exactly one private gem")

        states = dataset["states"][rows]
        next_states = dataset["next_states"][rows]
        positions = np.concatenate((states[:1, :4], next_states[:, :4]), axis=0)
        gems = states[0, 4:10].reshape(3, 2).astype(np.float64)
        collector = int(np.flatnonzero(private)[0])
        terminal_positions = positions[-1].reshape(2, 2)
        private_distances = np.linalg.norm(terminal_positions - gems[:2], axis=1)
        close_to_private = private_distances <= 0.04 + 1e-6
        if close_to_private.sum() != 1 or not close_to_private[collector]:
            raise ValueError(
                "Requested episode must end with only the collecting agent close "
                "to its private gem"
            )

    return {
        "rows": rows,
        "positions": positions.astype(np.float64),
        "gems": gems,
        "truth": truth,
        "inferred": inferred,
        "collector": collector,
        "terminal_private_distances": private_distances,
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def build_figure(data: dict, episode_id: int):
    truth = data["truth"]
    inferred = data["inferred"]
    transition_count = len(truth)

    fig = plt.figure(figsize=(9.0, 3.8), dpi=100, facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.34, 0.66),
        left=0.035,
        right=0.985,
        bottom=0.13,
        top=0.88,
        wspace=0.22,
    )
    env_ax = fig.add_subplot(outer[0])
    reward_grid = outer[1].subgridspec(4, 1, hspace=0.48)
    reward_axes = [fig.add_subplot(reward_grid[i]) for i in range(4)]

    fig.text(
        0.235,
        0.945,
        "Private-goal expert demonstration",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=BLACK,
    )
    fig.text(
        0.705,
        0.945,
        "Raw arrival-state rewards",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=BLACK,
    )

    env_ax.set_xlim(0.0, 1.0)
    env_ax.set_ylim(0.0, 1.0)
    env_ax.set_aspect("equal", adjustable="box")
    env_ax.set_xticks([])
    env_ax.set_yticks([])
    for spine in env_ax.spines.values():
        spine.set_color("#CBD4DB")
        spine.set_linewidth(0.8)

    gem_colors = (RED, BLUE, PURPLE)
    gem_labels = ("Red private gem", "Blue private gem", "Purple gem")
    for xy, color, label in zip(data["gems"], gem_colors, gem_labels):
        env_ax.add_patch(
            plt.Circle(
                xy,
                0.038,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=2,
                label=label,
            )
        )

    red_trail, = env_ax.plot([], [], color=RED, alpha=0.28, linewidth=1.35, zorder=3)
    blue_trail, = env_ax.plot([], [], color=BLUE, alpha=0.28, linewidth=1.35, zorder=3)
    red_dot, = env_ax.plot([], [], "o", color=RED, markersize=4.8, zorder=5)
    blue_dot, = env_ax.plot([], [], "o", color=BLUE, markersize=4.8, zorder=5)
    transition_text = env_ax.text(
        0.025,
        0.975,
        "",
        transform=env_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#59636D",
    )
    outcome_text = env_ax.text(
        0.5,
        -0.095,
        "",
        transform=env_ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color=RED if data["collector"] == 0 else BLUE,
    )

    labels = (
        "Red agent ground-truth arrival-state reward",
        "Red agent inferred arrival-state reward",
        "Blue agent ground-truth arrival-state reward",
        "Blue agent inferred arrival-state reward",
    )
    colors = (BLACK, RED, BLACK, BLUE)
    series = (truth[:, 0], inferred[:, 0], truth[:, 1], inferred[:, 1])
    y_low = min(-0.5, float(np.min(inferred)) - 0.12)
    y_high = max(3.0, float(np.max(inferred)) + 0.12)
    x = np.arange(1, transition_count + 1)
    reward_lines = []
    reward_points = []
    cursors = []

    for index, (axis, label, color) in enumerate(zip(reward_axes, labels, colors)):
        axis.set_xlim(1, transition_count)
        axis.set_ylim(y_low, y_high)
        axis.axhline(0, color=GRID, linewidth=0.7, zorder=0)
        axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#CBD4DB")
        axis.spines["bottom"].set_color("#CBD4DB")
        axis.set_yticks([0, 1, 2, 3])
        axis.tick_params(length=2.0, width=0.55, colors="#59636D", pad=1.5)
        if index < 3:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Transition", labelpad=2)
        axis.text(
            0.0,
            1.08,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color=color,
            fontsize=6.4,
            fontweight="bold",
        )
        line, = axis.plot([], [], color=color, linewidth=1.45, zorder=3)
        point, = axis.plot([], [], "o", color=color, markersize=3.0, zorder=4)
        cursor = axis.axvline(1, color="#AAB4BD", linewidth=0.7, linestyle=(0, (2, 2)))
        reward_lines.append(line)
        reward_points.append(point)
        cursors.append(cursor)

    fig.text(
        0.235,
        0.055,
        f"Held-out episode {episode_id}",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#707A83",
    )

    def update(frame: int) -> None:
        position_index = frame + 1
        red_xy = data["positions"][: position_index + 1, 0:2]
        blue_xy = data["positions"][: position_index + 1, 2:4]
        red_trail.set_data(red_xy[:, 0], red_xy[:, 1])
        blue_trail.set_data(blue_xy[:, 0], blue_xy[:, 1])
        red_dot.set_data([red_xy[-1, 0]], [red_xy[-1, 1]])
        blue_dot.set_data([blue_xy[-1, 0]], [blue_xy[-1, 1]])
        transition_text.set_text(f"Transition {frame + 1}/{transition_count}")

        visible_x = x[: frame + 1]
        for line, point, cursor, values in zip(
            reward_lines, reward_points, cursors, series
        ):
            visible = values[: frame + 1]
            line.set_data(visible_x, visible)
            point.set_data([visible_x[-1]], [visible[-1]])
            cursor.set_xdata([frame + 1, frame + 1])

        if frame == transition_count - 1:
            agent = "Red" if data["collector"] == 0 else "Blue"
            outcome_text.set_text(f"{agent} agent collects its private gem")
        else:
            outcome_text.set_text("")

    return fig, update, transition_count, reward_axes


def render(args: argparse.Namespace) -> None:
    configure_style()
    data = load_episode(args.dataset, args.test_values, args.episode_id)
    fig, update, frame_count, reward_axes = build_figure(data, args.episode_id)
    if args.alignment_json is not None:
        if require_matplotlib_panel_alignment is None:
            raise RuntimeError(
                "--alignment-json requires audit_panel_alignment on PYTHONPATH"
            )
        require_matplotlib_panel_alignment(
            fig,
            axes=reward_axes,
            panel_ids=["a", "b", "c", "d"],
            column_groups=[["a", "b", "c", "d"]],
            json_out=args.alignment_json,
            overlay_svg=args.alignment_overlay,
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    frames: list[Image.Image] = []
    for frame in range(frame_count):
        update(frame)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba, mode="RGBA").convert("RGB"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    durations = [165] * frame_count
    durations[-1] = 1200
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    update(frame_count - 1)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=160, facecolor="white")
    if args.qa_pdf is not None:
        args.qa_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.qa_pdf, facecolor="white")
    plt.close(fig)

    if args.source_data is not None:
        args.source_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.source_data,
            episode_id=np.asarray(args.episode_id),
            source_rows=data["rows"],
            positions=data["positions"],
            gem_positions=data["gems"],
            true_arrival_rewards=data["truth"],
            inferred_arrival_rewards=data["inferred"],
            collector=np.asarray(data["collector"]),
            terminal_private_distances=data["terminal_private_distances"],
        )

    if args.provenance is not None:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "episode_id": args.episode_id,
            "selection": args.selection_note,
            "alignment": "arrival state",
            "reward_display": "raw and uncalibrated with a shared y-axis across all four traces",
            "collector": "red" if data["collector"] == 0 else "blue",
            "transitions": frame_count,
            "terminal_true_reward": data["truth"][-1].tolist(),
            "terminal_inferred_reward": data["inferred"][-1].tolist(),
            "terminal_distance_to_own_private_gem": data[
                "terminal_private_distances"
            ].tolist(),
        }
        args.provenance.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    render(parse_args())
