"""Train MA-UITD from a single task configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import build_n_step_windows, load_trajectories, split_episodes
from .model import ActionEncoder, MAUITD
from .trainer import Trainer, TrainingConfig


def build_model(config: dict) -> MAUITD:
    model = config["model"]
    action = config["action"]
    activation = model.get("activation", "silu")
    negative_slope = model.get("negative_slope", 0.01)
    dropout = model.get("dropout", 0.0)

    action_encoder = ActionEncoder(action["kind"], action["size"], action.get("scale", 1.0))
    encoder_dims = model.get("shared_encoder_dims", [])
    state_encoder = None
    encoded_state_dim = model["state_input_dim"]
    if encoder_dims:
        assert model["reward_input_dim"] == model["state_input_dim"]
        encoder_name = model.get("encoder_activation", activation)
        if encoder_name == "relu":
            encoder_activation = nn.ReLU
        elif encoder_name == "silu":
            encoder_activation = nn.SiLU
        elif encoder_name == "elu":
            encoder_activation = nn.ELU
        elif encoder_name == "leaky_relu":
            encoder_activation = lambda: nn.LeakyReLU(negative_slope)
        else:
            raise ValueError(f"Unknown encoder activation: {encoder_name}")
        encoder_hidden, encoded_state_dim = encoder_dims
        state_encoder = nn.Sequential(
            nn.Linear(model["state_input_dim"], encoder_hidden),
            encoder_activation(),
            nn.Linear(encoder_hidden, encoded_state_dim),
            encoder_activation(),
        )

    return MAUITD(
        reward_input_dim=(encoded_state_dim if state_encoder is not None else model["reward_input_dim"]),
        state_input_dim=encoded_state_dim,
        action_encoder=action_encoder,
        feature_dim=model["feature_dim"],
        num_agents=model["num_agents"],
        phi_hidden_dims=model["phi_hidden_dims"],
        psi_hidden_dims=model["psi_hidden_dims"],
        activation=activation,
        dropout=dropout,
        negative_slope=negative_slope,
        state_encoder=state_encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.get("device", "cuda")
    model = build_model(config).to(device)
    data = load_trajectories(
        config["dataset"],
        config["action"]["kind"],
        config["action"]["size"],
        device,
    )
    split_config = config["split"]
    split = split_episodes(
        data.episode_id,
        split_config["training"],
        split_config["validation"],
        split_config["test"],
        seed,
    )
    windows = build_n_step_windows(data, config["n_step"], config["gamma"])
    training = TrainingConfig(**config["training"])
    trainer = Trainer(model, data, split, windows, training, seed)

    output = args.output_dir or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = trainer.fit(output)

    if config.get("evaluation"):
        from .evaluation import evaluate_run
        metrics = evaluate_run(model, data, split, config, history, output)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
