"""Alternating behavioral and inverse-TD training."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import EpisodeSplit, NStepWindows, TrajectoryData
from .losses import behavior_loss, successor_loss
from .model import MAUITD
from .sampling import ExpertActionSampler


@dataclass
class TrainingConfig:
    steps: int = 100_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    bc_learning_rate: float | None = None
    weight_decay: float = 1e-5
    weight_l1: float = 0.05
    tau: float = 1e-4
    negatives: int = 16
    temperature: float = 1.0
    evaluation_frequency: int = 2_000
    gradient_clip: float = 1.0


class Trainer:
    def __init__(
        self,
        model: MAUITD,
        data: TrajectoryData,
        split: EpisodeSplit,
        windows: NStepWindows,
        config: TrainingConfig,
        seed: int,
    ) -> None:
        self.model = model
        self.target = copy.deepcopy(model).eval().requires_grad_(False)
        self.data = data
        self.split = split
        self.windows = windows
        self.config = config
        self.bc_learning_rate = (
            config.learning_rate
            if config.bc_learning_rate is None
            else config.bc_learning_rate
        )

        behavior_parameters = model.behavior_parameters()
        feature_parameters = model.reward_feature_parameters()
        self.behavior_optimizer = torch.optim.AdamW(
            behavior_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.feature_optimizer = torch.optim.AdamW(
            feature_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.weight_optimizer = torch.optim.Adam(
            [model.reward_weights], lr=self.bc_learning_rate
        )
        self.behavior_parameters = behavior_parameters
        self.feature_parameters = feature_parameters

        device = data.states.device
        self.bc_generator = np.random.default_rng(2)
        self.td_generator = np.random.default_rng(1)
        self.negative_generator = torch.Generator(device=device).manual_seed(3)
        self.training_rows = np.asarray(split.training)
        training_index = torch.as_tensor(self.training_rows, device=device)
        self.negative_sampler = ExpertActionSampler(data.actions[training_index])
        self.td_rows = self.training_rows[windows.eligible[self.training_rows]]

    def _sample(self, rows: np.ndarray, generator: np.random.Generator) -> torch.Tensor:
        batch = generator.choice(
            rows,
            size=self.config.batch_size,
            replace=True,
        )
        return torch.as_tensor(batch, device=self.data.states.device)

    def behavioral_update(self) -> float:
        rows = self._sample(self.training_rows, self.bc_generator)
        negatives = None
        if self.model.action_encoder.kind == "continuous":
            negatives = self.negative_sampler.sample(
                len(rows), self.config.negatives, self.negative_generator
            )

        for group in self.behavior_optimizer.param_groups:
            group["lr"] = self.bc_learning_rate
        self.model.zero_grad(set_to_none=True)
        loss = behavior_loss(
            self.model,
            self.data.states[rows],
            self.data.actions[rows],
            negatives,
            self.config.temperature,
        )
        objective = loss + self.config.weight_l1 * self.model.reward_weights.abs().mean()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            self.behavior_parameters,
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.behavior_optimizer.step()
        self.weight_optimizer.step()
        for group in self.behavior_optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        return float(loss.detach())

    def inverse_td_update(self) -> float:
        rows = self._sample(self.td_rows, self.td_generator)
        self.model.zero_grad(set_to_none=True)
        loss = successor_loss(
            self.model,
            self.target,
            self.data.states,
            self.data.actions,
            self.data.arrival_states,
            rows,
            self.windows.step_rows[rows],
            self.windows.step_weights[rows],
            self.windows.bootstrap_rows[rows],
            self.windows.bootstrap_discount[rows],
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.behavior_parameters,
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        torch.nn.utils.clip_grad_norm_(
            self.feature_parameters,
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.behavior_optimizer.step()
        self.feature_optimizer.step()
        self._update_target()
        return float(loss.detach())

    @torch.no_grad()
    def _update_target(self) -> None:
        source = dict(self.model.named_parameters())
        for name, parameter in self.target.named_parameters():
            if name.startswith("state_encoder") or name.startswith("successors"):
                parameter.lerp_(source[name], self.config.tau)

    @torch.no_grad()
    def validation_loss(self) -> float:
        self.model.eval()
        rows = torch.as_tensor(self.split.validation, device=self.data.states.device)
        generator = torch.Generator(device=rows.device).manual_seed(10_001)
        total = 0.0
        for batch in rows.split(4096):
            negatives = None
            if self.model.action_encoder.kind == "continuous":
                negatives = self.negative_sampler.sample(
                    len(batch), self.config.negatives, generator
                )
            loss = behavior_loss(
                self.model,
                self.data.states[batch],
                self.data.actions[batch],
                negatives,
                self.config.temperature,
            )
            total += len(batch) * float(loss)
        self.model.train()
        return total / len(rows)

    def fit(self, output_dir: str | Path) -> list[dict]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        best_loss = float("inf")
        best_state = None

        for step in range(1, self.config.steps + 1):
            behavioral = self.behavioral_update()
            inverse_td = self.inverse_td_update()
            if step % self.config.evaluation_frequency == 0 or step == self.config.steps:
                validation = self.validation_loss()
                record = {
                    "step": step,
                    "behavioral_loss": behavioral,
                    "inverse_td_loss": inverse_td,
                    "validation_behavioral_loss": validation,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                if validation < best_loss:
                    best_loss = validation
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }

        assert best_state is not None
        self.model.load_state_dict(best_state)
        torch.save({"model": best_state, "best_validation_loss": best_loss}, output / "model.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        return history
