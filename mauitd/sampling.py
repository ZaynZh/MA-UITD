"""Negative-action sampling for continuous behavioral learning."""

from __future__ import annotations

import torch


class ExpertActionSampler:
    """Samples each agent's negatives from its global training-action marginal."""

    def __init__(self, training_actions: torch.Tensor) -> None:
        self.actions = training_actions

    def sample(
        self,
        batch_size: int,
        negatives: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        samples = []
        for agent in range(self.actions.shape[1]):
            indices = torch.randint(
                len(self.actions),
                (batch_size, negatives),
                device=self.actions.device,
                generator=generator,
            )
            samples.append(self.actions[:, agent][indices])
        return torch.stack(samples, dim=1)
