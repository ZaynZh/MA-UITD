"""Neural networks used by MA-UITD.

The algorithm is deliberately agnostic to the environment. A task supplies the
reward-feature network, the BCSF heads, and (optionally) a shared state encoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ActionEncoder(nn.Module):
    """One-hot discrete actions or scaled continuous actions."""

    def __init__(self, kind: str, size: int, scale: float | Sequence[float] = 1.0) -> None:
        super().__init__()
        self.kind = kind
        self.size = int(size)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).expand(self.size).clone()
        self.register_buffer("scale", scale_tensor)

    @property
    def output_dim(self) -> int:
        return self.size

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if self.kind == "discrete":
            return F.one_hot(actions.long(), self.size).float()
        return actions.float() / self.scale


class MAUITD(nn.Module):
    """Shared reward features with agent-specific BCSFs and reward weights."""

    def __init__(
        self,
        reward_input_dim: int,
        state_input_dim: int,
        action_encoder: ActionEncoder,
        feature_dim: int,
        num_agents: int,
        phi_hidden_dims: Sequence[int],
        psi_hidden_dims: Sequence[int],
        activation: str = "silu",
        dropout: float = 0.0,
        negative_slope: float = 0.01,
        state_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.feature_dim = int(feature_dim)

        activation = activation.lower()

        def activation_layer() -> nn.Module:
            if activation == "relu":
                return nn.ReLU()
            if activation == "silu":
                return nn.SiLU()
            if activation == "elu":
                return nn.ELU()
            if activation == "leaky_relu":
                return nn.LeakyReLU(negative_slope)
            raise ValueError(f"Unknown activation: {activation}")

        phi_hidden_1, phi_hidden_2 = phi_hidden_dims
        self.reward_features = nn.Sequential(
            nn.Linear(reward_input_dim, phi_hidden_1),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden_1, phi_hidden_2),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden_2, feature_dim),
        )

        # Preserve the paper implementation's initialization order: Phi, w, Psi.
        self.reward_weights = nn.Parameter(
            torch.randn(num_agents, self.feature_dim) / math.sqrt(self.feature_dim)
        )

        psi_hidden_1, psi_hidden_2 = psi_hidden_dims

        def successor_network() -> nn.Sequential:
            if action_encoder.kind == "discrete" and action_encoder.output_dim == 5:
                first = nn.Linear(state_input_dim + 2, psi_hidden_1)
                second = nn.Linear(psi_hidden_1, psi_hidden_2)
                output = nn.Linear(psi_hidden_2, feature_dim)
                first = nn.Linear(state_input_dim + action_encoder.output_dim, psi_hidden_1)
            else:
                first = nn.Linear(state_input_dim + action_encoder.output_dim, psi_hidden_1)
                second = nn.Linear(psi_hidden_1, psi_hidden_2)
                output = nn.Linear(psi_hidden_2, feature_dim)
            return nn.Sequential(
                first,
                activation_layer(),
                nn.Dropout(dropout),
                second,
                activation_layer(),
                nn.Dropout(dropout),
                output,
            )

        self.successors = nn.ModuleList(
            successor_network()
            for _ in range(num_agents)
        )

    @property
    def num_agents(self) -> int:
        return len(self.successors)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state) if self.state_encoder is not None else state

    def phi(self, arrival_state: torch.Tensor) -> torch.Tensor:
        return self.reward_features(self._encode_state(arrival_state))

    def reward(self, arrival_state: torch.Tensor) -> torch.Tensor:
        return self.phi(arrival_state) @ self.reward_weights.T

    def bcsf(self, state: torch.Tensor, action: torch.Tensor, agent: int) -> torch.Tensor:
        encoded_state = self._encode_state(state)
        encoded_action = self.action_encoder(action)
        return self.successors[agent](torch.cat((encoded_state, encoded_action), dim=-1))

    def q(self, state: torch.Tensor, action: torch.Tensor, agent: int) -> torch.Tensor:
        return self.bcsf(state, action, agent) @ self.reward_weights[agent]

    def all_discrete_q(self, state: torch.Tensor, agent: int) -> torch.Tensor:
        count = self.action_encoder.size
        actions = torch.arange(count, device=state.device).repeat(len(state))
        repeated_state = state.repeat_interleave(count, dim=0)
        return self.q(repeated_state, actions, agent).reshape(len(state), count)

    def behavior_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.successors.parameters())
        if self.state_encoder is not None:
            parameters += list(self.state_encoder.parameters())
        return parameters

    def reward_feature_parameters(self) -> list[nn.Parameter]:
        return list(self.reward_features.parameters())
