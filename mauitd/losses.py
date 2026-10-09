"""Behavioral and inverse-TD objectives."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .model import MAUITD


def behavior_loss(
    model: MAUITD,
    states: torch.Tensor,
    actions: torch.Tensor,
    negative_actions: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact discrete likelihood or sampled continuous contrastive likelihood."""

    losses = []
    if model.action_encoder.kind == "discrete":
        for agent in range(model.num_agents):
            logits = model.all_discrete_q(states, agent) / temperature
            losses.append(F.cross_entropy(logits, actions[:, agent].long()))
    else:
        for agent in range(model.num_agents):
            positive = model.q(states, actions[:, agent], agent)
            alternatives = negative_actions[:, agent]
            count = alternatives.shape[1]
            negative = model.q(
                states.repeat_interleave(count, dim=0),
                alternatives.flatten(0, 1),
                agent,
            ).reshape(len(states), count)
            logits = torch.cat((positive[:, None], negative), dim=1) / temperature
            labels = torch.zeros(len(states), dtype=torch.long, device=states.device)
            losses.append(F.cross_entropy(logits, labels))
    return torch.stack(losses).mean()


def successor_loss(
    model: MAUITD,
    target: MAUITD,
    states: torch.Tensor,
    actions: torch.Tensor,
    arrival_states: torch.Tensor,
    roots: torch.Tensor,
    step_rows: torch.Tensor,
    step_weights: torch.Tensor,
    bootstrap_rows: torch.Tensor,
    bootstrap_discount: torch.Tensor,
) -> torch.Tensor:
    """Per-agent n-step UITD losses using recorded continuation actions."""

    batch_size, _ = step_rows.shape
    row_array = step_rows.detach().cpu().numpy()
    active = step_weights.detach().cpu().numpy() > 0
    unique, inverse = np.unique(row_array[active], return_inverse=True)
    features = model.phi(
        arrival_states[torch.as_tensor(unique, device=step_rows.device)]
    )
    owner = torch.as_tensor(np.nonzero(active)[0], device=step_rows.device)
    values = features[torch.as_tensor(inverse, device=step_rows.device)]
    values = values * step_weights[torch.as_tensor(active, device=step_rows.device), None]
    feature_return = features.new_zeros((batch_size, model.feature_dim)).index_add(
        0, owner, values
    )

    losses = []
    for agent in range(model.num_agents):
        current = model.bcsf(states[roots], actions[roots, agent], agent)
        with torch.no_grad():
            continuation = target.bcsf(
                states[bootstrap_rows], actions[bootstrap_rows, agent], agent
            )
        expected = feature_return + bootstrap_discount * continuation
        losses.append(F.mse_loss(current, expected))
    return torch.stack(losses)
