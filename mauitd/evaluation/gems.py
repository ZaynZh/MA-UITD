"""VMAS implementation of the two-agent Gems task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.environment.environment import Environment
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


AGENT_NAMES = ("agent_red", "agent_blue")
GEM_NAMES = ("gem_red", "gem_blue", "gem_purple")


@dataclass(frozen=True)
class GemsConfig:
    arena_size: float = 2.0
    gem_radius: float = 0.08
    action_limit: float = 0.05
    horizon: int = 80
    private_reward: float = 1.0
    purple_reward: float = 6.0
    diagonal_actions: bool = True
    gem_positions: tuple[tuple[float, float], ...] = (
        (0.8, -0.8),
        (-0.8, -0.8),
        (0.0, 0.8),
    )


class DirectDisplacement(Dynamics):
    """The scenario applies actions as displacements without VMAS physics."""

    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size

    @property
    def needed_action_size(self) -> int:
        return self.action_size

    def process_action(self) -> None:
        self.agent.state.force.zero_()


class GemsScenario(BaseScenario):
    """Red and Blue collect private gems or cooperate at the purple gem."""

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.config = GemsConfig(
            arena_size=kwargs.pop("arena_size", 2.0),
            gem_radius=kwargs.pop("gem_radius", 0.08),
            action_limit=kwargs.pop("action_limit", 0.05),
            horizon=kwargs.pop("horizon", 80),
            private_reward=kwargs.pop("private_reward", 1.0),
            purple_reward=kwargs.pop("purple_reward", 6.0),
            diagonal_actions=kwargs.pop("diagonal_actions", True),
            gem_positions=tuple(kwargs.pop("gem_positions", GemsConfig().gem_positions)),
        )
        self.continuous_actions = kwargs.pop("scenario_continuous_actions")
        ScenarioUtils.check_kwargs_consumed(kwargs)

        half_extent = self.config.arena_size / 2
        world = World(
            batch_dim,
            device,
            dt=1.0,
            substeps=1,
            drag=0.0,
            linear_friction=0.0,
            x_semidim=half_extent,
            y_semidim=half_extent,
        )

        for index, (name, color) in enumerate(
            zip(AGENT_NAMES, (Color.RED, Color.BLUE))
        ):
            action_count = 9 if self.config.diagonal_actions else 5
            action_size = 2 if self.continuous_actions else 1
            agent = Agent(
                name=name,
                collide=False,
                rotatable=False,
                shape=Sphere(
                    radius=min(
                        0.025 * self.config.arena_size,
                        self.config.gem_radius / 3,
                    )
                ),
                color=color,
                render_action=False,
                u_range=self.config.action_limit if self.continuous_actions else 1.0,
                action_size=action_size,
                discrete_action_nvec=None if self.continuous_actions else [action_count],
                dynamics=DirectDisplacement(action_size),
            )
            agent.role_index = index
            world.add_agent(agent)

        for name, color in zip(GEM_NAMES, (Color.RED, Color.BLUE, Color.PURPLE)):
            world.add_landmark(
                Landmark(
                    name=name,
                    collide=False,
                    movable=False,
                    shape=Sphere(radius=self.config.gem_radius),
                    color=color,
                )
            )

        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self.private_collected = torch.zeros(batch_dim, 2, dtype=torch.bool, device=device)
        self.purple_collected = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.rewards = torch.zeros(batch_dim, 2, device=device)
        self.last_displacement = torch.zeros(batch_dim, 2, 2, device=device)
        return world

    def reset_world_at(self, env_index: int | None = None) -> None:
        indices = (
            torch.arange(self.world.batch_dim, device=self.world.device)
            if env_index is None
            else torch.tensor([env_index], device=self.world.device)
        )
        batch_size = len(indices)
        fixed_gems = torch.tensor(
            self.config.gem_positions, dtype=torch.float32, device=self.world.device
        )
        for gem, position in zip(self.world.landmarks, fixed_gems):
            values = position.expand(batch_size, -1)
            gem.set_pos(values if env_index is None else values[0], env_index)

        gem_positions = fixed_gems.expand(batch_size, -1, -1)
        previous_agents = []
        for agent in self.world.agents:
            positions = self._sample_agent_positions(gem_positions, previous_agents)
            agent.set_pos(positions if env_index is None else positions[0], env_index)
            velocity = torch.zeros_like(positions)
            agent.set_vel(velocity if env_index is None else velocity[0], env_index)
            previous_agents.append(positions)

        self.step_count[indices] = 0
        self.private_collected[indices] = False
        self.purple_collected[indices] = False
        self.rewards[indices] = 0
        self.last_displacement[indices] = 0

    def process_action(self, agent: Agent) -> None:
        if self.continuous_actions:
            displacement = agent.action.u[:, :2].clamp(
                -self.config.action_limit, self.config.action_limit
            )
        else:
            action_id = self._decode_discrete_action(agent.action.u[:, 0])
            directions = torch.tensor(
                [
                    [0, 1], [1, 0], [-1, 0], [0, -1], [0, 0],
                    [1, 1], [-1, 1], [1, -1], [-1, -1],
                ],
                dtype=torch.float32,
                device=self.world.device,
            )
            displacement = directions[action_id] * self.config.action_limit

        half_extent = self.config.arena_size / 2
        agent.state.pos = (agent.state.pos + displacement).clamp(-half_extent, half_extent)
        agent.state.vel.zero_()
        self.last_displacement[:, int(agent.role_index)] = displacement

    def post_step(self) -> None:
        self.step_count += 1
        agents = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        gems = torch.stack([gem.state.pos for gem in self.world.landmarks], dim=1)
        distances = torch.linalg.vector_norm(agents[:, :, None] - gems[:, None], dim=-1)

        self.private_collected[:, 0] = distances[:, 0, 0] <= self.config.gem_radius
        self.private_collected[:, 1] = distances[:, 1, 1] <= self.config.gem_radius
        self.purple_collected[:] = (distances[:, :, 2] <= self.config.gem_radius).all(dim=1)
        self.rewards[:] = self.private_collected.float() * self.config.private_reward
        self.rewards += self.purple_collected[:, None].float() * self.config.purple_reward

    def reward(self, agent: Agent) -> torch.Tensor:
        return self.rewards[:, int(agent.role_index)]

    def observation(self, agent: Agent) -> torch.Tensor:
        del agent
        positions = [entity.state.pos for entity in self.world.agents]
        positions += [gem.state.pos for gem in self.world.landmarks]
        return torch.cat(positions, dim=-1) / self.config.arena_size + 0.5

    def done(self) -> torch.Tensor:
        return self.private_collected.any(dim=1) | self.purple_collected

    def info(self, agent: Agent) -> dict[str, torch.Tensor]:
        index = int(agent.role_index)
        terminated = self.done()
        return {
            "own_collected": self.private_collected[:, index],
            "purple_collected": self.purple_collected,
            "terminated": terminated,
            "truncated": (self.step_count >= self.config.horizon) & ~terminated,
            "last_displacement": self.last_displacement[:, index],
        }

    def _sample_agent_positions(
        self, gem_positions: torch.Tensor, previous_agents: list[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = len(gem_positions)
        half_extent = self.config.arena_size / 2
        positions = torch.empty(batch_size, 2, device=self.world.device).uniform_(
            -half_extent, half_extent
        )
        invalid = self._invalid_agent_positions(positions, gem_positions, previous_agents)
        while invalid.any():
            positions[invalid] = torch.empty(
                int(invalid.sum()), 2, device=self.world.device
            ).uniform_(-half_extent, half_extent)
            invalid = self._invalid_agent_positions(positions, gem_positions, previous_agents)
        return positions

    def _invalid_agent_positions(
        self,
        positions: torch.Tensor,
        gem_positions: torch.Tensor,
        previous_agents: list[torch.Tensor],
    ) -> torch.Tensor:
        invalid = (
            torch.linalg.vector_norm(positions[:, None] - gem_positions, dim=-1)
            <= self.config.gem_radius
        ).any(dim=1)
        if previous_agents:
            invalid |= torch.linalg.vector_norm(positions - previous_agents[0], dim=-1) == 0
        return invalid

    def _decode_discrete_action(self, encoded: torch.Tensor) -> torch.Tensor:
        count = 9 if self.config.diagonal_actions else 5
        ids = torch.arange(count, dtype=encoded.dtype, device=encoded.device)
        shifted = torch.where(
            ids == 0, count // 2, torch.where(ids <= count // 2, ids - 1, ids)
        )
        encoded_values = shifted * (2.0 / (count - 1)) - 1.0
        return (encoded[:, None] - encoded_values[None]).abs().argmin(dim=1)


class GemsEnvironment(Environment):
    """Use success as termination and the horizon only as truncation."""

    def _done(self):
        terminated = self.scenario.done().clone()
        truncated = (self.steps >= self.max_steps) & ~terminated
        if self.terminated_truncated:
            return terminated, truncated
        return terminated | truncated


def make_env(
    config: GemsConfig | None = None,
    *,
    num_envs: int = 1,
    continuous_actions: bool = True,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    terminated_truncated: bool = False,
) -> GemsEnvironment:
    """Construct the vectorized VMAS Gems environment."""

    config = config or GemsConfig()
    scenario_kwargs = asdict(config)
    scenario_kwargs["scenario_continuous_actions"] = continuous_actions
    return GemsEnvironment(
        GemsScenario(),
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=config.horizon,
        seed=seed,
        clamp_actions=True,
        terminated_truncated=terminated_truncated,
        **scenario_kwargs,
    )
