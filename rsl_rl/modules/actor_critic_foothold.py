# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import MLP

from .actor_critic import ActorCritic


class FootholdLateFusionActor(nn.Module):
    """Actor with an auxiliary foothold head fused before its final hidden layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        foothold_output_dim: int = 6,
        foothold_hidden_dims: tuple[int, ...] | list[int] = (128,),
        activation: str = "elu",
        detach_foothold_prediction: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 4:
            raise ValueError(
                "FootholdLateFusionActor requires exactly four actor hidden dimensions; "
                f"received {list(hidden_dims)}."
            )

        self.input_dim = input_dim
        self.encoder = MLP(input_dim, hidden_dims[2], hidden_dims[:2], activation, last_activation=activation)
        self.foothold_head = MLP(hidden_dims[2], foothold_output_dim, foothold_hidden_dims, activation)
        self.actor_tail = MLP(
            hidden_dims[2] + foothold_output_dim,
            output_dim,
            [hidden_dims[3]],
            activation,
        )
        self.detach_foothold_prediction = detach_foothold_prediction
        self.foothold_prediction: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(observations)
        foothold_prediction = self.foothold_head(latent)
        self.foothold_prediction = foothold_prediction
        actor_foothold = foothold_prediction.detach() if self.detach_foothold_prediction else foothold_prediction
        return self.actor_tail(torch.cat([latent, actor_foothold], dim=-1))


class ActorCriticFoothold(ActorCritic):
    """Four-layer actor-critic with late fusion of predicted next footholds."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        foothold_output_dim: int = 6,
        foothold_hidden_dims: tuple[int, ...] | list[int] = (128,),
        detach_foothold_prediction: bool = True,
        activation: str = "elu",
        state_dependent_std: bool = False,
        **kwargs,
    ) -> None:
        if state_dependent_std:
            raise ValueError("ActorCriticFoothold does not support state-dependent action standard deviation.")
        if len(critic_hidden_dims) != 4:
            raise ValueError(
                "ActorCriticFoothold requires exactly four critic hidden dimensions; "
                f"received {list(critic_hidden_dims)}."
            )

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        self.actor = FootholdLateFusionActor(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            foothold_output_dim=foothold_output_dim,
            foothold_hidden_dims=foothold_hidden_dims,
            activation=activation,
            detach_foothold_prediction=detach_foothold_prediction,
        )
        print(f"Foothold late-fusion actor: {self.actor}")

    @property
    def foothold_prediction(self) -> torch.Tensor:
        prediction = self.actor.foothold_prediction
        if prediction is None:
            raise RuntimeError("Foothold prediction requested before an actor forward pass.")
        return prediction
