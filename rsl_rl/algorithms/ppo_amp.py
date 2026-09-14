from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, AMPDiscriminator
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import string_to_callable
from rsl_rl.algorithms import PPO
from rsl_rl.modules.amp import LossType


def _rasterize_rectangles(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    yaw: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    local_size: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    """Rasterize one oriented rectangle per sample into flattened local grids."""
    rows = round(local_size[0] / resolution)
    cols = round(local_size[1] / resolution)
    x = (torch.arange(rows, device=center_x.device) + 0.5) * resolution - 0.5 * local_size[0]
    y = (torch.arange(cols, device=center_x.device) + 0.5) * resolution - 0.5 * local_size[1]
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    dx = grid_x.unsqueeze(0) - center_x[:, None, None]
    dy = grid_y.unsqueeze(0) - center_y[:, None, None]
    cos_yaw = torch.cos(yaw)[:, None, None]
    sin_yaw = torch.sin(yaw)[:, None, None]
    rect_x = cos_yaw * dx + sin_yaw * dy
    rect_y = -sin_yaw * dx + cos_yaw * dy
    occupied = (rect_x.abs() <= 0.5 * length[:, None, None]) & (
        rect_y.abs() <= 0.5 * width[:, None, None]
    )
    return occupied.to(dtype=center_x.dtype).flatten(start_dim=1)


def _fill_non_overlapping_grid_regions(
    grid: torch.Tensor,
    density_range: tuple[float, float],
    rectangle_rows_range: tuple[int, int],
    rectangle_cols_range: tuple[int, int],
    forbidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill local grids with non-overlapping axis-aligned rectangular regions."""
    batch_size, rows, cols = grid.shape
    device = grid.device
    target_cells = torch.empty(batch_size, device=device).uniform_(*density_range)
    target_cells = torch.round(target_cells * rows * cols).to(torch.long)
    occupied_cells = grid.bool().sum(dim=(1, 2))
    blocked = grid.bool() if forbidden is None else grid.bool() | forbidden.bool()
    max_rectangle_cells = rectangle_rows_range[1] * rectangle_cols_range[1]
    max_attempts = int(target_cells.max().item()) + 4 * max_rectangle_cells
    batch_index = torch.arange(batch_size, device=device)[:, None]
    row_offset = torch.arange(rectangle_rows_range[1], device=device)[None, :, None]
    col_offset = torch.arange(rectangle_cols_range[1], device=device)[None, None, :]

    for _ in range(max_attempts):
        active = occupied_cells < target_cells
        if not active.any():
            break
        height = torch.randint(rectangle_rows_range[0], rectangle_rows_range[1] + 1, (batch_size,), device=device)
        width = torch.randint(rectangle_cols_range[0], rectangle_cols_range[1] + 1, (batch_size,), device=device)
        start_row = (torch.rand(batch_size, device=device) * (rows - height + 1)).to(torch.long)
        start_col = (torch.rand(batch_size, device=device) * (cols - width + 1)).to(torch.long)
        cell_mask = (row_offset < height[:, None, None]) & (col_offset < width[:, None, None])
        cell_rows = start_row[:, None, None] + row_offset
        cell_cols = start_col[:, None, None] + col_offset
        flat_indices = (cell_rows * cols + cell_cols).flatten(start_dim=1).clamp(0, rows * cols - 1)
        cell_mask = cell_mask.flatten(start_dim=1)
        conflicts = torch.gather(blocked.flatten(start_dim=1), 1, flat_indices) & cell_mask
        accepted = active & (~conflicts.any(dim=1))
        write_mask = accepted[:, None] & cell_mask
        expanded_batch = batch_index.expand_as(flat_indices)
        grid.flatten(start_dim=1)[expanded_batch[write_mask], flat_indices[write_mask]] = 1.0
        blocked.flatten(start_dim=1)[expanded_batch[write_mask], flat_indices[write_mask]] = True
        occupied_cells += accepted.to(torch.long) * height * width
    return grid


def _replace_grid_history(observations: torch.Tensor, grid: torch.Tensor, history_length: int) -> None:
    """Replace the trailing term-major occupancy-grid history in place."""
    grid_history_size = grid.shape[-1] * history_length
    if observations.shape[-1] < grid_history_size:
        raise ValueError(
            f"Grid history ({grid_history_size}) exceeds observation size ({observations.shape[-1]})."
        )
    observations[..., -grid_history_size:] = grid.repeat(1, history_length)


class PPOAMP(PPO):

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer, 
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Auxiliary next-foothold prediction parameters
        foothold_cfg: dict | None = None,
        # Counterfactual occupancy-grid augmentation parameters
        counterfactual_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            storage,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            normalize_advantage_per_mini_batch,
            device,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")
        
        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'")
        
        self.amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            device=device,
            **self.amp_cfg.get("amp_discriminator", {})
        ).to(self.device)
        
        # optimizer for policy and discriminator
        params = [
            {
                "name": "disc_trunk", 
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
            }
        ]
        # use a separate optimizer for the AMP discriminator
        self.disc_optimizer = optim.Adam(
            params,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)
        self.disc_update_interval = self.amp_cfg.get("disc_update_interval", 1)
        
        # Storage for AMP discriminator observations
        self.disc_obs_buffer: CircularBuffer = disc_obs_buffer
        self.disc_demo_obs_buffer: CircularBuffer = disc_demo_obs_buffer
        self.foothold_cfg = foothold_cfg
        self.counterfactual_cfg = counterfactual_cfg
        self._counterfactual_samples: list[dict] = []
        
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(extras["terminal_obs"], flatten_history_dim=False)
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]
        # Compute the Style Reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=self.amp_cfg["step_dt"])
        # Linearly interpolate between task reward and style reward
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)
        # Store the un-normalized disc obs and disc demo obs into buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        foothold_cfg = getattr(self, "foothold_cfg", None)
        foothold_step = self.storage.step if foothold_cfg is not None else None
        previous_foothold_state = None
        if foothold_cfg is not None:
            previous_foothold_state = self.transition.observations[foothold_cfg["supervision_group"]]

        counterfactual_cfg = self.counterfactual_cfg
        if counterfactual_cfg is not None:
            self._collect_counterfactual_samples(
                previous_observations=self.transition.observations,
                actions=self.transition.actions,
                current_observations=obs,
                dones=dones,
            )

        # Call the parent class method with the new rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

        if foothold_step is not None and previous_foothold_state is not None:
            self._backfill_foothold_targets(
                step=foothold_step,
                previous_state=previous_foothold_state,
                current_state=obs[self.foothold_cfg["supervision_group"]],
                dones=dones,
            )

    def _collect_counterfactual_samples(
        self,
        previous_observations: TensorDict,
        actions: torch.Tensor,
        current_observations: TensorDict,
        dones: torch.Tensor,
    ) -> None:
        """Create safe/collision one-step samples at newly detected touchdowns."""
        cfg = self.counterfactual_cfg
        state_group = cfg["state_group"]
        previous_state = previous_observations[state_group]
        current_state = current_observations[state_group]
        previous_contacts = previous_state[:, 12:14] > 0.5
        current_contacts = current_state[:, 12:14] > 0.5
        touchdowns = (~previous_contacts) & current_contacts & (~dones.bool()).unsqueeze(-1)
        touchdowns &= (previous_state[:, 14] > 0.5).unsqueeze(-1)
        local_size = tuple(cfg["local_size"])
        resolution = float(cfg["resolution"])
        rows = round(local_size[0] / resolution)
        cols = round(local_size[1] / resolution)
        touchdown_pairs = touchdowns.nonzero(as_tuple=False)
        if touchdown_pairs.numel() == 0:
            return

        samples_per_touchdown = int(cfg["samples_per_touchdown"])
        pair_indices = touchdown_pairs.repeat_interleave(samples_per_touchdown, dim=0)
        env_indices = pair_indices[:, 0]
        foot_indices = pair_indices[:, 1]
        count = env_indices.numel()

        fake_obs = previous_observations[env_indices].clone()
        fake_actions = actions[env_indices].clone()
        prev = previous_state[env_indices]
        cur = current_state[env_indices]

        # Touchdown pose in the robot-local frame at action sampling time.
        foot_offset = 4 + 4 * foot_indices
        gather_xy = torch.stack((foot_offset, foot_offset + 1), dim=-1)
        foot_xy_w = torch.gather(cur, 1, gather_xy)
        delta = foot_xy_w - prev[:, :2]
        root_cos, root_sin = prev[:, 2], prev[:, 3]
        foot_x = root_cos * delta[:, 0] + root_sin * delta[:, 1]
        foot_y = -root_sin * delta[:, 0] + root_cos * delta[:, 1]
        foot_cos_w = torch.gather(cur, 1, (foot_offset + 2).unsqueeze(-1)).squeeze(-1)
        foot_sin_w = torch.gather(cur, 1, (foot_offset + 3).unsqueeze(-1)).squeeze(-1)
        foot_cos = root_cos * foot_cos_w + root_sin * foot_sin_w
        foot_sin = root_cos * foot_sin_w - root_sin * foot_cos_w
        foot_yaw = torch.atan2(foot_sin, foot_cos)

        sample_number = torch.arange(count, device=self.device) % samples_per_touchdown
        safe_count = round(samples_per_touchdown * float(cfg["safe_fraction"]))
        is_safe = sample_number < safe_count

        # Foot centre is 3.5 cm ahead of the ankle; its half extents are
        # 8.5 cm forward/backward and 3 cm laterally.
        foot_center_x = foot_x + 0.035 * foot_cos
        foot_center_y = foot_y + 0.035 * foot_sin
        foot_grid = _rasterize_rectangles(
            foot_center_x,
            foot_center_y,
            foot_yaw,
            torch.full_like(foot_x, 0.17 + resolution),
            torch.full_like(foot_x, 0.06 + resolution),
            local_size,
            resolution,
        ).view(count, rows, cols).bool()

        # Every collision sample starts with one grid-aligned region containing
        # the ankle cell. Safe samples start empty and reserve the whole foot.
        grid = torch.zeros(count, rows, cols, device=self.device)
        min_length, max_length = cfg["rectangle_length_range"]
        min_width, max_width = cfg["rectangle_width_range"]
        rectangle_rows_range = (round(min_length / resolution), round(max_length / resolution))
        rectangle_cols_range = (round(min_width / resolution), round(max_width / resolution))
        collision_height = torch.randint(
            rectangle_rows_range[0], rectangle_rows_range[1] + 1, (count,), device=self.device
        )
        collision_width = torch.randint(
            rectangle_cols_range[0], rectangle_cols_range[1] + 1, (count,), device=self.device
        )
        anchor_row = torch.floor((foot_x + 0.5 * local_size[0]) / resolution).to(torch.long).clamp(0, rows - 1)
        anchor_col = torch.floor((foot_y + 0.5 * local_size[1]) / resolution).to(torch.long).clamp(0, cols - 1)
        row_inside = (torch.rand(count, device=self.device) * collision_height).to(torch.long)
        col_inside = (torch.rand(count, device=self.device) * collision_width).to(torch.long)
        collision_start_row = (anchor_row - row_inside).clamp(min=0)
        collision_start_col = (anchor_col - col_inside).clamp(min=0)
        collision_start_row = torch.minimum(collision_start_row, rows - collision_height)
        collision_start_col = torch.minimum(collision_start_col, cols - collision_width)
        row_indices = torch.arange(rows, device=self.device)[None, :, None]
        col_indices = torch.arange(cols, device=self.device)[None, None, :]
        collision_region = (
            (row_indices >= collision_start_row[:, None, None])
            & (row_indices < (collision_start_row + collision_height)[:, None, None])
            & (col_indices >= collision_start_col[:, None, None])
            & (col_indices < (collision_start_col + collision_width)[:, None, None])
        )
        grid[~is_safe] = collision_region[~is_safe].to(grid.dtype)
        grid = _fill_non_overlapping_grid_regions(
            grid,
            density_range=tuple(cfg["density_range"]),
            rectangle_rows_range=rectangle_rows_range,
            rectangle_cols_range=rectangle_cols_range,
            forbidden=foot_grid,
        )
        # Safe maps start empty and the filler reserves their complete foot
        # footprint. Collision maps retain the dedicated overlapping region.
        grid = grid.flatten(start_dim=1)
        history_length = int(cfg["history_length"])
        for group in cfg["grid_groups"]:
            group_obs = fake_obs[group]
            try:
                _replace_grid_history(group_obs, grid, history_length)
            except ValueError as error:
                raise ValueError(f"Invalid counterfactual observation group '{group}': {error}") from error

        rewards = torch.where(
            is_safe,
            torch.full_like(foot_x, float(cfg["safe_reward"])),
            torch.full_like(foot_x, float(cfg["collision_reward"])),
        ).unsqueeze(-1)
        with torch.no_grad():
            self.policy.act(fake_obs)
            old_log_prob = self.policy.get_actions_log_prob(fake_actions).unsqueeze(-1)
            old_mu = self.policy.action_mean.clone()
            old_sigma = self.policy.action_std.clone()
            values = self.policy.evaluate(fake_obs)
            advantages = (rewards - values).clamp(
                -float(cfg["advantage_clip"]), float(cfg["advantage_clip"])
            )
        self._counterfactual_samples.append(
            {
                "obs": fake_obs,
                "actions": fake_actions,
                "values": values,
                "advantages": advantages,
                "returns": rewards,
                "old_log_prob": old_log_prob,
                "mu": old_mu,
                "sigma": old_sigma,
            }
        )

    def _backfill_foothold_targets(
        self,
        step: int,
        previous_state: torch.Tensor,
        current_state: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Backfill labels for the contiguous airborne segment ending in a touchdown."""
        cfg = self.foothold_cfg
        supervision = self.storage.observations[cfg["supervision_group"]][: step + 1]
        targets = self.storage.observations[cfg["target_group"]]
        target_valid = self.storage.observations[cfg["valid_group"]]
        previous_contacts = previous_state[:, 8:10] > 0.5
        current_contacts = current_state[:, 8:10] > 0.5
        touchdowns = (~previous_contacts) & current_contacts & (~dones.bool()).unsqueeze(-1)

        time_indices = torch.arange(step + 1, device=self.device)
        touchdown_times = (step - time_indices + 1).to(torch.float32) * cfg["step_dt"]

        for foot_index in range(2):
            touchdown_envs = touchdowns[:, foot_index]
            if not touchdown_envs.any():
                continue

            airborne = supervision[..., 8 + foot_index] < 0.5
            contiguous_airborne = torch.flip(
                torch.cumprod(torch.flip(airborne.to(torch.int32), dims=[0]), dim=0),
                dims=[0],
            ).bool()
            label_mask = contiguous_airborne & touchdown_envs.unsqueeze(0)

            root_xy = supervision[..., :2]
            cos_yaw = supervision[..., 2]
            sin_yaw = supervision[..., 3]
            touchdown_xy = current_state[:, 4 + 2 * foot_index : 6 + 2 * foot_index]
            delta_xy = touchdown_xy.unsqueeze(0) - root_xy
            local_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
            local_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]

            target_offset = 3 * foot_index
            targets[: step + 1, :, target_offset][label_mask] = local_x[label_mask]
            targets[: step + 1, :, target_offset + 1][label_mask] = local_y[label_mask]
            expanded_times = touchdown_times.unsqueeze(1).expand_as(local_x)
            targets[: step + 1, :, target_offset + 2][label_mask] = expanded_times[label_mask]
            target_valid[: step + 1, :, foot_index][label_mask] = 1.0
        
    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # AMP discriminator loss and other info
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_score = 0
        mean_disc_demo_score = 0
        mean_foothold_loss = 0 if self.foothold_cfg is not None else None
        foothold_valid_samples = 0
        counterfactual_pool = None
        counterfactual_sample_count = 0
        if self._counterfactual_samples:
            counterfactual_pool = {
                key: torch.cat([sample[key] for sample in self._counterfactual_samples], dim=0)
                for key in self._counterfactual_samples[0]
            }
            counterfactual_sample_count = counterfactual_pool["actions"].shape[0]

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        # Iterate over batches
        mini_batch_idx = 0
        disc_updates_done = 0
        for samples, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = samples

            if counterfactual_pool is not None:
                synthetic_batch_size = max(
                    1,
                    round(actions_batch.shape[0] * float(self.counterfactual_cfg["batch_ratio"])),
                )
                synthetic_indices = torch.randint(
                    0, counterfactual_sample_count, (synthetic_batch_size,), device=self.device
                )
                obs_batch = torch.cat((obs_batch, counterfactual_pool["obs"][synthetic_indices]), dim=0)
                actions_batch = torch.cat((actions_batch, counterfactual_pool["actions"][synthetic_indices]), dim=0)
                target_values_batch = torch.cat(
                    (target_values_batch, counterfactual_pool["values"][synthetic_indices]), dim=0
                )
                advantages_batch = torch.cat(
                    (advantages_batch, counterfactual_pool["advantages"][synthetic_indices]), dim=0
                )
                returns_batch = torch.cat(
                    (returns_batch, counterfactual_pool["returns"][synthetic_indices]), dim=0
                )
                old_actions_log_prob_batch = torch.cat(
                    (old_actions_log_prob_batch, counterfactual_pool["old_log_prob"][synthetic_indices]), dim=0
                )
                old_mu_batch = torch.cat((old_mu_batch, counterfactual_pool["mu"][synthetic_indices]), dim=0)
                old_sigma_batch = torch.cat((old_sigma_batch, counterfactual_pool["sigma"][synthetic_indices]), dim=0)
            
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.foothold_cfg is not None:
                prediction = self.policy.foothold_prediction[:original_batch_size]
                target = obs_batch[self.foothold_cfg["target_group"]][:original_batch_size]
                valid = obs_batch[self.foothold_cfg["valid_group"]][:original_batch_size]
                scales = prediction.new_tensor(self.foothold_cfg["target_scales"]).repeat(2)
                element_loss = torch.nn.functional.smooth_l1_loss(
                    prediction,
                    target / scales,
                    reduction="none",
                )
                valid_elements = valid.repeat_interleave(3, dim=-1)
                valid_count = valid_elements.sum()
                if valid_count > 0:
                    foothold_loss = (element_loss * valid_elements).sum() / valid_count
                    foothold_valid_samples += int(valid.sum().item())
                else:
                    foothold_loss = prediction.sum() * 0.0
                loss += self.foothold_cfg["loss_coef"] * foothold_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # AMP discriminator loss
            with torch.no_grad():
                disc_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch) # [mini_batch_size, disc_obs_steps, disc_obs_dim]
                disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)
            
            mini_batch_size = disc_obs_batch_normed.shape[0]
            disc_score = self.amp_discriminator(disc_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            disc_demo_score = self.amp_discriminator(disc_demo_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            
            if self.loss_type == LossType.GAN:
                bce = torch.nn.BCEWithLogitsLoss()
                policy_loss = bce(
                    disc_score, torch.zeros_like(disc_score, device=self.device)
                )
                demo_loss = bce(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.LSGAN:
                policy_loss = torch.nn.MSELoss()(
                    disc_score, -1 * torch.ones_like(disc_score, device=self.device)
                )
                demo_loss = torch.nn.MSELoss()(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.WGAN:
                disc_loss = - torch.mean(disc_demo_score) + torch.mean(disc_score)
            else: 
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")

            disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=disc_demo_obs_batch_normed.reshape(mini_batch_size, -1),
                scale=self.amp_cfg["grad_penalty_scale"]
            )
            disc_total_loss = disc_loss + disc_grad_penalty

            # Whether to update the discriminator this mini-batch step
            do_disc_update = (mini_batch_idx % self.disc_update_interval == 0)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            # Compute the gradients for AMP discriminator (conditional)
            if do_disc_update:
                self.disc_optimizer.zero_grad()
                disc_total_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # Apply the gradients for AMP discriminator (conditional)
            if do_disc_update:
                self.disc_optimizer.step()
                self.amp_discriminator.update_normalization(disc_obs_batch)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # AMP: scores logged every step (saturation tracking); losses only when updating
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            if do_disc_update:
                mean_disc_loss += disc_loss.item()
                mean_disc_grad_penalty += disc_grad_penalty.item()
                disc_updates_done += 1

            mini_batch_idx += 1
            if mean_foothold_loss is not None:
                mean_foothold_loss += foothold_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # disc scores are accumulated every step; losses only on actual update steps
        mean_disc_score /= num_updates
        mean_disc_demo_score /= num_updates
        if disc_updates_done > 0:
            mean_disc_loss /= disc_updates_done
            mean_disc_grad_penalty /= disc_updates_done
        if mean_foothold_loss is not None:
            mean_foothold_loss /= num_updates

        # Clear the storage
        self.storage.clear()
        self._counterfactual_samples.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["amp/disc_loss"] = mean_disc_loss
        loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["amp/disc_score"] = mean_disc_score
        loss_dict["amp/disc_demo_score"] = mean_disc_demo_score
        if mean_foothold_loss is not None:
            loss_dict["foothold/loss"] = mean_foothold_loss
            loss_dict["foothold/valid_samples"] = foothold_valid_samples / num_updates
        if self.counterfactual_cfg is not None:
            loss_dict["counterfactual/samples"] = counterfactual_sample_count

        return loss_dict
