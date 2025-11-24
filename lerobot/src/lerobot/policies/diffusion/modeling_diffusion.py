#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    # get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, TASK

from lerobot.policies.diffusion.rgb_encoder import DiffusionRgbEncoder
from lerobot.policies.diffusion.rgb_encoder_vit import DiffusionRgbEncoderViT
from lerobot.policies.diffusion.text_encoder import DiffusionTextEncoder
from lerobot.policies.diffusion.positional_embedding import FixedSinusoidalPosEmb
from lerobot.policies.diffusion.positional_embedding import LearnedSinusoidalPosEmb
from lerobot.policies.diffusion.unet import DiffusionConditionalUnet1d
from lerobot.policies.diffusion.dit import DiffusionTransformer
from lerobot.policies.diffusion.mdt import MultimodalDiffusionTransformer
from lerobot.policies.diffusion.ditx import DiffusionTransformerX

from lerobot.policies.diffusion.vae.info_vae import InfoVAE
from lerobot.policies.diffusion.vae.patchifier import patchify_actions, unpatchify_actions, PatchifyMeta

from torch.distributions import Beta


def sample_beta(batch_size, s=0.999, alpha=1.0, beta=1.5, device="cuda"):
    """
    Samples timesteps using a shifted beta distribution with cutoff.

    Args:
        batch_size: Number of samples to generate
        s: Cutoff threshold (default 0.999 as used in paper)
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        device: Device to place tensors on

    Returns:
        Tensor of shape (batch_size, 1, 1) containing sampled timesteps in [0, s]
    """
    # Make the Beta parameters scalars (0-D), so batch_shape=()
    beta_dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )

    # This now returns shape (B, 1, 1), not (B, 1, 1, 1)
    raw_samples = beta_dist.sample((batch_size, 1, 1))  # (B, 1, 1)

    # Scale samples by s to get timesteps in [0, s]
    t = s * raw_samples
    return t


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        if self.config.use_language:
            tasks = batch.pop(TASK)
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        if self.config.use_language:
            batch[TASK] = tasks
        actions = self.diffusion.generate_actions(batch, noise=noise)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # if self.config.denoising_model_type == "unet" or self.config.denoising_model_type == "dit":
        #     assert self.config.image_feature_aggregation == "cls" and self.config.text_feature_aggregation == "eot"

        self.use_original_vision_backbone = "resnet" in self.config.image_encoder_model_name.lower()

        # Build observation encoders (depending on which observations are provided).
        # this is only used for unet and dit
        global_cond_dim = self.config.robot_state_feature.shape[0] * config.n_obs_steps
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoderViT(config) if not self.use_original_vision_backbone else DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images * config.n_obs_steps
            else:
                self.rgb_encoder = DiffusionRgbEncoderViT(config) if not self.use_original_vision_backbone else DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images * config.n_obs_steps
        
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0] * config.n_obs_steps

        if self.config.use_language:
            self.text_encoder = DiffusionTextEncoder(config)
            global_cond_dim += self.text_encoder.feature_dim * 1 # language is the same across previous steps

        self.diffusion_timestep_encoder = LearnedSinusoidalPosEmb(config.diffusion_step_embed_dim)

        if self.config.denoising_model_type == "unet":
            self.denoising_model = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        elif self.config.denoising_model_type == "dit":
            self.denoising_model = DiffusionTransformer(config, global_cond_dim=global_cond_dim)
        elif self.config.denoising_model_type == "mdt":
            self.denoising_model = MultimodalDiffusionTransformer(config)
        elif self.config.denoising_model_type == "ditx":
            self.denoising_model = DiffusionTransformerX(config)
        else:
            raise ValueError(f"Unsupported denoising model type {config.denoising_model_type}")

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

        if config.use_vae:
            self.vae = InfoVAE(
                nfeats=config.vae_in_channels,
                latent_dim=[config.vae_latent_size, config.vae_latent_dim],
                dropout=0.0,
            )
            state_dict = torch.load(config.vae_ckpt_path)
            self.vae.load_state_dict(state_dict)

            for p in self.vae.parameters():
                p.requires_grad = False

            self.vae.eval()

        # Initialize positional embedding MLP:
        if self.diffusion_timestep_encoder:
            nn.init.normal_(self.diffusion_timestep_encoder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.diffusion_timestep_encoder.mlp[2].weight, std=0.02)

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]) if not self.config.use_vae else (batch_size, self.config.vae_latent_dim // self.config.patch_size, self.config.patch_size),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        B, T, _ = sample.shape
        lengths = [T] * B

        if self.config.use_vae:
            meta = PatchifyMeta(
                seq_len=1,
                d_latent=self.config.vae_latent_dim,
                patch_dim=self.config.patch_size,
                pad_d=0,
            )
            # sample, meta = self.pad_encode_pachify(sample, lengths)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
            timestep_embed = self.diffusion_timestep_encoder(timestep)
            # Predict model output.
            model_output = self.denoising_model(
                sample,
                timestep_embed,
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        
        if self.config.use_vae:
            sample = self.unpatchify_decode_unpad(sample, lengths, meta)

        return sample
    

    @torch.no_grad()
    def conditional_sample_flow(self, batch_size, global_cond, generator=None):
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        x = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        N = self.num_inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((batch_size,), i / N, device=device)
            t_embed = self.diffusion_timestep_encoder(t)
            v = self.denoising_model(x, t_embed, global_cond=global_cond)
            x = x + dt * v

        return x
    

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> list:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]] # (B, S, D_state)
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera: # TODO: not implemented for new conditioning logic
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )

                if len(img_features.shape) < 3:
                    img_features = img_features.unsqueeze(1)

                img_seq_len, img_feat_dim = img_features.shape[-2:]
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) l ... -> b s (n l) ...", b=batch_size, s=n_obs_steps, l=img_seq_len
                )

            global_cond_feats.append(img_features) # (B, S, N * L_img, D_img)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Encode language and broadcast over the observation horizon.
        if self.config.use_language:
            # batch["task"] is a list[str] of length B
            # (or a token tensor if you decide to pre-tokenize upstream).
            lang_features = self.text_encoder(batch[TASK])  # (B, L_lang, D_lang)
            global_cond_feats.append(lang_features)

        # Concatenate features then flatten to (B, global_cond_dim).
        return global_cond_feats

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # list of feats

        # run sampling
        if self.config.training_objective == "diffusion":
            actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        elif self.config.training_objective == "flow_matching":
            actions = self.conditional_sample_flow(batch_size, global_cond=global_cond)
        
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch[ACTION]

        B, T, _ = trajectory.shape
        lengths = [T] * B

        if self.config.use_vae:
            trajectory, meta = self.pad_encode_pachify(trajectory, lengths)

        # Decide which training objective to use
        objective = getattr(self.config, "training_objective", "diffusion")

        if objective == "flow_matching":
            # Flow matching in whatever space `trajectory` currently lives in
            loss = self._compute_flow_matching_loss(
                trajectory=trajectory,
                global_cond=global_cond,
                batch=batch,
            )
            return loss

        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        timestep_embed = self.diffusion_timestep_encoder(timesteps)
        pred = self.denoising_model(noisy_trajectory, timestep_embed, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            if self.config.use_vae:
                pred = self.unpatchify_decode_unpad(pred, lengths, meta)
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
    

    def _compute_flow_matching_loss(
        self,
        trajectory: torch.Tensor,
        global_cond: list,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """
        Flow matching objective on the given trajectory.

        Matches ManiFlow's get_flow_velocity:
          - x0: noise  ~ N(0, I)
          - x1: data   = trajectory (actions)
          - x(t) = (1 - t) * x0 + t * x1
          - v_target = x1 - x0 (constant along the path)
        """
        B = trajectory.shape[0]
        device = trajectory.device

        # ---- sample time t ~ Beta(alpha, beta) scaled to [0, s] ----
        # same as ManiFlow's sample_t(mode="beta")
        t_3d = sample_beta(
            batch_size=B,
            s=0.999,
            alpha=1.0,
            beta=1.5,
            device=device,
        )  # (B, 1, 1) in [0, s]

        # For the path we want shape (B, 1, 1); for embedding we want (B,)
        t_b = t_3d             # (B, 1, 1)
        t = t_3d.view(B)       # (B,)

        # Straight-line path between x0 = noise and x1 = data
        x0 = torch.randn_like(trajectory, device=device)  # noise
        x1 = trajectory                                   # data / actions

        # x(t) = (1 - t) * x0 + t * x1
        xt = (1.0 - t_b) * x0 + t_b * x1                 # broadcast over (L, D)

        # Velocity along the path: v(t, x) = x1 - x0 (constant in t)
        v_target = x1 - x0

        # Time embedding (LearnedSinusoidalPosEmb can take float t in [0, s])
        t_embed = self.diffusion_timestep_encoder(t)  # (B, D_t)

        # Predict velocity field
        v_pred = self.denoising_model(xt, t_embed, global_cond=global_cond)

        # MSE in the same space
        loss = F.mse_loss(v_pred, v_target, reduction="none")

        # Optional masking (only safe if we're still in action space, not patchified)
        if self.config.do_mask_loss_for_padding and not getattr(self.config, "use_vae", False):
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]  # (B, T)
            if in_episode_bound.shape[:2] == loss.shape[:2]:
                loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


    def pad_encode_pachify(self, actions: torch.Tensor, lengths) -> torch.Tensor:
        B, T, D_act = actions.shape
        target_dim = self.config.vae_in_channels

        if D_act > target_dim:
            raise ValueError(
                f"Action dim {D_act} > target_dim {target_dim} "
                "(config.vae_in_channels)"
            )

        # Only pad if needed
        if D_act < target_dim:
            pad_dim = target_dim - D_act
            # zeros with same dtype/device as actions
            pad = actions.new_zeros(B, T, pad_dim)
            actions = torch.cat([actions, pad], dim=-1)  # (B, T, target_dim)

        latent, dist, mu, std = self.vae.encode(actions, lengths)
        latent = latent.permute(1, 0, 2) # [B, 1, D_latent]

        pachified_latent, meta = patchify_actions(latent, self.config.patch_size) # [B, N_patch, D_patch]

        return pachified_latent, meta


    def unpatchify_decode_unpad(self, pachified_latent: torch.Tensor, lengths, meta) -> torch.Tensor:
        latent = unpatchify_actions(pachified_latent, meta) # [B, T_latent, D_latent]

        latent = latent.permute(1, 0, 2)
        recon = self.vae.decode(latent, lengths) # [B, T, target_dim]

        actions = recon[:, :, :self.config.action_dof] # [B, T, D_act]

        return actions
