# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from typing import Tuple

import timm
import numpy as np
import logging

import math
from typing import Tuple
import copy

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.models.vision_transformer import Mlp, use_fused_attn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.positional_embedding import FixedSinusoidalPosEmb
from lerobot.policies.diffusion.positional_embedding import LearnedSinusoidalPosEmb

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale

            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            # Add attention mask if provided
            if attn_mask is not None:
                attn_scores += attn_mask

            # Apply softmax to get attention weights (softmax is applied along the last dimension)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Dropout on attention weights (if dropout is used)
            attn_weights = self.attn_drop(attn_weights)

            # Apply attention weights to value tensor (V)
            x = torch.matmul(attn_weights, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask) # norm, scale&shift, attn, scale,
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SteerDP, adapted from ScaleDP.
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def clone_cond(x):
    if isinstance(x, torch.Tensor):
        # keeps it in the autograd graph, unlike detach()
        return x.clone()
    elif isinstance(x, list):
        return [clone_cond(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(clone_cond(v) for v in x)
    elif isinstance(x, dict):
        return {k: clone_cond(v) for k, v in x.items()}
    else:
        # for non-tensor leaves, normal deepcopy is fine
        return copy.deepcopy(x)

class DiffusionTransformer(nn.Module):
    """
    Diffusion models with a Transformer denoising model.
    """
    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        self.cond_embedder = nn.Linear(cond_dim, config.embed_dim)

        self.x_embedder = nn.Linear(config.action_feature.shape[0] if not config.use_vae else config.patch_size, config.embed_dim)
        # Will use fixed sin-cos embedding:     
        # self.pos_embed = nn.Parameter(torch.zeros(1, config.prediction_horizon, config.embed_dim))
        self.x_pos_embed = LearnedSinusoidalPosEmb(config.embed_dim)

        # self.context_pos_embed = LearnedSinusoidalPosEmb(global_cond_dim)
        
        # self.state_embedder = nn.Linear(config.robot_state_feature.shape[0], config.embed_dim)
        # self.state_pos_embed = LearnedSinusoidalPosEmb(config.embed_dim)

        # if config.env_state_feature:
        #     self.env_state_embedder = nn.Linear(config.env_state_feature.shape[0], config.embed_dim)
        #     self.env_state_pos_embed = LearnedSinusoidalPosEmb(config.embed_dim)
        
        # if config.image_features:
        #     self.image_embedder = nn.Linear(config.image_feature_dim, config.embed_dim)
        #     self.image_pos_embed = LearnedSinusoidalPosEmb(config.embed_dim)
        
        # if config.use_language:
        #     self.text_embedder = nn.Linear(config.text_feature_dim, config.embed_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(config.embed_dim, config.num_heads, mlp_ratio=config.mlp_ratio) for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.embed_dim, output_dim=config.action_feature.shape[0] if not config.use_vae else config.patch_size)
        # self.initialize_weights()

        self.config = config

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"number of parameters in DiT: {num_params:.2e}")

    def forward(self, x: torch.Tensor, timesteps_embed: torch.Tensor | int, global_cond=None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the transformer.
            timesteps_embed: (B, timesteps_embed_dim) tensor of timestep embeddings.
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        global_feature = []

        global_cond_temp = clone_cond(global_cond)

        state_embed = global_cond_temp.pop(0) # (B, S, D_state)
        B = state_embed.shape[0]
        state_embed = state_embed.view(B, -1)
        global_feature.append(state_embed)

        if self.config.image_features:
            img_embed = global_cond_temp.pop(0) # (B, S, N * L_img, D_img)
            img_embed = img_embed.view(B, -1)
            global_feature.append(img_embed)

        if self.config.env_state_feature:
            env_state_embed = global_cond_temp.pop(0) # (B, S, D_state)
            env_state_embed = env_state_embed.view(B, -1)
            global_feature.append(env_state_embed)

        if self.config.use_language:
            lang_features = global_cond_temp.pop(0) # (B, L_lang, D_lang)
            lang_features = lang_features.view(B, -1)
            global_feature.append(lang_features)

        global_feature = torch.cat(global_feature, axis=-1) # (B, global_cond_dim)
        B, T = global_feature.shape

        # global_feature += self.context_pos_embed(torch.arange(B, device=x.device))

        global_feature = torch.cat([timesteps_embed, global_feature], axis=-1)
        global_feature_embed = self.cond_embedder(global_feature)

        B, T, _ = x.shape
        x = self.x_embedder(x) + self.x_pos_embed(torch.arange(T, device=x.device)).expand(B, -1, -1)

        for block in self.blocks:
            # x = block(x, c, attn_mask=self.mask)  # (N, T, D)
            x = block(x, global_feature_embed, attn_mask=None)  # (N, T, D)
        x = self.final_layer(x, global_feature_embed)  # (N, T, output_dim)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.cond_embedder.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_embedder.bias, 0)

        # Zero-out adaLN modulation layers in ScaleDP blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the models into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Attention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

 






