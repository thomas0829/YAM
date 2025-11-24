# diffusion_rgb_encoder_vit_tokens.py

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required for DiffusionRgbEncoderViT. Install with:\n"
        "  pip install open_clip_torch"
    ) from e


# Default OpenAI CLIP normalization stats
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DiffusionRgbEncoderViT(nn.Module):
    """
    Encodes an RGB image using the CLIP ViT vision encoder.

    Depending on `image_feature_aggregation`, it can return:
      - "cls"       -> CLS token only:             (B, 1, D)
      - "patch"     -> patch tokens only:          (B, N_patches, D)
      - "cls+patch" -> CLS + patch tokens concat:  (B, 1+N_patches, D)

    Expected DiffusionConfig fields (all optional, defaults provided):
        - image_encoder_model_name: str, e.g. 'ViT-B-32'
        - image_encoder_pretrained: str, e.g. 'openai'
        - image_feature_dim: int, desired output dim (if not set, uses CLIP dim)
        - freeze_image_encoder: bool, whether to freeze CLIP weights (default True)
        - normalize_image_features: bool, whether to L2-normalize features
        - crop_ratio: Optional[float]  # e.g. 0.95
        - crop_is_random: bool
        - image_feature_aggregation: str in {"patch", "cls", "cls+patch"}
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()

        # ----------------------------------
        # Cropping / preprocessing
        # ----------------------------------
        if getattr(config, "crop_ratio", None) is not None:
            images_shape = next(iter(config.image_features.values())).shape  # (C, H, W)
            H, W = images_shape[1], images_shape[2]
            ch = int(round(H * config.crop_ratio))
            cw = int(round(W * config.crop_ratio))
            self.crop_shape = (ch, cw)
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(self.crop_shape)
            if getattr(config, "crop_is_random", False):
                self.maybe_random_crop = torchvision.transforms.RandomCrop(self.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
            self.crop_shape = None

        # ----------------------------------
        # CLIP vision backbone (ViT)
        # ----------------------------------
        model_name = getattr(config, "image_encoder_model_name", "ViT-B-32")
        pretrained = getattr(config, "image_encoder_pretrained", "openai")
        self.freeze_image_encoder = getattr(config, "freeze_image_encoder", True)
        self.normalize_image_features = getattr(config, "normalize_image_features", True)

        # "patch", "cls", or "cls+patch"
        self.feature_aggregation = getattr(config, "image_feature_aggregation", "patch")
        assert self.feature_aggregation in {"patch", "cls", "cls+patch"}, \
            f"Invalid image_feature_aggregation: {self.feature_aggregation}"

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = self.clip_model.visual

        if self.freeze_image_encoder:
            for p in self.visual.parameters():
                p.requires_grad = False

        # CLIP expected image size
        if hasattr(self.visual, "image_size"):
            if isinstance(self.visual.image_size, int):
                self.clip_image_size: Tuple[int, int] = (
                    self.visual.image_size,
                    self.visual.image_size,
                )
            else:
                self.clip_image_size = tuple(self.visual.image_size)
        else:
            if self.crop_shape is not None:
                self.clip_image_size = tuple(self.crop_shape)
            else:
                images_shape = next(iter(config.image_features.values())).shape
                self.clip_image_size = (images_shape[1], images_shape[2])

        # Infer CLIP feature dim from pooled encoding
        with torch.no_grad():
            dummy = torch.randn(
                1, 3, self.clip_image_size[0], self.clip_image_size[1]
            )
            dummy_features = self.clip_model.encode_image(dummy)  # (1, clip_dim)
        clip_dim = dummy_features.shape[-1]

        # Desired output dim (can be same as CLIP or projected)
        self.feature_dim = getattr(config, "image_feature_dim", clip_dim)

        # Projection head applied per token
        self.out = nn.Linear(clip_dim, self.feature_dim)
        self.relu = nn.ReLU()

        # CLIP normalization buffers
        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

        del self.clip_model

    # -------------------------
    # Preprocessing
    # -------------------------
    def _preprocess_images(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W) in [0, 1] -> (B, 3, H_clip, W_clip) normalized.
        """
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        _, _, h, w = x.shape
        target_h, target_w = self.clip_image_size
        if (h, w) != (target_h, target_w):
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        x = (x - self.clip_mean) / self.clip_std
        return x

    # -------------------------
    # ViT tokens (CLS + patches)
    # -------------------------
    def _encode_tokens(self, x: Tensor) -> Tensor:
        """
        Run the CLIP ViT visual tower but KEEP all tokens (CLS + patches).

        Returns:
            tokens: (B, 1 + N_patches, clip_dim)
        """
        visual = self.visual
        B = x.shape[0]

        # 1) Patch embedding conv
        x = visual.conv1(x)                    # (B, C', H', W')
        x = x.reshape(B, x.shape[1], -1)       # (B, C', H'*W')
        x = x.permute(0, 2, 1)                 # (B, N_patches, C')

        # 2) CLS token
        class_emb = visual.class_embedding.to(x.dtype)
        class_emb = class_emb + torch.zeros(
            B, 1, class_emb.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([class_emb, x], dim=1)   # (B, 1 + N_patches, C')

        # 3) Positional embeddings
        pos = visual.positional_embedding.to(x.dtype)  # (1 + N_patches, C')
        if pos.shape[0] != x.shape[1]:
            raise ValueError(
                f"Positional embedding length {pos.shape[0]} "
                f"does not match token length {x.shape[1]}."
            )
        x = x + pos

        # 4) Pre-transformer LN
        x = visual.ln_pre(x)

        # 5) Transformer
        x = x.permute(1, 0, 2)  # (1 + N, B, C')
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, 1 + N, C')

        # 6) Post-LN + proj per token
        if visual.ln_post is not None:
            x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj

        return x  # (B, 1 + N_patches, clip_dim)

    # -------------------------
    # Public forward
    # -------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].

        Returns:
            Depending on image_feature_aggregation:
                - "cls":       (B, 1, feature_dim)
                - "patch":     (B, N_patches, feature_dim)
                - "cls+patch": (B, 1+N_patches, feature_dim)
        """
        device = next(self.parameters()).device
        x = x.to(device)

        # Preprocess (crop + resize + normalize)
        x = self._preprocess_images(x)

        grad_context = (
            torch.enable_grad
            if (self.training and not self.freeze_image_encoder)
            else torch.no_grad
        )
        with grad_context():
            tokens = self._encode_tokens(x)  # (B, 1+N_patches, clip_dim)

        # Split CLS vs patches
        cls_tok = tokens[:, :1, :]      # (B, 1, C)
        patch_toks = tokens[:, 1:, :]   # (B, N_patches, C)

        if self.feature_aggregation == "cls":
            feats = cls_tok
        elif self.feature_aggregation == "patch":
            feats = patch_toks
        elif self.feature_aggregation == "cls+patch":
            feats = torch.cat([cls_tok, patch_toks], dim=1)
        else:
            raise RuntimeError(f"Unexpected feature_aggregation: {self.feature_aggregation}")

        # Optional L2-normalize per token
        if self.normalize_image_features:
            feats = F.normalize(feats, dim=-1)

        # Per-token projection + non-linearity
        feats = self.out(feats)
        feats = self.relu(feats)  # (B, N_tokens, feature_dim)
        return feats