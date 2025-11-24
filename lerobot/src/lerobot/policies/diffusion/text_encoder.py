# text_encoder.py

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Union

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip is required for DiffusionTextEncoder. Install with:\n"
        "  pip install open_clip_torch"
    ) from e


class DiffusionTextEncoder(nn.Module):
    """Encodes text into a feature vector (or sequence of vectors) using CLIP.

    This is the text analogue of DiffusionRgbEncoderViT. It can:
      - return a single sentence embedding (EOT-pooled, like standard CLIP), or
      - return token-level embeddings (one vector per token), optionally flattened.

    Expected DiffusionConfig fields (all optional, defaults provided):
        - text_encoder_model_name: str, e.g. 'ViT-B-32'
        - text_encoder_pretrained: str, e.g. 'openai'
        - text_feature_dim: int, desired output dim (if not set, uses CLIP dim)
        - freeze_text_encoder: bool, whether to freeze CLIP weights (default True)
        - normalize_text_features: bool, whether to L2-normalize features
        - text_feature_aggregation: str, one of ['eot', 'tokens', 'tokens_flat']
              'eot'         -> (B, 1, D)
              'tokens'      -> (B, L, D)
              'tokens_flat' -> (B, L*D)
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()

        model_name = getattr(config, "text_encoder_model_name", "ViT-B-32")
        pretrained = getattr(config, "text_encoder_pretrained", "openai")
        self.freeze_text_encoder = getattr(config, "freeze_text_encoder", True)
        self.normalize_text_features = getattr(config, "normalize_text_features", True)
        self.pooling_type = getattr(config, "text_feature_aggregation", "eot")  # 'eot' | 'tokens' | 'tokens_flat'

        # Get CLIP model
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Get tokenizer separately
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Optionally freeze CLIP weights
        if self.freeze_text_encoder:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Infer CLIP "output" text feature dimension with a dummy forward.
        # NOTE: This is the *projected* sentence embedding dimension (after text_projection).
        with torch.no_grad():
            dummy_tokens = self.tokenizer(["dummy"])
            dummy_features = self.clip_model.encode_text(dummy_tokens)
        clip_dim = dummy_features.shape[-1]

        # Desired output dim (can be same as CLIP or projected)
        self.feature_dim = getattr(config, "text_feature_dim", clip_dim)

        # Simple projection head (works for both sentence- and token-level features)
        self.out = nn.Linear(clip_dim, self.feature_dim)
        self.relu = nn.ReLU()

        del self.clip_model.visual

    # ------------ Internal helpers ------------

    def _encode_tokens_via_hook(
        self,
        tokens: Tensor,
        grad_context,
    ) -> Tensor:
        """Return per-token embeddings using CLIP internals.

        Steps:
          1. Run `encode_text` as usual (so we don't touch attn_mask / transformer directly).
          2. Use a forward hook on `ln_final` to grab token-level hidden states (B, L, width).
          3. Apply `text_projection` to every token to get (B, L, embed_dim), consistent
             with the sentence-level embedding dimension from encode_text.
        """
        m = self.clip_model
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        # Container where the hook will stash ln_final activations
        cache = {}

        def ln_final_hook(module, inp, out):
            # out: (B, L, width)
            cache["x"] = out

        handle = m.ln_final.register_forward_hook(ln_final_hook)
        try:
            with grad_context():
                # We ignore the return value; we only care about the hook.
                _ = m.encode_text(tokens)
        finally:
            handle.remove()

        if "x" not in cache:
            raise RuntimeError(
                "ln_final hook did not fire; open_clip version might differ from expected."
            )

        x = cache["x"]  # (B, L, width)
        # Apply the same text_projection used for the pooled EOT embedding, but to all tokens.
        # text_projection: (width, embed_dim)
        proj = m.text_projection.to(x.dtype).to(x.device)
        token_embeddings = x @ proj  # (B, L, embed_dim)

        return token_embeddings

    def _tokenize_if_needed(self, text: Union[List[str], Tensor]) -> Tensor:
        device = next(self.parameters()).device
        if isinstance(text, list):
            tokens = self.tokenizer(text)
        elif isinstance(text, Tensor):
            tokens = text
        else:
            raise TypeError(
                f"DiffusionTextEncoder expected a List[str] or Tensor, got {type(text)}"
            )
        return tokens.to(device)

    # ------------ Public API ------------

    def forward(self, text: Union[List[str], Tensor]) -> Tensor:
        """Encode text to feature(s).

        Args:
            text:
                - List[str]: raw sentences; will be tokenized with CLIP tokenizer.
                - Tensor: token IDs of shape (B, L) already produced by the CLIP tokenizer.

        Returns:
            Depending on `self.pooling_type`:
                - 'eot':         (B, D)
                - 'tokens':      (B, L, D)
                - 'tokens_flat': (B, L*D)
        """
        tokens = self._tokenize_if_needed(text)

        # Grad context: if frozen, we can safely disable gradients for speed.
        grad_context = (
            torch.enable_grad
            if (self.training and not self.freeze_text_encoder)
            else torch.no_grad
        )

        pooling = self.pooling_type.lower()

        if pooling == "eot":
            # Standard CLIP sentence embedding: EOT-pooled.
            with grad_context():
                features = self.clip_model.encode_text(tokens)  # (B, clip_dim)
            features = features.unsqueeze(1)  # (B, 1, clip_dim)
        else:
            # Token-level embeddings via hook.
            token_embeddings = self._encode_tokens_via_hook(tokens, grad_context)  # (B, L, clip_dim)

            if pooling == "tokens":
                features = token_embeddings  # (B, L, clip_dim)
            elif pooling == "tokens_flat":
                B, L, C = token_embeddings.shape
                features = token_embeddings.reshape(B, L * C)  # (B, L*C)
                # We want a Linear(clip_dim, feature_dim) but now input dim is L*C.
                # Easiest: apply out token-wise, then flatten AFTER projection.
                # So we redo: project token-wise, then flatten.
                token_embeddings = self.relu(self.out(token_embeddings))  # (B, L, feature_dim)
                if self.normalize_text_features:
                    token_embeddings = F.normalize(token_embeddings, dim=-1)
                return token_embeddings.reshape(B, L * self.feature_dim)
            else:
                raise ValueError(
                    f"Unknown text_feature_aggregation: {self.pooling_type}. "
                    f"Expected one of ['eot', 'tokens', 'tokens_flat']."
                )

        # Now `features` is either (B, clip_dim) or (B, L, clip_dim)
        if self.normalize_text_features:
            features = F.normalize(features, dim=-1)

        # Linear projection + ReLU; works for both 2D and 3D inputs.
        features = self.out(features)
        features = self.relu(features)
        return features


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        text_encoder_model_name="ViT-B-32",
        text_encoder_pretrained="openai",
        text_feature_dim=512,
        freeze_text_encoder=True,
        normalize_text_features=True,
        text_feature_aggregation="tokens",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = DiffusionTextEncoder(config).to(device)
    encoder.eval()

    demo_texts = [
        "pick up the red block",
        "place it in the green bin",
    ]
    print(f"Encoding {len(demo_texts)} demo sentences")

    with torch.no_grad():
        feats = encoder(demo_texts)

    print("Output shape:", feats.shape)
    print("Example embedding (first item, first token):")
    if feats.ndim == 3:
        # (B, N_tokens, D)
        print(feats[0, 0, :5])  # print first 5 dims
    else:
        # (B, D)
        print(feats[0, :5])