import torch
from dataclasses import dataclass


@dataclass
class PatchifyMeta:
    seq_len: int      # original T
    d_latent: int     # original D
    patch_dim: int    # feature dim per patch
    pad_d: int        # how many feature dims were padded


def patchify_actions(latents: torch.Tensor, patch_dim: int):
    """
    Patchify along the LAST dimension only.

    Args:
        latents: (B, T, D)  -- typically T = 1 for your use case.
        patch_dim: size of each patch along the feature axis.

    Returns:
        patches: (B, N_patches, patch_dim)
            where N_patches = T * ceil(D / patch_dim)
        meta: PatchifyMeta with info to invert.
    """
    if patch_dim <= 0:
        raise ValueError(f"patch_dim must be > 0, got {patch_dim}")

    B, T, D = latents.shape

    # pad feature dim to multiple of patch_dim
    pad_d = (patch_dim - (D % patch_dim)) % patch_dim
    if pad_d > 0:
        pad = latents.new_zeros(B, T, pad_d)
        latents = torch.cat([latents, pad], dim=-1)  # (B, T, D_pad)
    D_pad = latents.size(-1)

    # number of patches along feature dim
    n = D_pad // patch_dim  # integer

    # reshape: (B, T, D_pad) -> (B, T, n, patch_dim) -> (B, T*n, patch_dim)
    latents_4d = latents.view(B, T, n, patch_dim)
    patches = latents_4d.reshape(B, T * n, patch_dim)  # (B, N_patches, patch_dim)

    meta = PatchifyMeta(
        seq_len=T,
        d_latent=D,
        patch_dim=patch_dim,
        pad_d=pad_d,
    )
    return patches, meta


def unpatchify_actions(patches: torch.Tensor, meta: PatchifyMeta):
    """
    Inverse of patchify_actions.

    Args:
        patches: (B, N_patches, patch_dim)
        meta: PatchifyMeta returned by patchify_actions

    Returns:
        latents: (B, seq_len, d_latent)  -- original (unpadded) shape
    """
    B, N_patches, patch_dim = patches.shape
    if patch_dim != meta.patch_dim:
        raise ValueError(
            f"Patch dim mismatch: got {patch_dim}, expected {meta.patch_dim}"
        )

    T = meta.seq_len
    D = meta.d_latent
    D_pad = D + meta.pad_d

    if N_patches % T != 0:
        raise ValueError(
            f"N_patches={N_patches} not divisible by seq_len={T}; "
            "cannot reshape back."
        )
    n = N_patches // T  # number of patches along feature dim

    # (B, N_patches, patch_dim) -> (B, T, n, patch_dim) -> (B, T, D_pad)
    patches_4d = patches.view(B, T, n, patch_dim)
    latents_pad = patches_4d.reshape(B, T, D_pad)  # (B, T, D_pad)

    # remove feature padding
    latents = latents_pad[:, :, :D]  # (B, T, D)
    return latents


if __name__ == "__main__":
    # Example: latents of shape (B, 1, D)
    B = 4
    T = 1          # sequence length (can be >1 as well)
    D_latent = 512 # latent dim
    patch_dim = 16 # feature dim per patch

    latents = torch.randn(B, T, D_latent)
    print(f"Original latents shape: {latents.shape}")   # (B, 1, D_latent)

    # Patchify
    patches, meta = patchify_actions(latents, patch_dim=patch_dim)
    print(f"Patches shape: {patches.shape}")            # (B, N_patches, patch_dim)

    # (Optional) simulate a DiT head round-trip
    patches_hat = patches.clone()

    # Unpatchify
    latents_recon = unpatchify_actions(patches_hat, meta)
    print(f"Reconstructed latents shape: {latents_recon.shape}")  # (B, 1, D_latent)

    # Check reconstruction error
    diff = (latents - latents_recon).abs().max().item()
    print(f"Max |original - recon|: {diff:.6f}")