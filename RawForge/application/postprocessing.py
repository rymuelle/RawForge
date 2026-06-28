import numpy as np
from RawForge.application.helpers.censored_fit import censored_linear_fit_twosided


def match_colors_linear(
    src: np.ndarray,
    tgt: np.ndarray,
    sample_fraction: float = 0.05,
    censored_fit: bool = True,
):
    """
    Fit per-channel affine color transforms using NumPy.
    src/tgt: [B, C, H, W]
    """
    B, C, H, W = src.shape

    # Flatten spatial dims: [B, C, H*W]
    src_flat = src.reshape(B, C, -1)
    tgt_flat = tgt.reshape(B, C, -1)

    # Sample subset of pixels
    N = src_flat.shape[-1]
    k = max(64, int(N * sample_fraction))

    # Generate random indices
    idx = np.random.randint(0, N, size=(k,))

    src_s = src_flat[..., idx]  # [B, C, k]
    tgt_s = tgt_flat[..., idx]

    # Compute scale and bias using least squares
    src_mean = src_s.mean(axis=-1, keepdims=True)
    tgt_mean = tgt_s.mean(axis=-1, keepdims=True)

    src_centered = src_s - src_mean
    tgt_centered = tgt_s - tgt_mean

    var_src = (src_centered**2).mean(axis=-1)
    cov = (src_centered * tgt_centered).mean(axis=-1)

    scale = cov / (var_src + 1e-8)  # [B, C]
    # Squeeze the mean to [B, C] for bias calculation
    bias = tgt_mean.squeeze(-1) - scale * src_mean.squeeze(-1)

    # Apply correction: reshape for broadcasting to [B, C, H, W]
    scale_ = scale[:, :, np.newaxis, np.newaxis]
    bias_ = bias[:, :, np.newaxis, np.newaxis]
    transformed = src * scale_ + bias_

    if censored_fit:
        a, b, sigma = censored_linear_fit_twosided(
            tgt_s.astype(np.float32()), src_s.astype(np.float32())
        )
        transformed = a + b * transformed

    return transformed, scale, bias


def scaled_dot_product(x1, x2, eps=1e-6):
    # Sum along axis 2 (C if input is B, H, C)
    dot = (x1 * x2).sum(axis=2, keepdims=True)
    x1_mag = np.sqrt((x1 * x1).sum(axis=2, keepdims=True))
    x2_mag = np.sqrt((x2 * x2).sum(axis=2, keepdims=True))
    return dot / (x1_mag + x2_mag + eps)


def postprocess(
    img,
    denoised,
    lumi_blend=0,
    chroma_blend=0,
    eps=1e-6,
    clip_highlights=False,
    affine=False,
):
    if affine:
        denoised, _, _ = match_colors_linear(denoised, img)


    if lumi_blend > 0:
        dot = (img * denoised).sum(axis=1, keepdims=True)
        denoised_mag = (denoised * denoised).sum(axis=1, keepdims=True) ** 0.5
        # Project denoised along original image vector
        lumi = dot / (denoised_mag**2 + eps) * denoised
        chroma = img - lumi
        denoised = (1 - lumi_blend) * denoised + lumi * (lumi_blend) + chroma_blend * chroma
    

    if clip_highlights:
        denoised = clip_highlights_func(img, denoised)

    return denoised


def clip_highlights_func(img, denoised):
    out = np.copy(denoised)
    mask = img == 1
    out[mask] = 1
    return out
