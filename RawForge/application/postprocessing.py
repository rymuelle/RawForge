import torch

def match_colors_linear(
    src: torch.Tensor, 
    tgt: torch.Tensor, 
    sample_fraction: float = 0.05
):
    """
    Fit per-channel affine color transforms:
        tgt ≈ scale * src + bias

    Args:
        src: [B, C, H, W] source tensor
        tgt: [B, C, H, W] target tensor
        sample_fraction: fraction of pixels to use for fitting

    Returns:
        transformed_src: source after color matching
        scale: [B, C]
        bias:  [B, C]
    """

    B, C, H, W = src.shape
    device = src.device

    # Flatten spatial dims
    src_flat = src.view(B, C, -1)
    tgt_flat = tgt.view(B, C, -1)

    # Sample subset of pixels
    N = src_flat.shape[-1]
    k = max(64, int(N * sample_fraction))

    idx = torch.randint(0, N, (k,), device=device)

    src_s = src_flat[..., idx]  # [B, C, k]
    tgt_s = tgt_flat[..., idx]

    # Compute scale and bias using least squares
    # scale = cov(src, tgt) / var(src)
    src_mean = src_s.mean(-1, keepdim=True)
    tgt_mean = tgt_s.mean(-1, keepdim=True)

    src_centered = src_s - src_mean
    tgt_centered = tgt_s - tgt_mean

    var_src = (src_centered ** 2).mean(-1)
    cov = (src_centered * tgt_centered).mean(-1)

    scale = cov / (var_src + 1e-8)            # [B, C]
    bias = tgt_mean.squeeze(-1) - scale * src_mean.squeeze(-1)

    # Apply correction
    scale_ = scale.view(B, C, 1, 1)
    bias_ = bias.view(B, C, 1, 1)
    transformed = src * scale_ + bias_

    return transformed, scale, bias


def scaled_dot_product(x1, x2, eps=1e-6):
    dot = (x1 * x2).sum(axis=2, keepdims=True)
    x1_mag = (x1 * x1).sum(axis=2, keepdims=True) ** .5
    x2_mag = (x2 * x2).sum(axis=2, keepdims=True) ** .5
    return dot/(x1_mag+x2_mag+eps)


def postprocess(img, denoised, lumi_blend=0, chroma_blend=0, eps=1e-6):
    # Suggested by Jakob Andrén
    dot = (img * denoised).sum(axis=2, keepdims=True)
    img_mag = (img * img).sum(axis=2, keepdims=True) ** .5
    denoised_mag = (denoised * denoised).sum(axis=2, keepdims=True) ** .5
    # Project denoised along original image vector
    lumi = dot / (denoised_mag ** 2 + eps) * denoised
    chroma = img - lumi 
    output = (1-lumi_blend) * denoised + lumi * (lumi_blend) + chroma_blend * chroma

    return output
