import numpy as np
import math


def norm_cdf(z):
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1 + erf_vec(z / np.sqrt(2)))


def norm_pdf(z):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)


def censored_linear_fit_twosided(
    x, y, clip_low=0, clip_high=1, max_iter=200, tol=1e-6, include_offset=True
):
    """
    Fit y ≈ a + b*x + ε, ε ~ N(0, σ²) under two-sided censoring:
    clip_low ≤ y_true ≤ clip_high
    Observed y are clipped to [clip_low, clip_high].
    Returns (a, b, sigma) estimated via EM.

    Parameters
    ----------
    x, y : array_like
        Input data.
    clip_low, clip_high : float or None
        Lower/upper clip levels. Can be None for one-sided clipping.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Relative tolerance for convergence.
    include_offset : bool
        If False, forces intercept a=0 (fit y ≈ b*x).

    Returns
    -------
    a, b, sigma : floats
        Estimated regression parameters.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        raise ValueError("Not enough data points.")

    # Initial guess (ordinary least squares)
    if include_offset:
        A = np.vstack([np.ones_like(x), x]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        b = np.dot(x, y) / np.dot(x, x)
        a = 0.0
    sigma = np.std(y - (a + b * x))

    for _ in range(max_iter):
        mu = a + b * x
        y_exp = y.copy()

        # Handle right-censoring (high clip)
        if clip_high is not None:
            high_mask = y >= clip_high - 1e-12
            if np.any(high_mask):
                z = (clip_high - mu[high_mask]) / sigma
                Phi = norm_cdf(z)
                phi = norm_pdf(z)
                one_minus_Phi = 1.0 - Phi
                lambda_ = np.zeros_like(z)
                valid = one_minus_Phi > 1e-15
                lambda_[valid] = phi[valid] / one_minus_Phi[valid]
                y_exp[high_mask] = mu[high_mask] + sigma * lambda_

        # Handle left-censoring (low clip)
        if clip_low is not None:
            low_mask = y <= clip_low + 1e-12
            if np.any(low_mask):
                z = (clip_low - mu[low_mask]) / sigma
                Phi = norm_cdf(z)
                phi = norm_pdf(z)
                lambda_ = np.zeros_like(z)
                valid = Phi > 1e-15
                lambda_[valid] = -phi[valid] / Phi[valid]
                y_exp[low_mask] = mu[low_mask] + sigma * lambda_

        # M-step: re-fit with imputed expectations
        if include_offset:
            A = np.vstack([np.ones_like(x), x]).T
            a_new, b_new = np.linalg.lstsq(A, y_exp, rcond=None)[0]
        else:
            b_new = np.dot(x, y_exp) / np.dot(x, x)
            a_new = 0.0

        sigma_new = np.std(y_exp - (a_new + b_new * x))

        if np.allclose([a, b, sigma], [a_new, b_new, sigma_new], rtol=tol, atol=tol):
            break

        a, b, sigma = a_new, b_new, sigma_new

    return a, b, sigma
