"""Distribution-matching regularizers for EEG-LeJEPA.

Two anti-collapse objectives, selectable via ``reg_type``:

  - ``sigreg``: the actual SIGReg from LeJEPA (Balestriero & LeCun, 2025).
    Sketched Isotropic Gaussian Regularization. By Cramer-Wold, z ~ N(0, I)
    iff every 1D projection onto a unit direction is N(0, 1). We draw random
    unit directions, project the batch, and test each projection against the
    standard normal via the empirical characteristic function (Epps-Pulley
    goodness-of-fit). One test jointly enforces zero mean, unit variance, and
    Gaussian shape. Cost is linear in batch size.

  - ``vicreg``: variance + covariance terms (VICReg-style). This is what the
    repo previously (mis)labelled "SIGReg". Kept as an ablation so the prior
    var+cov results stay reproducible.
"""

import torch
import torch.nn.functional as F


def vicreg_loss(z: torch.Tensor):
    """VICReg-style variance + covariance regularization.

    Args:
        z: [M, D] embeddings, flattened over batch and tokens.
    Returns:
        (var_loss, cov_loss)
    """
    D = z.shape[1]
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
    cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D
    return var_loss, cov_loss


def sigreg_loss(z: torch.Tensor, n_slices: int = 64, n_freqs: int = 17,
                t_max: float = 5.0, eps: float = 1e-6) -> torch.Tensor:
    """True SIGReg: sketched isotropic-Gaussian regularization.

    Tests whether ``z ~ N(0, I)`` by projecting onto random unit directions
    (Cramer-Wold sketch) and matching each 1D projection's empirical
    characteristic function to the standard-normal CF ``exp(-t^2/2)``
    (Epps-Pulley test). Linear in M (batch * tokens).

    Args:
        z: [M, D] embeddings.
        n_slices: number of random projection directions (resampled each call).
        n_freqs: number of characteristic-function frequencies in the quadrature.
        t_max: largest frequency in the grid; the Gaussian weight makes the
            integrand negligible beyond ~3-4.
    """
    M, D = z.shape

    # Random unit directions on the sphere; resampled each step (Monte-Carlo
    # over the Cramer-Wold integral). Treated as constants (no gradient).
    V = torch.randn(D, n_slices, device=z.device, dtype=z.dtype)
    V = V / (V.norm(dim=0, keepdim=True) + eps)
    u = z @ V                                            # [M, n_slices]

    # Positive frequency grid (the integrand is even in t).
    t = torch.linspace(t_max / n_freqs, t_max, n_freqs,
                       device=z.device, dtype=z.dtype)   # [K]
    weight = torch.exp(-0.5 * t * t)                     # Epps-Pulley weight [K]
    target = torch.exp(-0.5 * t * t)                     # standard-normal CF [K]

    # Empirical characteristic function of each projection at each frequency.
    tu = u.unsqueeze(-1) * t                             # [M, n_slices, K]
    cos_part = torch.cos(tu).mean(dim=0)                 # [n_slices, K]
    sin_part = torch.sin(tu).mean(dim=0)                 # [n_slices, K]

    # |phi_hat(t) - phi_0(t)|^2 with phi_0 real (sin target = 0).
    disc = (cos_part - target).pow(2) + sin_part.pow(2)  # [n_slices, K]

    per_slice = (disc * weight).sum(dim=-1) / weight.sum()
    return per_slice.mean()


def distribution_reg(z: torch.Tensor, reg_type: str = "sigreg", **kwargs):
    """Dispatch to the chosen regularizer.

    Returns:
        (reg_total, info) where info holds per-component scalars for logging:
        keys ``sigreg``, ``var``, ``cov`` (unused ones are zero tensors).
    """
    zero = torch.zeros((), device=z.device, dtype=z.dtype)
    if reg_type == "sigreg":
        sr = sigreg_loss(z, **kwargs)
        return sr, {"sigreg": sr, "var": zero, "cov": zero}
    if reg_type == "vicreg":
        var_loss, cov_loss = vicreg_loss(z)
        return var_loss + cov_loss, {"sigreg": zero, "var": var_loss, "cov": cov_loss}
    raise ValueError(f"Unknown reg_type: {reg_type!r} (expected 'sigreg' or 'vicreg')")
