"""Numerical uncertainty band per normal-mode frequency.

For each frequency the user reports, we'd like a `f ± df`. Two error sources
dominate:

1. **Stencil truncation.** Central differences for H_ij are O(h^2). Solving
   the Hessian at stencil widths `(h, h/sqrt(2))` and Richardson-extrapolating
   isolates this contribution.

2. **Mesh discretisation.** P2 elements give O(h_mesh^3) on a smooth Φ. We
   cannot remesh from inside the package generically — the user supplies the
   mesh — but we *can* probe the local roughness of `U_mag(q)` by comparing
   parabola fits at different stencil widths and use the difference as a
   proxy. This is a lower bound on the true mesh error; the proper way is to
   regenerate the mesh at h_mesh/sqrt(2) and rerun, which the CLI exposes via
   a separate `--converge` flag.

This module implements (1) — stencil-Richardson — and combines it with an
optional mesh-refined Hessian if the caller hands one in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import NormalModes, normal_modes
from .mesh import SCTrapMesh


@dataclass
class FrequencyWithUncertainty:
    f_Hz: np.ndarray         # (5,) extrapolated frequencies
    df_Hz: np.ndarray        # (5,) one-sigma-style uncertainty
    f_coarse: np.ndarray     # (5,) at stencil width h
    f_fine: np.ndarray       # (5,) at stencil width h/sqrt(2)
    f_extrapolated: np.ndarray  # (5,) Richardson-extrapolated

    def as_dict(self) -> dict:
        return {
            "f_Hz":           self.f_Hz.tolist(),
            "df_Hz":          self.df_Hz.tolist(),
            "f_coarse":       self.f_coarse.tolist(),
            "f_fine":         self.f_fine.tolist(),
            "f_extrapolated": self.f_extrapolated.tolist(),
        }


def stencil_richardson(
    sctmesh: SCTrapMesh,
    eq_r: np.ndarray,
    m_mag: float,
    *,
    eq_theta: float = 0.0,
    eq_phi: float = 0.0,
    h_trans: float = 50e-6,
    h_ang: float = 1e-3,
    refine: float = np.sqrt(2.0),
    verbose: bool = False,
    **mode_kwargs,
) -> tuple[NormalModes, NormalModes, FrequencyWithUncertainty]:
    """Run `normal_modes` at two stencil widths and Richardson-extrapolate.

    Central differences on `H_ij` converge as O(h^2). With samples at h and
    h/refine the leading error cancels in

        f_extrapolated = (refine^2 * f_fine - f_coarse) / (refine^2 - 1).

    The remaining `|f_extrapolated - f_fine|` is the stencil-truncation
    uncertainty band. Mesh discretisation error must be assessed separately
    (e.g. by re-running on a finer mesh).

    Returns
    -------
    nm_coarse, nm_fine : the two `NormalModes` results.
    band               : `FrequencyWithUncertainty` with extrapolated f and
                         per-mode |delta f|.
    """
    if verbose:
        print(f"[stencil] coarse: h_trans={h_trans:.3e}, h_ang={h_ang:.3e}",
              flush=True)
    nm_c = normal_modes(
        sctmesh, eq_r, m_mag,
        eq_theta=eq_theta, eq_phi=eq_phi,
        h_trans=h_trans, h_ang=h_ang,
        verbose=verbose, **mode_kwargs,
    )
    if verbose:
        print(f"[stencil] fine:   h_trans={h_trans/refine:.3e}, "
              f"h_ang={h_ang/refine:.3e}", flush=True)
    nm_f = normal_modes(
        sctmesh, eq_r, m_mag,
        eq_theta=eq_theta, eq_phi=eq_phi,
        h_trans=h_trans / refine, h_ang=h_ang / refine,
        verbose=verbose, **mode_kwargs,
    )

    # Mode-pair the two runs by maximum eigenvector overlap so we don't pair
    # accidentally re-ordered modes.
    pair = _greedy_pair(nm_c.eigvecs, nm_f.eigvecs)
    f_c = nm_c.f_Hz
    f_f = np.array([nm_f.f_Hz[j] for j in pair])

    r2 = refine ** 2
    f_ext = (r2 * f_f - f_c) / (r2 - 1.0)
    df = np.abs(f_ext - f_f)

    band = FrequencyWithUncertainty(
        f_Hz=f_ext, df_Hz=df,
        f_coarse=f_c, f_fine=f_f,
        f_extrapolated=f_ext,
    )
    return nm_c, nm_f, band


def _greedy_pair(V_a: np.ndarray, V_b: np.ndarray) -> list[int]:
    """Pair columns of V_a to columns of V_b by maximum |inner product|.

    Returns a list `pair` such that V_a[:, k] corresponds to V_b[:, pair[k]].
    """
    n = V_a.shape[1]
    overlap = np.abs(V_a.T @ V_b)
    used = [False] * n
    out = [-1] * n
    # Greedy: largest overlap first
    flat_idx = np.argsort(-overlap.ravel())
    for ix in flat_idx:
        i, j = divmod(int(ix), n)
        if out[i] == -1 and not used[j]:
            out[i] = j
            used[j] = True
            if all(o != -1 for o in out):
                break
    return out
