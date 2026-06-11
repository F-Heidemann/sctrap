"""Small Vinante-like driver for profiling (coarse mesh, full pipeline).

Runs the same stages as quickstart_elliptical.py (mesh -> 5D equilibrium ->
5x5 Hessian/normal modes) but on a deliberately coarse mesh so a cProfile
run finishes in a couple of minutes:

    python -m cProfile -s cumtime scripts/profile_driver.py

Not a benchmark of accuracy -- only of where the time goes.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from sctrap.frequencies import find_equilibrium_5d
from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.modes import normal_modes
from sctrap.particle import SphericalParticle


def main() -> None:
    t_start = time.time()
    particle = SphericalParticle(radius=27e-6, density=7430.0,
                                 B_remanence=0.71)
    solver_kwargs = dict(element="P2", subtract_singularity=True)

    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = Path(tmp) / "cavity.msh"
        t0 = time.time()
        elliptical_cavity(2.0e-3, 2.0e-3, 4.0e-3,
                          mesh_size_sc=0.5e-3,   # coarse on purpose
                          output_file=str(mesh_path), verbose=False)
        sct = load_msh(mesh_path)
        t_mesh = time.time() - t0
        print(f"[mesh]        {t_mesh:6.1f} s   {sct}")

        t0 = time.time()
        eq_r, eq_theta, eq_phi = find_equilibrium_5d(
            sct, particle.moment, np.array([0.0, 0.0, 0.30e-3]),
            theta_guess=np.pi / 2.0, phi_guess=0.0,
            mass=particle.mass, **solver_kwargs)
        t_eq = time.time() - t0
        print(f"[equilibrium] {t_eq:6.1f} s   r_eq(mm)="
              f"({eq_r[0]*1e3:+.4f},{eq_r[1]*1e3:+.4f},{eq_r[2]*1e3:+.4f})"
              f"  theta={eq_theta:+.4f} phi={eq_phi:+.4f}")

        t0 = time.time()
        nm = normal_modes(sct, eq_r, particle.moment,
                          eq_theta=eq_theta, eq_phi=eq_phi,
                          mass=particle.mass, inertia=particle.inertia,
                          h_trans=5e-6, h_ang=5e-3, **solver_kwargs)
        t_h = time.time() - t0
        print(f"[hessian]     {t_h:6.1f} s")
        print(nm)

    print(f"[total]       {time.time()-t_start:6.1f} s")


if __name__ == "__main__":
    main()
