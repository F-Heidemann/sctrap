# Example meshes

Both meshes were generated with `sctrap mesh ...` and ship with named
`SC` / `FF` / `air` physical groups, ready for `sctrap simulate`.

## `two_plates.msh`

Two parallel SC plates (15 mm half-side, 0.5 mm thick) separated by 4 mm,
inside a 60 mm far-field sphere.

```bash
sctrap mesh plates --L 0.015 --t 5e-4 --d 4e-3 --R 0.06 \
                   --size-sc 1.5e-3 --size-ff 1.2e-2 \
                   --out two_plates.msh
```

The midplane `U_mag` for an x-oriented default magnet agrees with the
image-dipole analytic result to ~2 % at this resolution
(see `tests/test_two_plates.py`).

## `elliptical_cylinder.msh`

A solid SC elliptical cylinder (a = 10 mm, b = 12 mm semi-axes, height
8 mm), inside a 60 mm far-field sphere.

```bash
sctrap mesh ellipse --a 0.01 --b 0.012 --h 4e-3 --R 0.06 \
                    --size-sc 1.5e-3 --size-ff 1.2e-2 \
                    --out elliptical_cylinder.msh
```

This is the geometry the original FEniCS code targeted. Equilibrium
sits a small distance above the upper face along the z-axis; pass
`--r0 0 0 5e-3` to seed the search there.
