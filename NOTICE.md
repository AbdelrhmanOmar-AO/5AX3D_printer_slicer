# Provenance

The code under `src/atom/`, `tools/`, `experiment/`, `see_toolpath_with_blender.py`, and `pyproject.toml`
is vendored from **Atomizer** by Xavier Chermain, Giovanni Cocco, Cédric Zanni, Eric Garner,
Pierre-Alexandre Hugron, and Sylvain Lefebvre (Université de Lorraine, CNRS, Inria, LORIA),
published at Computer Graphics Forum / SGP 2025.

- Upstream repository: https://github.com/xavierchermain/atomizer
- License: BSD-3-Clause — see `LICENSE` (the original notice, kept verbatim as the license requires)
- Paper: https://doi.org/10.1111/cgf.70189

We vendor it directly (rather than using it as an external dependency) because our work modifies its
internals (see the implementation plan doc for exactly which files and functions). Everything under
`atom/` at the repository root, plus anything in `tools/` not present upstream, is new code written for
this project.

The triple-Z-axis bed-tilt kinematics referenced by this project's mechanical design draws on
"Towards Accessible Non-Planar FFF Using Triple Z-Axis Kinematics" (Cocco et al., SCF 2025); notably,
Atomizer's own `src/atom/kinematics3z.py` already implements closed-form kinematics for that class of
machine, which is why Phase 0/1 of the implementation plan is about adapting constants rather than
writing kinematics from scratch.
