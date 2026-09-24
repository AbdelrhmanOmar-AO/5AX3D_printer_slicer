"""Kinematics round trips in 64-bit floats: run by tests/test_kinematics3z.py.

Not a test module itself (no ``test_`` prefix). It runs in its own process
because Taichi's default float precision is fixed by ``ti.init``, and
re-initialising it inside the test session would reset the runtime under every
other test.

The pipeline runs `kinematics3z` in 32-bit floats. There the build-direction
round trip is only good to about 4.5e-4 rad (docs/plan_corrections.md 7a,
P1-8). This script shows the maths itself is exact: the same kernels in 64-bit
floats. It prints one JSON object of the largest errors.

No `from __future__ import annotations`: this module defines Taichi kernels.
"""

import json
import sys

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64, log_level=ti.ERROR)

import atom.kinematics3z as k  # noqa: E402  (after ti.init: the constants take f64)


@ti.kernel
def solve_inverse(point: ti.types.ndarray(), direction: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(point.shape[0]):
        x, y, z0, z1, z2, offset = k.inverse(
            ti.math.vec3(point[i, 0], point[i, 1], point[i, 2]),
            ti.math.vec3(direction[i, 0], direction[i, 1], direction[i, 2]),
        )
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z0
        out[i, 3] = z1
        out[i, 4] = z2
        out[i, 5] = offset


@ti.kernel
def solve_forward(machine: ti.types.ndarray(), position: ti.types.ndarray(), normal: ti.types.ndarray()):
    for i in range(machine.shape[0]):
        p, n = k.forward(machine[i, 0], machine[i, 1], machine[i, 2], machine[i, 3], machine[i, 4])
        for j in ti.static(range(3)):
            position[i, j] = p[j]
            normal[i, j] = n[j]


def main() -> int:
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    count = 400
    theta = np.radians(rng.uniform(0.0, k.MAX_TILT_ANGLE_DEG - 1.0, count))
    phi = rng.uniform(-np.pi, np.pi, count)
    direction = np.column_stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    point = np.column_stack(
        [rng.uniform(100.0, 200.0, count), rng.uniform(100.0, 190.0, count), rng.uniform(0.0, 60.0, count)]
    )

    solved = np.zeros((count, 6))
    solve_inverse(point, direction, solved)
    valid = np.isfinite(solved[:, 5])

    position = np.zeros((count, 3))
    normal = np.zeros((count, 3))
    solve_forward(np.ascontiguousarray(solved[:, :5]), position, normal)
    back = np.zeros((count, 6))
    solve_inverse(position, normal, back)

    cross = np.linalg.norm(np.cross(normal, direction), axis=1)
    angle = np.arctan2(cross, np.sum(normal * direction, axis=1))
    print(json.dumps({
        "valid": int(valid.sum()),
        "count": count,
        "ik_fk_position_mm": float(np.linalg.norm(position - point, axis=1)[valid].max()),
        "ik_fk_normal_rad": float(angle[valid].max()),
        "fk_ik_screw_mm": float(np.abs(back[:, 2:5] - solved[:, 2:5])[valid].max()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
