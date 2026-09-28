"""Whether `order_atoms` runs with Taichi's kernel profiler (build plan task P1.6).

Upstream starts the ordering stage with ``ti.init(..., kernel_profiler=True)``
and prints the profiler's table at the end. `order_atoms` is 86 % of the
pipeline's runtime, and a profiler times every kernel launch (tens of thousands
per run), so it is not free. It changes no result, only the timing, so it is
now **off by default** and switched on with an environment variable:

    $env:ATOM_TI_PROFILER = "1"       # PowerShell: profile, and print the table
    Remove-Item Env:\\ATOM_TI_PROFILER  # back to the default: off

Accepted values, case-insensitive: ``1 true yes on`` for on, ``0 false no off``
or unset/empty for off. Anything else is an error rather than a silent default,
as for ``ATOM_TI_ARCH`` in `atom.ti_env`: a typo that silently left the
profiler on (or off) would distort exactly the timings this switch exists for.

No Taichi import here, so the switch can be tested without a Taichi runtime.
"""

from __future__ import annotations

import os

#: The environment variable that switches the kernel profiler on.
ENV_VAR = "ATOM_TI_PROFILER"

_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"", "0", "false", "no", "off"})


def profiler_enabled() -> bool:
    """True when ``ATOM_TI_PROFILER`` asks for the kernel profiler."""
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(
        f"{ENV_VAR}={os.environ.get(ENV_VAR)!r} is not understood. "
        f"Use one of {sorted(_ON)} to switch the profiler on, or "
        f"{sorted(_OFF - {''})} (or leave it unset) for off."
    )
