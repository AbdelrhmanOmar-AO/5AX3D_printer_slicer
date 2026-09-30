"""Run `tools/atomize.py` with every stage stubbed out, and record what it did.

Build plan P5.1a moved `atomize.py`'s stage commands into
`build_stage_commands` and `run_stages`, with the command line to behave
exactly as before. This script is how that is checked without a GPU or
Blender: it replaces `os.system` with a recorder, runs the script, and writes
down every command it would have run, everything it printed and the log file
it wrote. `tests/fixtures/atomize/recorded.json` holds the same record taken
from `atomize.py` *before* the refactor (commit `f5ffcd2`), and
`tests/test_atomize_stages.py` compares the two.

It runs in its own process, never inside pytest: importing `atomize.py` starts
Taichi, which would reset the runtime under every other test.

Usage::

    python tests/_atomize_record.py CASES.json OUT.json WORKDIR

``CASES.json`` is a list of cases, each
``{"name", "params": {...}, "argv": [...], "mode": "main" | "api", "only", "skip", "env"}``.
"main" runs the command line with ``argv`` after the parameter file; "api"
imports the module and calls ``run_stages(params, only=..., skip=...)``.
``env`` sets environment variables for that case alone (``ATOM_MACHINE``, for
a parameter file without ``max_slope``).
"""

import contextlib
import io
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATOMIZE = REPO_ROOT / "tools" / "atomize.py"


def _clean_stdout(text):
    """Drop Taichi's own banner lines, which name the OS and Python version."""
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith("[Taichi]")
    )


def run_case(case, workdir):
    case_dir = Path(workdir) / case["name"]
    (case_dir / "data" / "log").mkdir(parents=True, exist_ok=True)
    param_path = case_dir / "params.json"
    param_path.write_text(json.dumps(case["params"], indent=4), encoding="utf-8")

    commands = []

    def record(command):
        commands.append(command)
        return 0

    captured = io.StringIO()
    error = None
    previous_dir = os.getcwd()
    previous_system = os.system
    previous_argv = sys.argv
    previous_env = {name: os.environ.get(name) for name in case.get("env", {})}
    os.environ.update(case.get("env", {}))
    os.chdir(case_dir)
    os.system = record
    try:
        with contextlib.redirect_stdout(captured):
            if case.get("mode", "main") == "main":
                sys.argv = [str(ATOMIZE), str(param_path), *case.get("argv", [])]
                runpy.run_path(str(ATOMIZE), run_name="__main__")
            else:
                sys.argv = [str(ATOMIZE)]
                module = runpy.run_path(str(ATOMIZE), run_name="atomize_under_test")
                params = module["Parameters"]()
                params.load(str(param_path))
                ran = module["run_stages"](
                    params, only=case.get("only"), skip=case.get("skip")
                )
                print(f"run_stages returned {json.dumps(list(ran))}")
    except SystemExit as exc:
        error = f"SystemExit({exc.code})"
    except Exception as exc:  # noqa: BLE001 - recorded for the test to judge
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    finally:
        os.system = previous_system
        sys.argv = previous_argv
        os.chdir(previous_dir)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    log_path = case_dir / "data" / "log" / f"{case['params']['solid_name']}.log"
    return {
        "commands": commands,
        "stdout": _clean_stdout(captured.getvalue()),
        "log": log_path.read_text(encoding="utf-8") if log_path.is_file() else None,
        "error": error,
    }


def main():
    cases_path, out_path, workdir = sys.argv[1:4]
    sys.path.insert(0, str(REPO_ROOT / "src"))
    # Import once up front, outside the capture, so Taichi's banner goes to
    # the real stdout and never into a record.
    import taichi  # noqa: F401
    import atom.fff3  # noqa: F401

    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    results = {case["name"]: run_case(case, workdir) for case in cases}
    Path(out_path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
