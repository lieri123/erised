#!/usr/bin/env python3
"""
Import every module under adplatform/ and fail on the first one that raises.

This exists because a cleanup commit deleted `from dataclasses import dataclass,
field` from rtb.py while leaving the @dataclass decorator in place. Every file
still compiled — the error only surfaces at import time — so `compileall` and a
linter pass would both have stayed green. The gateway would not boot.

pytest catches this for modules a test imports. It does not catch it for
db.py, events.py, or train_ctr.py, which nothing imports today. This does.

Walks the filesystem rather than using pkgutil.walk_packages, because
adplatform/ml has no __init__.py and walk_packages skips namespace packages.

    python -m scripts.check_imports
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

PACKAGE = "adplatform"


def module_names() -> list[str]:
    names = []
    for path in sorted(pathlib.Path(PACKAGE).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = ".".join(path.with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        names.append(name)
    return names


def main() -> int:
    failures: list[tuple[str, BaseException]] = []

    for name in module_names():
        try:
            importlib.import_module(name)
        except BaseException as exc:  # SystemExit at import time is also a bug
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print(f"ok    {name}")

    if failures:
        print(f"\n{len(failures)} module(s) failed to import:\n", file=sys.stderr)
        for name, exc in failures:
            print(f"--- {name} ---", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return 1

    print(f"\nall {len(module_names())} modules import cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
