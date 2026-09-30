"""Make `pytest legacy/tests` work without setting PYTHONPATH first.

`hrm_eqb/` lives here since 2026-09-29, so `import hrm_eqb` fails from the repo
root. Without this, the three EQB test files raise *collection* errors rather
than a clean run, which reads like three failures instead of one.

This only puts `legacy/` on `sys.path`; it imports nothing. It is deliberately
not wired into the main suite: `pyproject.toml` sets `testpaths = ["tests"]`, so
`python -m pytest` still collects the live tests only.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
import os
import sys

LEGACY = os.path.dirname(os.path.abspath(__file__))
if LEGACY not in sys.path:
    sys.path.insert(0, LEGACY)
