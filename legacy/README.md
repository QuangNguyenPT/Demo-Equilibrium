# `legacy/` — code that is no longer part of the model

Everything here is **out of the live tree**. Nothing in `hrm_x/`, `experiments/` or
`scripts/` imports any of it, and `pyproject.toml` no longer packages it. The
knowledge-graph pipeline already excludes this directory (`.graphifyignore`).

There are two different kinds of thing in here, and they are **not** the same
claim. Do not read a directory listing of this folder as one verdict.

## 1. Never built — the design was decided and never implemented

| path | ADR | what it is |
|---|---|---|
| `cpp_engine/` | ADR-001, ADR-002 | the hybrid C++ engine + Python blueprint: arena allocator, tensor types, kernels, thread pool, KV cache. **Written as C++ and headers; never compiled into the project, never linked by anything.** |
| `configs/`, `python/`, `scripts/` | ADR-001 | the Python-blueprint side of the same design |
| `tests/test_blueprint.py` | ADR-001 | the tests for the above |

The only idea that survived from ADR-001 is the **C-ABI boundary**, re-landed
scoped to the e-graph kernels as `hrm_x/native/include/hrmx.h`.

## 2. Superseded baseline — it was built, measured, and then replaced

| path | ADR | what it is |
|---|---|---|
| `hrm_eqb/` | ADR-005, ADR-006 | HRM-Φ / HRM-EQB: a two-level neural equilibrium solver with a contraction certificate recomputed from the weights at every load, quaternion product-key memory, `.hrmc` checkpoints, an autoresearch keep/discard gate, and a CLI. 23 modules, torch. |
| `tests/test_eqb_*.py` | | 55 tests: 23 theorem checks against spec v0.2, 19 model, 13 runtime |

**It is here because nothing uses it, not because it was wrong.** Measured and
recorded in ADR-006: generated 0.298, language 0.983, schedule invariance 1.000,
**sudoku_extreme 0.000, ARC 0.000**, and a probe that found no arithmetic
(additive-consistency error re-measured at 1.34 against a 1.40 ± 0.015 null). The
diagnosis and the case for re-taking its *method* are in
[[ADR-011 Reintroducing EQB Ideas into HRM-X]].

It was moved on 2026-09-29 after an audit found that **zero** files outside its
own test suite imported it. `hrm/` was audited the same way and **stayed**,
because `scripts/make_benchmark.py` imports `hrm.learning` — the program-search
package is still the source of the frozen benchmark.

## Running the legacy tests

They are out of `testpaths`, so `python -m pytest` does not collect them. Put this
directory on the path first.

The **EQB** tests — the 55 that moved with the package — all pass:

```powershell
$env:PYTHONPATH = "legacy"
python -m pytest legacy/tests/test_eqb_theorems.py legacy/tests/test_eqb_model.py `
                        legacy/tests/test_eqb_runtime.py -q     # 55 passed
```

Running the whole folder instead gives **1 failed, 57 passed**, and that failure is
pre-existing and unrelated to EQB: `legacy/tests/test_blueprint.py::test_native_c_engine_bridge`
needs `legacy/bin/hrm_engine.dll`, the compiled ADR-001 C++ engine — the very thing
ADR-001 never built. It was already outside `testpaths` before `hrm_eqb/` moved here,
so the move neither caused it nor hid it. `test_blueprint.py` covers `cpp_engine/`,
which is not importable as a Python package and is a record rather than a test suite.

## Rules

- **Nothing here may be imported by live code.** If a change needs something from
  here, the honest move is to re-derive it in `hrm_x/`, not to reach across.
- **Do not delete anything here without a decision record.** These are the
  project's own prior designs and the evidence for why the current one exists;
  ADR-003 and ADR-005/006 are unreadable without them.
- **`.graphifyignore` excludes this directory on purpose** ("superseded engine
  kept for reference only; must not appear in impact queries"). Keep it that way —
  these symbols must never turn up in an `affected` query against `hrm_x/`.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
