"""Build a frozen benchmark (run once, keep the output).

    python scripts/make_benchmark.py [--out data/benchmark/benchmark-v1.json]
    python scripts/make_benchmark.py --version 2

A model created with a benchmark copies it into its bundle and locks it. Editing
a benchmark later breaks that lock on purpose: the judge must never change
silently, so this script refuses to overwrite an existing file. Make a new
version (v2) instead and let a human re-sign.

**v1** covers grid + sequence, 15 families, ~300 candidates per task.

**v2** adds the two axes v1 could not measure:
  * depth  — the vector3d d3/d4/d5 families need 3, 4 and 5 composed steps.
    `d4_rotate` and `d5_rotate` are past `reasoning.synthesis_depth`, so bounded
    beam search cannot reach them by composition. A solve rate that holds as depth
    rises is evidence the fast path is doing work, not that the search is
    exhaustive.
  * size   — the vector3d vocabulary (axis x angle, plus reflect/scale/project)
    is ~43 candidates against a search that verifies `fast_top_k`, so the phasor
    ranking has to discriminate rather than confirm a short list.
Plus per-task `depth` and per-depth scoring in the scorer, and a solvability
check so an out-of-vocabulary family (gravity) is not scored as a failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrm.learning import GridTasks, Harness, SequenceTasks, Vector3DTasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen benchmark.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--version", type=int, default=1, choices=(1, 2))
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--decoys", type=int, default=3, help="decoy tasks per domain")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    out = Path(args.out or f"data/benchmark/benchmark-v{args.version}.json")
    if out.exists():
        raise SystemExit(f"[BENCHMARK] {out} exists; a frozen benchmark is never overwritten")
    if args.version == 1:
        generators = [GridTasks(), SequenceTasks()]
    else:
        generators = [GridTasks(), SequenceTasks(), Vector3DTasks()]
    path = Harness.build(out, generators, args.per_family, args.decoys, args.seed)
    harness = Harness(path)
    bands: dict = {}
    for t in harness.tasks:
        bands[t.depth] = bands.get(t.depth, 0) + 1
    print(f"[BENCHMARK] wrote {path} ({len(harness.tasks)} tasks)")
    print(f"[BENCHMARK] tasks per depth band: {dict(sorted(bands.items()))}")


if __name__ == "__main__":
    main()
