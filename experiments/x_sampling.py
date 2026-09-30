"""E15b — reaching features beyond the exhaustive budget: random sampling vs enumeration vs library.

Hypothesis M2 (`vault/10 - Research Program/hypotheses.md`), from the mushroom body: Kenyon cells
are random conjunctions of ~5 inputs (M2 in `experiments/results/connectome_malecns.json`), so a
large feature space can be *covered* by random samples instead of enumerated. The competitor is
M3, library compression: a stored concept is one token, so features built on it are short.

Targets: canonical features of exactly 9 tokens that no expression of <= 7 tokens reaches (the
exhaustive budget of construction depth 3). Each strategy gets the same budget B of *evaluated
expressions*, and scores the fraction of targets whose behaviour class it produced:

  enumerate   bottom-up, smallest first (what construction does), stopping after B expressions
  random      B expressions drawn by random growth of the grammar, size <= 9 (the M2 strategy)
  library     enumeration to 7 tokens with one stored concept as a primitive (M3); B counts its
              whole bank, and only targets expressible through that concept can be reached

Run: `python -m experiments.x_sampling`  (writes `experiments/results/x_sampling.json`)
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict, Optional, Set, Tuple

import numpy as np

from experiments.common import record
from hrm_x import construct as C
from hrm_x.mbi import feature_key

Q = Fraction
BUDGETS = (5_000, 20_000, 80_000)


def _values(expr: Tuple[int, ...], xs, ys, lib=()) -> Optional[np.ndarray]:
    out = []
    for x, y in zip(xs, ys):
        v = C.evaluate(expr, Q(x), Q(y), lib)
        if v is None or abs(v) > C.MAX_VALUE:
            return None
        out.append(float(v))
    return np.array(out)


def random_expr(rng: random.Random, max_size: int) -> Tuple[int, ...]:
    """Grow a random tree: leaf with probability rising with depth; ops chosen uniformly."""
    def grow(budget: int, depth: int) -> Tuple[int, ...]:
        if budget < 3 or rng.random() < 0.25 + 0.15 * depth:
            return (rng.choice(C.LEAVES),)
        if budget >= 5 and rng.random() < 0.2:
            parts, left = [], budget - 1
            for i in range(4):
                sub = grow(max(1, left - (3 - i)), depth + 1)
                parts.append(sub)
                left -= len(sub)
            return sum(parts, ()) + (C.IFGT,)
        a = grow(budget - 2, depth + 1)
        b = grow(budget - 1 - len(a), depth + 1)
        return a + b + (rng.choice(C.BINARY),)
    return grow(max_size, 0)


def main() -> Dict:
    xs = [float(p[0]) for p in C.PROBES]
    ys = [float(p[1]) for p in C.PROBES]
    upto7 = {feature_key(f.values) for f in C.features(7)}
    nine = [f for f in C.features(9) if len(f.expr) == 9 and feature_key(f.values) not in upto7]
    rng = random.Random(0)
    targets = rng.sample(nine, 200)
    target_keys = {feature_key(f.values) for f in targets}
    # Named composites: expressible through max with a short body, 9 tokens without it.
    named = {"max(a, b^2)": lambda a, b: max(a, b * b), "max(a^2, b)": lambda a, b: max(a * a, b),
             "max(a, b) * a": lambda a, b: max(a, b) * a, "max(a+1, b)": lambda a, b: max(a + 1, b)}
    named_keys = {k: feature_key([float(fn(Q(x), Q(y))) for x, y in C.PROBES]) for k, fn in named.items()}

    out: Dict = {"n_targets": len(targets), "n_size9_new_classes": len(nine), "budgets": {}}
    mx = (lambda x, y: max(x, y))
    lib_feats = C.features(7, [((0, 1, 0, 1, C.IFGT), mx)])
    lib_keys = {feature_key(f.values) for f in lib_feats}
    lib_bank_size = sum(len(v) for v in C.enumerate_bank(7, [((0, 1, 0, 1, C.IFGT), mx)]).values())
    for B in BUDGETS:
        enum_bank = C.enumerate_bank(9)
        flat = [e for s in sorted(enum_bank) for e in enum_bank[s]][:B]
        enum_keys = set()
        for _, v in flat:
            k = feature_key(v)
            if k is not None:
                enum_keys.add(k)
        rnd_keys: Set[str] = set()
        rs = random.Random(1)
        for _ in range(B):
            v = _values(random_expr(rs, 9), xs, ys)
            if v is not None:
                k = feature_key(v)
                if k is not None:
                    rnd_keys.add(k)
        row = {"enumerate": len(target_keys & enum_keys) / len(target_keys),
               "random": len(target_keys & rnd_keys) / len(target_keys),
               "distinct_classes": {"enumerate": len(enum_keys), "random": len(rnd_keys)},
               "named": {k: {"enumerate": v in enum_keys, "random": v in rnd_keys, "library": v in lib_keys}
                         for k, v in named_keys.items()}}
        out["budgets"][str(B)] = row
        print(f"[E15b] B={B:6d} targets hit: enumerate {row['enumerate']:.3f} random {row['random']:.3f}  "
              f"classes {row['distinct_classes']}  named {row['named']}", flush=True)
    out["library"] = {"bank_size_to_7_tokens": lib_bank_size, "targets_hit": len(target_keys & lib_keys) / len(target_keys),
                      "named_hit": {k: v in lib_keys for k, v in named_keys.items()}}
    print(f"[E15b] library (max stored), bank {lib_bank_size}: random targets {out['library']['targets_hit']:.3f} "
          f"named {out['library']['named_hit']}")
    return out


if __name__ == "__main__":
    record("x_sampling", {"experiment": "E15b random sampling vs enumeration vs library (hypotheses M2, M3)",
                          "budgets": BUDGETS, "targets": "200 random 9-token classes unreachable at <= 7"}, main())
