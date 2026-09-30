"""E16 — does a learned index (FHRR signature -> mushroom-body code -> readouts) order features well?

Hypothesis M4 (`vault/10 - Research Program/hypotheses.md`, doc 14 N1). The index is trained by
softmax regression on *dreamed* tasks -- random affine images a g + b x + c y + d of 40 feature
classes at random points -- and tested on fresh dreams of the same classes (new points, new
coefficients). Reported against chance among the trained classes, not against the MDL rank over
all features (which would flatter it: untrained classes are ranked last by construction).

Also reported: the cost of one routing against the cost of simply prefiltering every feature of
the level on the evidence -- the comparison that decides whether ordering can ever pay here.

Ablations: kWTA 2 % (the APL rule), kWTA 10 %, dense ReLU (no inhibition).

Run: `python -m experiments.x_router`  (writes `experiments/results/x_router.json`)
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict

import numpy as np

from experiments.common import record
from hrm_x.construct import evaluate, features
from hrm_x.mbi import MushroomBodyIndex, feature_key

Q = Fraction


def dream(f, r: random.Random, n: int):
    a = Q(r.choice([-3, -2, -1, 1, 2, 3]))
    b, c, d = (Q(r.randint(-3, 3)) for _ in range(3))
    ev = []
    while len(ev) < n:
        x, y = Q(r.randint(1, 9)), Q(r.randint(1, 9))
        g = evaluate(f.expr, x, y)
        if g is not None:
            ev.append(((x, y), a * g + b * x + c * y + d))
    return ev


def main() -> Dict:
    feats = features(7)
    rng = random.Random(0)
    classes = rng.sample(feats, 40)
    out: Dict = {"classes": len(classes), "level_features": len(feats)}
    for name, kw in [("kWTA 2% (APL)", {}), ("kWTA 10%", {"frac": 0.1}), ("dense ReLU", {"kwta": False})]:
        m = MushroomBodyIndex(**kw)
        train = [(m.code(dream(f, rng, rng.randint(4, 6))), feature_key(f.values)) for f in classes for _ in range(30)]
        acc = m.train(train, epochs=15)
        ranks = []
        for f in classes:
            k = feature_key(f.values)
            for _ in range(10):
                sc = m.scores(m.code(dream(f, rng, rng.randint(4, 6))))
                ranks.append(sorted(sc, key=lambda c: -sc[c]).index(k))
        r = np.array(ranks)
        out[name] = {"train_acc": round(acc, 3), "held_out_top1": round(float(np.mean(r == 0)), 3),
                     "held_out_top5": round(float(np.mean(r < 5)), 3), "median_rank": float(np.median(r)),
                     "chance_top1": round(1 / len(classes), 3), "chance_median_rank": (len(classes) - 1) / 2,
                     "routing_ops": m.cost()}
        print(f"[E16] {name:14s} {out[name]}", flush=True)
    out["prefilter_ops_whole_level_5_rows"] = int(sum(5 * len(f.expr) for f in feats))
    print(f"[E16] prefiltering every feature of the level on 5 rows costs {out['prefilter_ops_whole_level_5_rows']} ops")
    return out


if __name__ == "__main__":
    record("x_router", {"experiment": "E16 learned mushroom-body index over FHRR signatures (hypothesis M4)",
                        "train": "30 dreams x 40 classes", "test": "10 fresh dreams x 40 classes"}, main())
