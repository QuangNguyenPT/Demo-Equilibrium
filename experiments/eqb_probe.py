"""EQB-P — what the trained HRM-EQB actually learned from its task field, measured.

ADR-006 scored the model; this experiment asks what the scores are made of. Every probe
turns a word the project uses ("reasoning", "understanding", "generalization") into a
number that can come out wrong:

  P1 linearity     Is the equilibrium core a nonlinear reasoner? With the shipped
                   shrink = 0 the radial ReLU is the identity (spec v0.2 §3 remark), so
                   for fixed routings the fixed point must be *linear* in the injections:
                   z*(2c) = 2 z*(c). Measured on real contexts, then with shrink > 0 to
                   show the probe can fail.
  P2 compute       Does more recurrence help? Accuracy vs segments (1..6, trained with 3)
                   and vs a cap on equilibrium cycles (1, 2, 5, uncapped).
  P3 demos         Is the rule read from the demonstrations? Query and target kept, demos
                   swapped for another family's: accuracy must fall if demos are used.
  P4 novelty       Memorised table or reusable rule? Text families on words outside the
                   90-word training list; sequences longer than trained (7-8 vs 4-6);
                   grids larger than trained (sides 11-14 vs <= 10).
  P5 views         Invariance to superficial form: the same grid task under dihedral x
                   colour relabellings, answers mapped back.

The model directory is copied to a temporary folder first (Mind writes state on close),
so the probe never modifies a trained model.

Run: `python -m experiments.eqb_probe --model ../models/eqb-small`
     (writes `experiments/results/eqb_probe.json`; add --cuda for the capped GPU)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from hrm_eqb.device import select_device
from hrm_eqb.equilibrium import Context
from hrm_eqb.field import IDENTITY, answers_equal, collate, encode_task, views
from hrm_eqb.mind import Mind, _undo
from hrm_eqb.model import HRMEQB
from hrm_eqb.tasks import FAMILIES, Grid, Task, _grid_rule, _sequence_rule, _text_rule, generate
from hrm_eqb.train import predict

SEED = 20260929          # distinct from training (seed 0 + step) and BENCH_SEED (9090)
RESULTS = Path(__file__).parent / "results" / "eqb_probe.json"
TRAIN_MAX_SIDE = 10


def accuracy(model: HRMEQB, tasks: Sequence[Task], device: torch.device, segments: int = None) -> Dict:
    """Exact-match accuracy overall and per family, with a binomial standard error."""
    preds = predict(model, tasks, device, segments=segments)
    return score(preds, tasks)


def score(preds: Sequence[Grid], tasks: Sequence[Task]) -> Dict:
    ok = [answers_equal(p, t.answer) for p, t in zip(preds, tasks)]
    fam: Dict[str, List[bool]] = {}
    for hit, t in zip(ok, tasks):
        fam.setdefault(f"{t.domain}/{t.family}", []).append(hit)
    acc = float(np.mean(ok)) if ok else 0.0
    return {"n": len(ok), "acc": round(acc, 4), "se": round(float(np.sqrt(acc * (1 - acc) / max(len(ok), 1))), 4),
            "by_family": {k: round(float(np.mean(v)), 3) for k, v in sorted(fam.items())}}


def held_out(rng: np.random.Generator, n: int, domains=("grid", "sequence", "text")) -> List[Task]:
    return [generate(d, f, rng, TRAIN_MAX_SIDE) for d in domains for f in FAMILIES[d] for _ in range(n)]


def build(domain: str, family: str, rng: np.random.Generator, sample: Callable[[np.random.Generator], Grid],
          n_demos: int = 3) -> Task:
    """A task of a training family with a replaced input sampler (the rule itself is unchanged)."""
    rule, text, _ = {"text": lambda: _text_rule(family, rng), "sequence": lambda: _sequence_rule(family, rng),
                     "grid": lambda: _grid_rule(family, rng, 14)}[domain]()
    inputs = [sample(rng) for _ in range(n_demos + 1)]
    pairs = [(x, rule(x)) for x in inputs]
    return Task(domain, family, pairs[:-1], pairs[-1][0], pairs[-1][1], text)


# ---------- P1 ----------

@torch.no_grad()
def p1_linearity(model: HRMEQB, tasks: Sequence[Task], device: torch.device) -> Dict:
    """Relative additivity error ||z*(c1 + c2) - z*(c1) - z*(c2)|| / ||z*(c1 + c2)|| of segment-1 equilibria.

    c1 is the real injection of each task and c2 a second task's (same routings, same zero
    start). Doubling c is not used: scaling by 2 is exact in floating point, so it would pass
    for a trivial reason.
    """
    batch = collate([encode_task(t) for t in tasks]).to(device)
    x = model.prelude(batch)
    zero = torch.zeros_like(x)
    ctx = model.context(x, zero, [], batch, 1.0)
    roll = (lambda c: c.roll(1, dims=0) * batch.valid.unsqueeze(-1))
    c1 = Context(ctx.route_low, ctx.route_high, ctx.inject_low, ctx.inject_high)
    c2 = Context(ctx.route_low, ctx.route_high, roll(ctx.inject_low), roll(ctx.inject_high))
    both = Context(ctx.route_low, ctx.route_high, c1.inject_low + c2.inject_low, c1.inject_high + c2.inject_high)
    out = {}
    shipped = model.core.shrink
    for shrink in (shipped, 0.05, 0.5):
        model.core.shrink = shrink
        z = [model.core.solve(zero, zero, c, model.cfg.n_low, 1e-4, 500) for c in (c1, c2, both)]
        err = torch.linalg.vector_norm(z[2].high - z[0].high - z[1].high) / torch.linalg.vector_norm(z[2].high)
        out[f"shrink={shrink}"] = {"rel_additivity_error": float(err), "cycles": max(s.cycles for s in z),
                                   "converged": all(s.converged for s in z)}
    model.core.shrink = shipped
    rate, _ = model.core.certificate(model.cfg.n_low)
    budget_rate, _ = model.cfg.budgets().perron(model.cfg.n_low)
    out["measured_cycle_rate"] = round(rate, 4)
    out["budget_cycle_rate"] = round(budget_rate, 4)
    return out


# ---------- P2 ----------

def p2_compute(model: HRMEQB, tasks: Sequence[Task], device: torch.device) -> Dict:
    """Accuracy, and the fraction of answers identical to the shipped setting's, under less or more compute."""
    shipped = model.cfg
    reference = predict(model, tasks, device)
    same = (lambda preds: round(float(np.mean([answers_equal(a, b) for a, b in zip(preds, reference)])), 4))
    out = {"segments": {}, "max_cycles": {}}
    for s in (1, 2, 3, 6):
        preds = predict(model, tasks, device, segments=s)
        out["segments"][str(s)] = dict(score(preds, tasks), same_answer_as_shipped=same(preds))
    for k in (1, 2, 5, shipped.max_cycles):
        model.cfg = dataclasses.replace(shipped, max_cycles=k)
        preds = predict(model, tasks, device)
        out["max_cycles"][str(k)] = dict(score(preds, tasks), same_answer_as_shipped=same(preds))
    model.cfg = shipped
    return out


# ---------- P3 ----------

def p3_demos(model: HRMEQB, tasks: Sequence[Task], device: torch.device, rng: np.random.Generator) -> Dict:
    """Demos from a different family of the same domain; the query and its target are kept."""
    swapped = []
    for t in tasks:
        others = [f for f in FAMILIES[t.domain] if f != t.family]
        donor = generate(t.domain, others[int(rng.integers(0, len(others)))], rng, TRAIN_MAX_SIDE)
        swapped.append(Task(t.domain, t.family, donor.demos, t.query, t.answer, t.description))
    return {"true_demos": accuracy(model, tasks, device), "swapped_demos": accuracy(model, swapped, device)}


# ---------- P4 ----------

def p4_novelty(model: HRMEQB, device: torch.device, rng: np.random.Generator, n: int) -> Dict:
    text_novel = [generate("text", f, rng, text_source="letters") for f in FAMILIES["text"] for _ in range(n)]
    long_seq = [build("sequence", f, rng,
                      lambda r: r.choice(np.arange(0, 10), int(r.integers(7, 9)), replace=False)
                      .astype(np.int64)[None, :]) for f in FAMILIES["sequence"] for _ in range(n)]
    big_grid = []
    while len(big_grid) < n * len(FAMILIES["grid"]):
        fam = FAMILIES["grid"][len(big_grid) // n]
        t = generate("grid", fam, rng, 14)
        if max(max(g.shape) for pair in t.demos for g in pair) > TRAIN_MAX_SIDE or max(t.query.shape) > TRAIN_MAX_SIDE:
            big_grid.append(t)
    return {"text_in_vocab": accuracy(model, held_out(rng, n, ("text",)), device),
            "text_novel_words": accuracy(model, text_novel, device),
            "sequence_len_4_6": accuracy(model, held_out(rng, n, ("sequence",)), device),
            "sequence_len_7_8": accuracy(model, long_seq, device),
            "grid_side_le_10": accuracy(model, held_out(rng, n, ("grid",)), device),
            "grid_side_11_14": accuracy(model, big_grid, device)}


# ---------- P5 ----------

def p5_views(model: HRMEQB, tasks: Sequence[Task], device: torch.device, rng: np.random.Generator,
             n_views: int = 4) -> Dict:
    grids = [t for t in tasks if t.kind == "grid"]
    per_view = [[] for _ in range(n_views)]
    agree = []
    for t in grids:
        augs = views(t, n_views, rng)
        preds = predict(model, [a.apply(t) for a in augs], device)
        back = [_undo(a, p, t) for a, p in zip(augs, preds)]
        for i, b in enumerate(back):
            per_view[i].append(answers_equal(b, t.answer))
        agree.append(all(answers_equal(b, back[0]) for b in back[1:]))
    assert augs[0] == IDENTITY
    return {"n": len(grids), "acc_identity": round(float(np.mean(per_view[0])), 4),
            "acc_augmented_views": round(float(np.mean([v for pv in per_view[1:] for v in pv])), 4),
            "all_views_agree": round(float(np.mean(agree)), 4),
            "solved_in_every_view": round(float(np.mean([all(pv[i] for pv in per_view)
                                                         for i in range(len(grids))])), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained model directory (copied, never modified)")
    ap.add_argument("--n", type=int, default=30, help="tasks per family")
    ap.add_argument("--cuda", action="store_true", help="use the capped GPU (select_device)")
    args = ap.parse_args()
    device = select_device(args.cuda)
    torch.manual_seed(0)
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "model"
        shutil.copytree(args.model, work)
        mind = Mind(work, device)
        try:
            model = mind.model
            start = time.time()
            base = held_out(rng, args.n)
            print(f"[PROBE] {len(base)} held-out tasks on {device}", flush=True)
            result = {"model": Path(args.model).name, "config": model.cfg.to_dict(), "seed": SEED,
                      "n_per_family": args.n, "device": str(device)}
            sample = [t for t in base if max(t.query.shape) <= 6][:24]
            result["P1_linearity"] = p1_linearity(model, sample, device)
            print(f"[PROBE] P1 {result['P1_linearity']}", flush=True)
            result["P2_compute"] = p2_compute(model, base, device)
            print(f"[PROBE] P2 (acc, same-as-shipped) segments "
                  f"{[(k, v['acc'], v['same_answer_as_shipped']) for k, v in result['P2_compute']['segments'].items()]} "
                  f"cycles {[(k, v['acc'], v['same_answer_as_shipped']) for k, v in result['P2_compute']['max_cycles'].items()]}",
                  flush=True)
            result["P3_demos"] = p3_demos(model, base, device, rng)
            print(f"[PROBE] P3 true {result['P3_demos']['true_demos']['acc']} "
                  f"swapped {result['P3_demos']['swapped_demos']['acc']}", flush=True)
            result["P4_novelty"] = p4_novelty(model, device, rng, args.n)
            print(f"[PROBE] P4 {[(k, v['acc']) for k, v in result['P4_novelty'].items()]}", flush=True)
            result["P5_views"] = p5_views(model, base, device, rng)
            print(f"[PROBE] P5 {result['P5_views']}", flush=True)
            result["seconds"] = round(time.time() - start, 1)
        finally:
            mind.close()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[PROBE] wrote {RESULTS}")


if __name__ == "__main__":
    main()
