"""EQB-T — can HRM-EQB learn a text *rule* when the training words cannot be memorised?

EQB-P (`experiments/eqb_probe.py`) measured the shipped model at 0.956 on the text families
over the 90 training words and ~0.01 on unseen letter strings: the language score of ADR-006
is a recalled table, not a rule. That leaves two explanations, and this experiment separates
them by training fresh models on the text domain only, same budget, same seed:

  words      the ADR-006 distribution (inputs from the 90-word list)
  letters    random lowercase strings, so no (word, rule) pair repeats and only a rule can score
  letters+σ  as `letters`, with shrink = 0.1 so the equilibrium core is no longer linear
             (EQB-P P1: with the shipped shrink = 0 the core's fixed point is linear in its input)

Each arm is scored on training-distribution words, unseen strings of the trained lengths
(3-5), and longer unseen strings (6-8) -- the last one a length-generalisation probe no arm
was trained on. If `letters` is still near zero on unseen strings, the architecture cannot
learn these rules at this budget, and memorisation was not merely a data artefact.

GPU work goes through `select_device()` (70 % VRAM cap) with `--pause` duty cycling.

Run: `python -m experiments.eqb_text --minutes 12`  (writes `experiments/results/eqb_text.json`)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from hrm_eqb.config import PRESETS
from hrm_eqb.device import select_device
from hrm_eqb.model import HRMEQB
from hrm_eqb.tasks import TEXT_FAMILIES, Task, _as_bytes, _text_rule, generate, random_letters
from hrm_eqb.train import TrainConfig, evaluate, train

RESULTS = Path(__file__).parent / "results" / "eqb_text.json"
ARMS = {"words": ("words", 0.0), "letters": ("letters", 0.0), "letters+shrink0.1": ("letters", 0.1)}
EVAL_SEED = 777


def long_letters(rng: np.random.Generator, family: str) -> Task:
    rule, text, _ = _text_rule(family, rng, "letters")
    inputs = [_as_bytes(random_letters(rng, (6, 8))) for _ in range(4)]
    pairs = [(x, rule(x)) for x in inputs]
    return Task("text", family, pairs[:-1], pairs[-1][0], pairs[-1][1], text)


def suites(n: int) -> Dict[str, List[Task]]:
    rng = np.random.default_rng(EVAL_SEED)
    return {"train_words": [generate("text", f, rng) for f in TEXT_FAMILIES for _ in range(n)],
            "unseen_letters_3_5": [generate("text", f, rng, text_source="letters")
                                   for f in TEXT_FAMILIES for _ in range(n)],
            "unseen_letters_6_8": [long_letters(rng, f) for f in TEXT_FAMILIES for _ in range(n)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0, help="training budget per arm")
    ap.add_argument("--n", type=int, default=50, help="eval tasks per family per suite")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()
    device = select_device()
    evals = suites(args.n)
    out: Dict[str, Dict] = {}
    for arm in args.arms.split(","):
        source, shrink = ARMS[arm]
        torch.manual_seed(0)
        cfg = dataclasses.replace(PRESETS["small"], shrink=shrink)
        model = HRMEQB(cfg)
        tcfg = TrainConfig(steps=10**9, minutes=args.minutes, only=tuple(f"text/{f}" for f in TEXT_FAMILIES),
                           text_source=source, pause=args.pause, coda_weight=0.0, log_every=250, seed=0)
        start = time.time()
        log: List[str] = []
        ema = train(model, tcfg, device, log=lambda s: (log.append(s), print(f"[{arm}] {s}", flush=True)))
        row = {"text_source": source, "shrink": shrink, "train_seconds": round(time.time() - start),
               "last_log": log[-2:]}
        for name, tasks in evals.items():
            ev = evaluate(ema, tasks, device)
            row[name] = {"acc": round(ev.accuracy, 4), "by_family": {k: round(v, 3) for k, v in ev.by_family().items()}}
            print(f"[{arm}] {name}: {ev.accuracy:.3f}", flush=True)
        out[arm] = row
        del model, ema
        if device.type == "cuda":
            torch.cuda.empty_cache()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"minutes_per_arm": args.minutes, "n_per_family": args.n, "seed": 0,
                                   "eval_seed": EVAL_SEED, "preset": "small", "arms": out}, indent=2),
                       encoding="utf-8")
    print(f"[EQB-T] wrote {RESULTS}")


if __name__ == "__main__":
    main()
