"""CLI for the HRM-X v0.6 model.

    python -m hrm_x create  models/hrmx                 # empty model directory (config + empty memory)
    python -m hrm_x train   models/hrmx --tasks 60      # curriculum with a teacher; resumes from the log
    python -m hrm_x chat    models/hrmx "2 ⊕ 3 = 3; 4 ⊕ 1 = 4; 5 ⊕ 5 = 5; 1 ⊕ 6 = 6; 7 ⊕ 2 = ?"
    python -m hrm_x notation models/hrmx --base 7       # learn a numeral notation from written sums
    python -m hrm_x info    models/hrmx                 # config, step, stored concepts
    python -m hrm_x eval    models/hrmx                 # the understanding suite on a copy of the model
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from hrm_x.curriculum import Curriculum
from hrm_x.model import HRMX, ModelConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hrm_x")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create", "train", "chat", "notation", "info", "eval"):
        s = sub.add_parser(name)
        s.add_argument("dir")
        if name == "train":
            s.add_argument("--tasks", type=int, default=None)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--pools", default="written,constructible,composite")
            s.add_argument("--repeats", type=int, default=2)
        if name == "chat":
            s.add_argument("text", nargs="+")
        if name == "notation":
            s.add_argument("--base", type=int, default=10)
            s.add_argument("--sums", type=int, default=150)
    args = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")          # operator symbols are not ASCII
    if args.cmd == "create":
        HRMX(ModelConfig()).save(args.dir)
        print(f"[HRMX] created {args.dir}")
        return 0
    model = HRMX.load(args.dir)
    if args.cmd == "train":
        cur = Curriculum(args.pools.split(","), seed=args.seed, repeats=args.repeats)
        out = model.train(cur, max_tasks=args.tasks, checkpoint=args.dir)
        print(json.dumps({k: v for k, v in out.items()}, indent=2, default=str))
    elif args.cmd == "chat":
        print(model.say(" ".join(args.text)))
        model.save(args.dir)
    elif args.cmd == "notation":
        from hrm_x.reader import make_sums
        ok = model.learn_notation(make_sums(random.Random(0), args.sums, base=args.base, max_int=300))
        print(f"[HRMX] notation base {args.base}: {'learned and verified' if ok else 'rejected (not exact)'}")
        if ok:
            model.save(args.dir)
    elif args.cmd == "info":
        print(json.dumps(model.info(), indent=2))
    elif args.cmd == "eval":
        from experiments.x_understanding import run_suite
        print(json.dumps(run_suite(model), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
