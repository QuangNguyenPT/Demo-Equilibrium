"""CLI: python -m hrm_eqb create | info | ask | learn | study | bench | seal."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from hrm_eqb import bench, hrmc
from hrm_eqb import knowledge as kn
from hrm_eqb.config import PRESETS
from hrm_eqb.device import select_device
from hrm_eqb.mind import CORE, KNOWLEDGE, STATE, Answer, Mind, as_array, task_from_json
from hrm_eqb.model import HRMEQB
from hrm_eqb.study import study_once
from hrm_eqb.tasks import fits, load_arc
from hrm_eqb.train import TrainConfig, train

ARC_TRAIN_PAIRS = 2_000_000


def _grid_json(a: np.ndarray, domain: str):
    if domain == "text":
        return bytes(int(v) for v in a.ravel() if 0 <= v < 256).decode("utf-8", errors="replace")
    return a.tolist()


def cmd_create(args) -> int:
    device = select_device(not args.cpu)
    cfg = PRESETS[args.preset]
    model = HRMEQB(cfg)
    extra = []
    if args.arc_train:
        extra = [t for t in load_arc(Path(args.arc_train))
                 if t.answer is not None and fits(t, int(ARC_TRAIN_PAIRS ** 0.5))]
        print(f"[CREATE] {len(extra)} ARC training tasks fit the attention budget")
    checkpoint = args.checkpoint or str(Path("runs") / f"checkpoint-train-{Path(args.dir).name}.pt")
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    tcfg = TrainConfig(steps=args.steps, minutes=args.minutes, seed=args.seed,
                       extra_weight=args.arc_weight if extra else 0.0, log_every=args.log_every,
                       checkpoint=checkpoint, pause=args.pause)
    ema = train(model, tcfg, device, extra)
    sha = Mind.create(Path(args.dir), ema.to("cpu"), {
        "generation": 0, "parent": None, "note": "created",
        "training": {"preset": args.preset, "steps": args.steps, "minutes": args.minutes, "seed": args.seed,
                     "arc_tasks": len(extra)}})
    print(f"[CREATE] wrote {args.dir} (core sha256 {sha[:16]})")
    return 0


def cmd_info(args) -> int:
    header, _, _ = hrmc.read_header(Path(args.dir) / CORE)
    state = json.loads((Path(args.dir) / STATE).read_text(encoding="utf-8"))
    cert = header["certificate"]
    print(json.dumps({"config": header["config"], "certificate": cert, "lineage": header["lineage"],
                      "knowledge": header["knowledge"], "state": state}, indent=2))
    return 0


def _load_task(path: str):
    return task_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_ask(args) -> int:
    mind = Mind(Path(args.dir), select_device(not args.cpu))
    try:
        task = _load_task(args.task)
        result = mind.ask(task, n_views=args.views)
        out = {"kind": result.kind, "confidence": round(result.confidence, 4), "description": result.description}
        if isinstance(result, Answer):
            out.update({"answer": _grid_json(result.grid, task.domain), "votes": f"{result.votes}/{result.views}",
                        "certified_error": result.certified_error, "segments": result.segments})
        else:
            out.update({"question": "what is the output for this input?",
                        "input": _grid_json(result.query, task.domain), "bits": round(result.bits, 4),
                        "candidates": [{"answer": _grid_json(a, task.domain), "votes": c} for a, c in result.candidates],
                        "reasons": result.reasons})
        print(json.dumps(out, indent=2))
    finally:
        mind.close()
    return 0


def cmd_learn(args) -> int:
    mind = Mind(Path(args.dir), select_device(not args.cpu))
    try:
        task = _load_task(args.task)
        answer = as_array(json.loads(Path(args.answer).read_text(encoding="utf-8"))["output"])
        report = mind.learn(task, answer, steps=args.steps)
        print(json.dumps(report.__dict__, indent=2))
    finally:
        mind.close()
    return 0


def cmd_study(args) -> int:
    device = select_device(not args.cpu)
    for _ in range(args.rounds):
        study_once(Path(args.dir), args.minutes, args.per_family, args.delta, device)
    return 0


def cmd_bench(args) -> int:
    device = select_device(not args.cpu)
    mind = Mind(Path(args.dir), device)
    model = mind.model
    results = []
    suites = args.suites.split(",")
    try:
        for suite in suites:
            if suite == "generated":
                results.append(bench.generated(model, device, args.per_family))
            elif suite == "language":
                results.append(bench.language(model, device, args.per_family))
            elif suite == "sudoku":
                results.append(bench.sudoku(model, device, Path(args.sudoku_csv) if args.sudoku_csv else None,
                                            args.limit))
            elif suite == "arc":
                results.append(bench.arc(model, device, Path(args.arc_dir), args.limit))
            elif suite == "empty_brain":
                results.append(bench.empty_brain(model, device, args.per_family))
            elif suite == "schedule":
                results.append(bench.schedule_invariance(model, device, max(1, args.per_family // 5)))
            else:
                raise ValueError(f"unknown suite '{suite}'")
            print(f"[BENCH] {results[-1].line()}", flush=True)
    finally:
        mind.close()
    bench.report(results, Path(args.dir) / "bench.json")
    return 0


def cmd_seal(args) -> int:
    kn.seal(Path(args.dir) / KNOWLEDGE)
    print(f"[SEAL] {Path(args.dir) / KNOWLEDGE} rewritten with valid checksums")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hrm_eqb", description="HRM-EQB: certified equilibrium reasoning")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_text: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_text)
        s.add_argument("dir", help="model directory (model.hrmc + knowledge.pt + state.json)")
        s.add_argument("--cpu", action="store_true", help="never use CUDA")
        s.set_defaults(fn=fn)
        return s

    s = add("create", cmd_create, "train a new model and write its two files")
    s.add_argument("--preset", choices=sorted(PRESETS), default="small")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--minutes", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--arc-train", default=None, help="ARC-AGI training directory to mix in")
    s.add_argument("--arc-weight", type=float, default=0.15)
    s.add_argument("--log-every", type=int, default=100)
    s.add_argument("--checkpoint", default=None, help="rolling checkpoint (resumed if present)")
    s.add_argument("--pause", type=float, default=0.0, help="seconds idle per step (GPU thermal headroom)")
    add("info", cmd_info, "print config, certificate, lineage and state")
    s = add("ask", cmd_ask, "answer a task JSON, or ask a question about it")
    s.add_argument("task")
    s.add_argument("--views", type=int, default=4)
    s = add("learn", cmd_learn, "teach the answer to a task (verified knowledge write)")
    s.add_argument("task")
    s.add_argument("answer", help='JSON file {"output": ...}')
    s.add_argument("--steps", type=int, default=30)
    s = add("study", cmd_study, "self-study rounds with the keep/discard gate")
    s.add_argument("--minutes", type=float, default=10.0)
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("--per-family", type=int, default=20)
    s.add_argument("--delta", type=float, default=0.05)
    s = add("bench", cmd_bench, "run benchmark suites")
    s.add_argument("--suites", default="generated,language,sudoku,empty_brain,schedule")
    s.add_argument("--per-family", type=int, default=50)
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--arc-dir", default="data/external/arc/evaluation")
    s.add_argument("--sudoku-csv", default="data/external/sudoku-extreme/test.csv")
    add("seal", cmd_seal, "rewrite knowledge.pt so its zip checksums are valid")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(0)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
