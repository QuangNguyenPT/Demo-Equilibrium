"""Command line for a model bundle.

    python -m hrm create models/main --benchmark data/benchmark/benchmark-v1.json
    python -m hrm ask models/main "1 2 3 -> 3 2 1 ; 4 5 6 -> ?"
    python -m hrm feedback models/main TASK_ID wrong --correction "6 5 4"
    python -m hrm explain models/main TASK_ID
    python -m hrm study models/main --tasks 100 --minutes 480
    python -m hrm sleep | report | promote GEN | rollback GEN | resign   models/main
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from hrm.config import LanguageConfig, ModelConfig, SubstrateConfig
from hrm.language import parse_value
from hrm.learning import StudyBudget
from hrm.metacognition import Message
from hrm.model import Model


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        cfg = ModelConfig(substrate=SubstrateConfig(dim=args.dim),
                          language=LanguageConfig(backend=args.language, model=args.lm or ""))
        with Model.create(args.model, cfg.validate(), benchmark=args.benchmark):
            print(f"[MODEL] created {args.model}")
        return 0
    with Model.open(args.model) as m:
        return _COMMANDS[args.command](m, args)


def _ask(m: Model, args) -> int:
    resp = m.perceive(Message(text=args.text, task_id=args.task_id))
    if resp.question:
        print(f"[MODEL] {resp.question}")
        return 2
    if resp.solution is not None:
        sol = resp.solution
        print(f"[MODEL] intent={resp.intent} status={sol.status} confidence={sol.confidence:.2f} "
              f"task={sol.trace.task_id}")
        if sol.solved:
            domain = m.domains.get(sol.domain)
            print(f"[ANSWER] {domain.render(sol.answer)}")
            print(f"[RULE] {sol.program.describe()}")
        return 0
    print(f"[MODEL] feedback recorded; quarantined={resp.feedback.quarantined}")
    return 0


def _feedback(m: Model, args) -> int:
    correction = parse_value(args.correction) if args.correction else None
    if args.correction and correction is None:
        print("[ERROR] correction must be numbers, grid rows separated by '/'")
        return 2
    result = m.feedback(args.task_id, args.verdict, correction)
    print(f"[FEEDBACK] quarantined={result.quarantined} forgot_prior={result.forgotten_prior}")
    if result.relearned is not None:
        print(f"[FEEDBACK] relearned: {result.relearned.status} "
              f"{result.relearned.program.describe() if result.relearned.solved else ''}")
    return 0


def _explain(m: Model, args) -> int:
    print(f"[EXPLAIN] {m.explain(args.task_id)}")
    return 0


def _sleep(m: Model, args) -> int:
    report = m.sleep()
    print(f"[SLEEP] {report.status} generation={report.generation} "
          f"consolidated={report.consolidated} graduated={report.graduated} "
          f"reasons={report.reasons}")
    return 0


def _study(m: Model, args) -> int:
    report = m.study(StudyBudget(max_tasks=args.tasks, max_seconds=args.minutes * 60.0,
                                 sleep_every=args.sleep_every, seed=args.seed))
    print(f"[STUDY] tasks={report.tasks} correct={report.correct} wrong={report.wrong} "
          f"abstained={report.abstained} cycles={[c.status for c in report.cycles]} "
          f"stopped={report.stopped!r}")
    return 0


def _report(m: Model, args) -> int:
    print(f"[MODEL] generation={m.generation} knowledge={m.knowledge.stats()}")
    for g in m.generations:
        print(f"[GENERATION] {g['generation']} {g['status']} {g.get('note', '')}")
    for row in m.self_report():
        print(f"[COMPETENCE] {row['region']}: {row['status']} competence={row['competence']} "
              f"progress={row['progress']} attempts={row['attempts']}")
    return 0


def _promote(m: Model, args) -> int:
    print(f"[MODEL] promoted {args.generation}; graduated={m.promote(args.generation)}")
    return 0


def _rollback(m: Model, args) -> int:
    back, lost = m.rollback(args.generation)
    print(f"[MODEL] active generation {m.generation}; reactivated={back} lost={lost}")
    return 0


def _resign(m: Model, args) -> int:
    print(f"[HARNESS] re-signed lock {m.resign_harness()[:16]}...")
    return 0


_COMMANDS = {"ask": _ask, "feedback": _feedback, "explain": _explain, "sleep": _sleep,
             "study": _study, "report": _report, "promote": _promote, "rollback": _rollback,
             "resign": _resign}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hrm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="new model bundle")
    c.add_argument("model")
    c.add_argument("--benchmark", help="frozen benchmark JSON (enables sleep and study)")
    c.add_argument("--dim", type=int, default=4096)
    c.add_argument("--language", default="none",
                   choices=["none", "hashing", "transformers", "ollama"])
    c.add_argument("--lm", help="language model id (transformers) or tag (ollama)")
    a = sub.add_parser("ask", help="answer, learn or take feedback from text")
    a.add_argument("model")
    a.add_argument("text")
    a.add_argument("--task-id")
    f = sub.add_parser("feedback", help="judge an earlier answer")
    f.add_argument("model")
    f.add_argument("task_id")
    f.add_argument("verdict", choices=["correct", "wrong"])
    f.add_argument("--correction", help="the right answer, e.g. '6 5 4' or '1 0 / 0 1'")
    e = sub.add_parser("explain", help="why the model answered as it did")
    e.add_argument("model")
    e.add_argument("task_id")
    s = sub.add_parser("study", help="self-directed study of knowledge gaps")
    s.add_argument("model")
    s.add_argument("--tasks", type=int, default=50)
    s.add_argument("--minutes", type=float, default=10.0)
    s.add_argument("--sleep-every", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    for name, text in (("sleep", "one consolidation cycle"), ("report", "competence map"),
                       ("resign", "human: accept the current benchmark as the judge")):
        sub.add_parser(name, help=text).add_argument("model")
    for name, text in (("promote", "human: make a staged candidate live"),
                       ("rollback", "human: return to an earlier generation")):
        g = sub.add_parser(name, help=text)
        g.add_argument("model")
        g.add_argument("generation", type=int)
    return p


if __name__ == "__main__":
    sys.exit(main())
