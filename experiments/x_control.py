"""E9 adaptive computation, E10 verification effectiveness, and the hierarchy (H vs L) test.

Hypotheses:
  E9  the value-of-computation controller spends little on easy tasks and more on hard ones,
      matching the accuracy of a fixed full pipeline at a fraction of its cost, and beating a
      greedy "derive the best guess and answer" controller on hard tasks;
  E10 the independent verifier turns injected reasoning faults into abstentions instead of wrong
      answers; it does NOT catch perception (parse) faults, because it replays the same reading;
  H   separating deliberation (H: few value-based decisions) from inference (L: many rule
      firings to a fixpoint) is what keeps decision cost small: a flat controller would decide
      at every firing.
Falsified if: easy and hard tasks cost the same; the fixed pipeline is not more expensive at
equal accuracy; removing the verifier does not raise wrong answers under reasoning faults.
Truth comes from independent generators/oracles (experiments.x_story / x_grid, Fraction ops).
  python -m experiments.x_control
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

from experiments.common import record, table
from experiments.x_grid import COMPOSED, SINGLE, correct as grid_correct, make_task
from experiments.x_story import TRAIN_ITEMS, TRAIN_NAMES, story
from hrm_x.domains.arith import Arith, ArithTask
from hrm_x.domains.grid import Grids
from hrm_x.domains.story import Parser, Story, StoryTask
from hrm_x.induction import Law
from hrm_x.mind import Controller, Mind, Result

SEED = 0
CONCEPTS = {"×": lambda a, b: a * b, "⊕": lambda a, b: a + b, "⊗": lambda a, b: 2 * a + b}


def text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{float(v):.1f}"


def arith_tasks(rng: random.Random, n: int, hard: bool):
    out = []
    for _ in range(n):
        sym = rng.choice(list(CONCEPTS))
        f = CONCEPTS[sym]
        a, b = Fraction(rng.randint(2, 40)), Fraction(f"{rng.uniform(0.5, 30):.1f}")
        if hard:
            ex = [(Fraction(2), Fraction(2)), (Fraction(2), Fraction(3))]
            examples = [f"{text(x)} {sym} {text(y)} = {text(f(x, y))}" for x, y in ex]
            out.append(("arith", ArithTask.from_text(examples, f"{text(a)} {sym} {text(b)}", sym, oracle=f), f(a, b), sym))
        else:
            out.append(("arith", ArithTask.from_text([], f"{text(a)} {sym} {text(b)}", sym), f(a, b), sym))
    return out


def learned_arith_mind() -> Mind:
    mind, dom = Mind(), Arith()
    for sym, f in CONCEPTS.items():
        mind.solve(dom, ArithTask.from_text([f"2 {sym} 3 = {text(f(Fraction(2), Fraction(3)))}",
                                             f"3 {sym} 5 = {text(f(Fraction(3), Fraction(5)))}"],
                                            f"4 {sym} 2", sym, oracle=f))
    return mind


def greedy(domain, task, mind: Mind) -> Result:
    """L-only: propose once, derive the most probable hypothesis, answer it."""
    got = domain.propose(task, 0, mind)
    hyps, ops = got if got else ([], 0)
    alive = [h for h in hyps if h.weight > 0]
    if not alive:
        return Result("abstain", None, 0.0, "no hypothesis", [], ops, hyps)
    best = max(alive, key=lambda h: h.weight)
    if not best.derived:
        ops += domain.derive(task, best, mind)
    status = "answer" if best.answer is not None else "abstain"
    return Result(status, best.answer, best.weight, "greedy", [], ops, hyps)


def fixed(domain, task, mind: Mind) -> Result:
    """Everything, always: all proposal levels, derive and verify every hypothesis, two questions."""
    hyps, ops = [], 0
    level = 0
    while True:
        got = domain.propose(task, level, mind)
        if got is None:
            break
        hyps += got[0]
        ops += got[1]
        level += 1
    for _ in range(2):
        alive = [h for h in hyps if h.weight > 0]
        qs = domain.questions(task, alive, mind) if len(alive) > 1 else []
        if qs:
            ops += domain.ask(task, qs[0][0], alive, mind)
    for h in hyps:
        if h.weight > 0 and not h.derived:
            ops += domain.derive(task, h, mind)
        if h.weight > 0 and h.answer is not None:
            ops += domain.verify(task, h, mind)
    ok = [h for h in hyps if h.weight > 0 and h.verified]
    if not ok:
        return Result("abstain", None, 0.0, "nothing verified", [], ops, hyps)
    best = max(ok, key=lambda h: h.weight)
    return Result("answer", best.answer, best.weight, "fixed", [], ops, hyps)


def main() -> None:
    rng = random.Random(SEED)
    parser = Parser()
    parser.train(seed=SEED)
    story_dom = Story(parser)
    suites: Dict[str, List[Tuple]] = {
        "easy": [], "hard": [],
    }
    suites["easy"] += arith_tasks(rng, 15, hard=False)
    suites["hard"] += arith_tasks(rng, 15, hard=True)
    for _ in range(15):
        c = story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, 1, 1, 20)
        suites["easy"].append(("story", StoryTask(c.sentences, c.question), c.truth, None))
    bad = [c for c in (story(rng, TRAIN_NAMES, TRAIN_ITEMS, 3, 5, 1, 6, allow_bad=True) for _ in range(300))]
    suites["hard"] += [("story", StoryTask(c.sentences, c.question), c.truth, None) for c in bad[:15]]
    for spec, novel, key in [((n,), False, "easy") for n in rng.sample(list(SINGLE), 8) for _ in range(2)] + \
                            [(c, True, "hard") for c in COMPOSED for _ in range(3)]:
        t = make_task(rng, spec, novel)
        if t is not None:
            suites[key].append(("grid", t[0], t[1], None))

    arith_mind = learned_arith_mind()

    def run(policy: str, kind: str, task, truth) -> Tuple[bool, bool, int, int]:
        if kind == "arith":
            dom = Arith()
            mind = Mind() if task.examples else _clone(arith_mind)
        elif kind == "story":
            dom, mind = story_dom, Mind()
        else:
            dom, mind = Grids(), Mind()
        if policy == "voc":
            r = mind.solve(dom, task)
        elif policy == "no-verifier":
            mind.control = Controller(use_verifier=False)
            r = mind.solve(dom, task)
        elif policy == "no-questions":
            mind.control = Controller(use_questions=False)
            r = mind.solve(dom, task)
        elif policy == "greedy":
            r = greedy(dom, task, mind)
        else:
            r = fixed(dom, task, mind)
        if kind == "grid":
            right = grid_correct(r, truth)
        else:
            right = (r.status == "abstain") if truth is None else (r.status == "answer" and r.answer == truth)
        wrong = r.status == "answer" and not right
        return right, wrong, r.ops, max(r.cycles, 1)

    policies = ["voc", "greedy", "fixed", "no-verifier", "no-questions"]
    rows, metrics = [], {}
    for policy in policies:
        for level, tasks in suites.items():
            res = [run(policy, *t[:3]) for t in tasks]
            m = {"n": len(res), "accuracy": float(np.mean([r[0] for r in res])),
                 "wrong_answers": float(np.mean([r[1] for r in res])),
                 "mean_ops": float(np.mean([r[2] for r in res])), "mean_cycles": float(np.mean([r[3] for r in res]))}
            metrics[f"{policy}/{level}"] = m
            rows.append([policy, level, m["n"], f"{m['accuracy']:.3f}", f"{m['wrong_answers']:.3f}",
                         f"{m['mean_ops']:.0f}", f"{m['mean_cycles']:.1f}"])
    print(table(rows, ["policy", "tasks", "n", "accuracy", "wrong", "ops", "H cycles"]))

    # Hierarchy: decisions a flat controller would need (one per L-level firing) vs H cycles
    voc_ops = [metrics[f"voc/{k}"]["mean_ops"] for k in suites]
    voc_cyc = [metrics[f"voc/{k}"]["mean_cycles"] for k in suites]
    metrics["H vs flat decisions per task"] = {k: {"H": c, "flat": c + o} for k, c, o in zip(suites, voc_cyc, voc_ops)}
    print("[H] decisions per task: " + ", ".join(f"{k}: H {c:.1f} vs flat {c + o:.0f}" for k, c, o in zip(suites, voc_cyc, voc_ops)))

    # E10: injected reasoning faults (a false law) and perception faults (swapped roles)
    fault = {}
    for use_verifier in (True, False):
        mind = _clone(arith_mind)
        dom = Arith()
        from hrm_x.domains.arith import show
        key = f"concept:{show(dom.best(mind, '×'))}"
        mind.knowledge["arith"]["laws"][("×", key)] = [Law(("f", "?x", "?z"), ("add", "?x", "?z"), 0)]   # fires on every task
        mind.control = Controller(use_verifier=use_verifier)
        wrong = 0
        for t in arith_tasks(random.Random(3), 30, hard=False):
            if t[3] != "×":
                continue
            r = mind.solve(dom, t[1])
            wrong += r.status == "answer" and r.answer != t[2]
        fault[f"false law, verifier={'on' if use_verifier else 'off'}"] = wrong
    import hrm_x.domains.story as S
    original = S.frames_for

    def swapped(parser_, sentence, keep=0.02):
        frames = original(parser_, sentence, keep)
        out = []
        for f, p in frames:
            if f.schema == "TRANSFER":
                f = S.Frame(f.schema, f.owner, f.dst, f.src, f.amount, f.item)
            out.append((f, p))
        return out
    for use_verifier in (True, False):
        S.frames_for = swapped
        wrong = 0
        stories = [story(random.Random(k), TRAIN_NAMES, TRAIN_ITEMS, 2, 2, 5, 20) for k in range(40)]
        for c in stories:
            mind = Mind(control=Controller(use_verifier=use_verifier))
            r = mind.solve(story_dom, StoryTask(c.sentences, c.question))
            wrong += r.status == "answer" and r.answer != c.truth
        S.frames_for = original
        fault[f"swapped transfer roles (perception), verifier={'on' if use_verifier else 'off'}"] = wrong
    metrics["E10 wrong answers under injected faults"] = fault
    for k, v in fault.items():
        print(f"[E10] {k}: {v} wrong answers")
    record("x_control", {"seed": SEED, "lambda": Controller().lam, "wrong_penalty": Controller().wrong_penalty,
                         "concepts": list(CONCEPTS), "policies": policies}, metrics)


def _clone(mind: Mind) -> Mind:
    import copy
    return copy.deepcopy(mind)


if __name__ == "__main__":
    main()
