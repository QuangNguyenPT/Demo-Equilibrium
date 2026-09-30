"""E19 — the understanding suite: operational properties U1-U10, with causal ablations.

`vault/10 - Research Program/understanding_definition.md` defines each property as a test that
can fail. This runner trains one model on the curriculum (`hrm_x/curriculum.py`, seed 0), then
evaluates it and its ablations on **held-out** tasks (seed 1: new evidence points, new queries
outside the evidence range, and operators that never appeared in training). Memory is restored
from the trained snapshot before every task, so no evaluation task can teach another: what is
measured is transfer from training, not online learning during the test.

Categories (correct / wrong / abstain, asks, ops, concepts constructed, concepts reused):
  written        in the 68 families (control: must not construct, must stay cheap)          U5
  constructible  outside the families, trained operators on new evidence                     U8
  novel          outside the families and never trained: (a-b)^2, b^2+a, |a-b|+1, a/(b+1)    U8 U4
  composite      needs a stored concept to be reachable within budget                        U2 U7
  outside        outside the primitives (gcd, a mod b, digit sum): must be refused           U9
  surface        `constructible` tasks re-presented through six templates and new symbols    U1
  revision       evidence that first fits max, then contradicts it                           U6

Conditions: trained model; fresh model (no training); ablations of construction, memory, library
tokens, BOTTOM (closed world), the teacher, the question horizon; a nearest-neighbour and a
least-squares pattern matcher as non-reasoning baselines.

Run: `python -m experiments.x_understanding`  (writes `experiments/results/x_understanding.json`)
"""
from __future__ import annotations

import copy
import math
import random
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.common import record
from hrm_x.curriculum import POOLS, Curriculum, make_task
from hrm_x.domains.concept import ConceptTask
from hrm_x.language import Conversation
from hrm_x.mind import Controller, Mind
from hrm_x.model import HRMX, ModelConfig

Q = Fraction

NOVEL = {
    "(a-b)^2": lambda a, b: (a - b) ** 2, "b^2+a": lambda a, b: b * b + a,
    "|a-b|+1": lambda a, b: abs(a - b) + 1, "a/(b+1)": lambda a, b: a / (b + 1),
    "3min-a": lambda a, b: 3 * min(a, b) - a,
}
OUTSIDE = dict(POOLS["outside"], **{"digitsum": lambda a, b: Q(sum(int(c) for c in str(int(a) + int(b))))})
COMPOSITE_NOVEL = {"max(b,a^2)": lambda a, b: max(b, a * a), "max*a": lambda a, b: max(a, b) * a}
TEMPLATES = ["{a} {s} {b} = {c}", "{c} = {a} {s} {b}", "{s}({a}, {b}) = {c}", "{a} {s} {b} gives {c}",
             "combining {a} and {b} with {s} gives {c}", "if you {s} {a} and {b}, you get {c}"]
SYMBOLS = ["⊕", "★", "∆", "⊞", "♦", "∘"]


def held_out(seed: int = 1, per_op: int = 3) -> Dict[str, List[Tuple[str, ConceptTask, Q]]]:
    rng = random.Random(seed)
    out: Dict[str, List] = {}
    sets = {"written": POOLS["written"], "constructible": POOLS["constructible"], "novel": NOVEL,
            "composite": dict(POOLS["composite"], **COMPOSITE_NOVEL), "outside": OUTSIDE}
    k = 0
    for cat, ops in sets.items():
        for name, op in ops.items():
            for _ in range(per_op):
                k += 1
                task, truth = make_task(name, op, rng, rng.randint(4, 6), f"h{k}", teacher=True)
                out.setdefault(cat, []).append((name, task, truth))
    return out


def _fresh_task(t: ConceptTask, teacher: bool) -> ConceptTask:
    return ConceptTask(t.symbol, list(t.examples), t.question, t.oracle if teacher else None)


def evaluate(model: HRMX, tasks: Dict[str, List], teacher: bool = True) -> Dict[str, Dict[str, Any]]:
    snapshot = copy.deepcopy(model.mind.knowledge)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, items in tasks.items():
        t = {"n": 0, "correct": 0, "wrong": 0, "abstain": 0, "asks": 0, "ops": 0, "constructed": 0, "reused": 0,
             "wrong_items": []}
        for name, task, truth in items:
            model.mind.knowledge = copy.deepcopy(snapshot)
            model.domain = model.new_domain()
            r = model.solve(_fresh_task(task, teacher))
            t["n"] += 1
            t["ops"] += r.ops
            t["asks"] += sum(1 for s in r.trace if s.action == "ask")
            if r.status != "answer":
                t["abstain"] += 1
                continue
            ok = r.answer == truth
            t["correct" if ok else "wrong"] += 1
            if not ok:
                t["wrong_items"].append(f"{name}: said {r.answer}, truth {truth}")
            win = max((h for h in r.hypotheses if h.verified and h.answer == r.answer and h.payload),
                      key=lambda h: h.weight, default=None)
            if win is not None:
                fam = win.payload.family.name
                t["constructed"] += fam.startswith("construct[")
                t["reused"] += fam.startswith("lib[")
        t["mean_ops"] = round(t["ops"] / max(1, t["n"]), 1)
        del t["ops"]
        out[cat] = t
    model.mind.knowledge = snapshot
    model.domain = model.new_domain()
    return out


def surface(model: HRMX, tasks: List[Tuple[str, ConceptTask, Q]]) -> Dict[str, Any]:
    """U1: the same evidence through different templates and symbols gives the same answer and concept."""
    snapshot = copy.deepcopy(model.mind.knowledge)
    same_answer = same_concept = n = 0
    for i, (name, task, truth) in enumerate(tasks):
        answers, concepts = [], []
        for j, template in enumerate(TEMPLATES):
            model.mind.knowledge = copy.deepcopy(snapshot)
            model.domain = model.new_domain()
            chat = Conversation(model.mind, model.domain)
            sym = SYMBOLS[(i + j) % len(SYMBOLS)]
            rows = list(task.examples)
            random.Random(i * 10 + j).shuffle(rows)                   # order is irrelevant evidence
            for a, b, c in rows:
                chat.say(template.format(a=a, b=b, c=c, s=sym))
            reply = chat.say(f"{task.question[0]} {sym} {task.question[1]} = ?")
            answers.append(reply.split(" = ")[1].split(";")[0].split(" ")[0] if " = " in reply else "abstain")
            concepts.append(chat.concept_of.get(sym))
        n += 1
        same_answer += len(set(answers)) == 1
        same_concept += len(set(concepts)) == 1
    model.mind.knowledge = snapshot
    model.domain = model.new_domain()
    return {"tasks": n, "variants_per_task": len(TEMPLATES), "identical_answers": same_answer,
            "identical_concept": same_concept}


def revision(model: HRMX, trials: int = 6) -> Dict[str, Any]:
    """U6: evidence fits max; a later row contradicts it; the answer must follow the new evidence."""
    rng = random.Random(5)
    snapshot = copy.deepcopy(model.mind.knowledge)
    revised = kept_old = abstained = 0
    truth_op = lambda a, b: max(a, b) + (a > b) * (a - b)        # = max until a > b: then 2a - b
    for _ in range(trials):
        model.mind.knowledge = copy.deepcopy(snapshot)
        model.domain = model.new_domain()
        chat = Conversation(model.mind, model.domain)
        rows = []
        for _ in range(4):                                        # only a <= b rows: max(a, b) = b fits
            b = rng.randint(3, 9)
            a = rng.randint(1, b)
            rows.append((a, b))
        for a, b in rows:
            chat.say(f"{a} ◇ {b} = {truth_op(Q(a), Q(b))}")
        chat.say("9 ◇ 2 = ?")
        for a, b in [(7, 3), (8, 1), (6, 2), (5, 4)]:             # contradicting rows: a > b
            chat.say(f"{a} ◇ {b} = {truth_op(Q(a), Q(b))}")
        reply = chat.say("9 ◇ 2 = ?")
        truth = truth_op(Q(9), Q(2))
        if f"= {truth}" in reply:
            revised += 1
        elif "do not know" in reply:
            abstained += 1
        else:
            kept_old += 1
    model.mind.knowledge = snapshot
    model.domain = model.new_domain()
    return {"trials": trials, "revised_correctly": revised, "abstained": abstained, "wrong": kept_old}


def pattern_matchers(tasks: Dict[str, List]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Non-reasoning baselines: nearest evidence row, and least squares on (1, x, y, xy)."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {"nearest_row": {}, "least_squares": {}}
    for cat, items in tasks.items():
        nn = {"correct": 0, "wrong": 0}
        ls = {"correct": 0, "wrong": 0}
        for _, task, truth in items:
            ev = task.evidence()
            qx, qy = task.question
            near = min(ev, key=lambda d: (d[0][0] - qx) ** 2 + (d[0][1] - qy) ** 2)[1]
            nn["correct" if near == truth else "wrong"] += 1
            A = np.array([[1.0, float(x), float(y), float(x * y)] for (x, y), _ in ev])
            z = np.array([float(v) for _, v in ev])
            w, *_ = np.linalg.lstsq(A, z, rcond=None)
            pred = float(np.array([1.0, float(qx), float(qy), float(qx * qy)]) @ w)
            ls["correct" if abs(pred - float(truth)) < 1e-6 else "wrong"] += 1
        out["nearest_row"][cat] = nn
        out["least_squares"][cat] = ls
    return out


def trained_model(cfg: Optional[ModelConfig] = None, seed: int = 0) -> Tuple[HRMX, Dict[str, Any]]:
    model = HRMX(cfg or ModelConfig())
    report = model.train(Curriculum(seed=seed), log=None)
    return model, report


def run_suite(model: HRMX, per_op: int = 3) -> Dict[str, Any]:
    tasks = held_out(per_op=per_op)
    return {"held_out": evaluate(model, tasks), "surface": surface(model, tasks["constructible"][:8]),
            "revision": revision(model)}


def ablate(model: HRMX, tasks: Dict[str, List], **changes: Any) -> Dict[str, Any]:
    cfg = ModelConfig(**{**model.cfg.__dict__, **changes})
    clone = HRMX(cfg, Mind(Controller(lam=cfg.lam, wrong_penalty=cfg.wrong_penalty, ask_horizon=cfg.ask_horizon,
                                      use_questions=cfg.use_questions), knowledge=copy.deepcopy(model.mind.knowledge)))
    return evaluate(clone, tasks)


def main() -> Dict[str, Any]:
    model, train_report = trained_model()
    tasks = held_out()
    results: Dict[str, Any] = {"training": {k: v for k, v in train_report.items() if k != "seconds"}}
    results["trained"] = run_suite(model)
    results["fresh (no training)"] = evaluate(HRMX(ModelConfig()), tasks)
    results["v0.5 + ADR-012 (no construction, no memory)"] = evaluate(HRMX(ModelConfig(construct=False)), tasks)
    results["ablation: construction off at test (trained memory kept)"] = ablate(model, tasks, construct=False)
    results["ablation: memory off"] = ablate(model, tasks, use_memory=False)
    results["ablation: library tokens off"] = ablate(model, tasks, use_library_tokens=False)
    results["ablation: closed world (no BOTTOM)"] = ablate(model, tasks, bits_bottom=math.inf)
    results["ablation: question horizon 1"] = ablate(model, tasks, ask_horizon=1)
    no_teacher = HRMX(model.cfg, Mind(Controller(), knowledge=copy.deepcopy(model.mind.knowledge)))
    results["ablation: no teacher"] = evaluate(no_teacher, tasks, teacher=False)
    fixed, _ = trained_model(ModelConfig(answer_space="fixed"))
    results["variant: fixed |Y| = 2^10 (v0.5 constant, trained with it)"] = evaluate(fixed, tasks)
    results["baselines"] = pattern_matchers(tasks)
    return results


def _line(label: str, cats: Dict[str, Dict[str, Any]]) -> str:
    return f"{label:46s} " + "  ".join(f"{c[:6]} {v['correct']}/{v['wrong']}/{v['abstain']} ({v['mean_ops']:.0f})"
                                       for c, v in cats.items())


if __name__ == "__main__":
    res = main()
    print("[E19] correct/wrong/abstain (mean ops) per category")
    print(_line("trained", res["trained"]["held_out"]))
    for key, val in res.items():
        if key.startswith(("fresh", "ablation", "variant", "v0.5")):
            print(_line(key, val))
    print("[E19] surface", res["trained"]["surface"], "revision", res["trained"]["revision"])
    print("[E19] baselines", res["baselines"])
    record("x_understanding", {"experiment": "E19 understanding suite (U1-U10) with ablations",
                               "train_seed": 0, "held_out_seed": 1, "per_op": 3}, res)
