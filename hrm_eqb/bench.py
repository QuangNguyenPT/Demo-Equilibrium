"""Benchmarks: generated families, language (text answers + coda descriptions), Sudoku, ARC-AGI-1.

Every suite reports what it could not attempt (e.g. ARC tasks too large for the token budget)
next to what it scored, so coverage is never hidden inside an average.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hrm_eqb.curiosity import tally
from hrm_eqb.field import answers_equal, budget_chunks, collate, decode_answer, encode_task, views
from hrm_eqb.mind import _undo
from hrm_eqb.model import HRMEQB
from hrm_eqb.tasks import FAMILIES, Task, fits, generate, load_arc, load_sudoku_csv
from hrm_eqb.train import EVAL_PAIRS, evaluate

BENCH_SEED = 9090
MAX_TOKENS = 4096


@dataclass
class SuiteResult:
    name: str
    attempted: int
    skipped: int
    accuracy: float
    detail: Dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        extra = ", ".join(f"{k}={v:.3f}" for k, v in self.detail.items())
        return (f"{self.name}: {self.accuracy:.3f} on {self.attempted} attempted, {self.skipped} skipped"
                + (f" ({extra})" if extra else ""))


def generated_tasks(n_per_family: int, domains: Sequence[str] = ("grid", "sequence", "text"),
                    max_side: int = 10) -> List[Task]:
    rng = np.random.default_rng(BENCH_SEED)
    return [generate(d, f, rng, max_side) for d in domains for f in FAMILIES[d] for _ in range(n_per_family)]


def generated(model: HRMEQB, device: torch.device, n_per_family: int = 50) -> SuiteResult:
    ev = evaluate(model, generated_tasks(n_per_family), device)
    return SuiteResult("generated", len(ev.tasks), 0, ev.accuracy, ev.by_family())


@torch.no_grad()
def language(model: HRMEQB, device: torch.device, n_per_family: int = 50) -> SuiteResult:
    """Text-domain answers (structured head) and descriptions (coda) of every generated domain."""
    tasks = generated_tasks(n_per_family, ("text",))
    ev = evaluate(model, tasks, device)
    described = generated_tasks(max(1, n_per_family // 5), ("grid", "sequence", "text"))
    exact, chars = [], []
    model.to(device).eval()
    encoded = [encode_task(t) for t in described]
    for group in budget_chunks(encoded, EVAL_PAIRS):
        chunk = [described[i] for i in group]
        batch = collate([encoded[i] for i in group]).to(device)
        outs = model(batch)
        mask = torch.zeros_like(batch.valid)
        mask.scatter_(1, batch.answer_pos, batch.answer_valid)
        texts = model.coda.describe(model.coda.memory([o.high for o in outs], batch.valid, mask),
                                    model.coda.memory_mask(batch.valid))
        for t, said in zip(chunk, texts):
            exact.append(said == t.description)
            n = max(len(t.description), len(said), 1)
            chars.append(sum(a == b for a, b in zip(said, t.description)) / n)
    return SuiteResult("language", len(tasks), 0, ev.accuracy,
                       {**ev.by_family(), "describe_exact": float(np.mean(exact)), "describe_chars": float(np.mean(chars))})


@torch.no_grad()
def committee_accuracy(model: HRMEQB, tasks: Sequence[Task], device: torch.device, n_views: int = 4,
                       attempts: int = 2) -> List[bool]:
    """pass@attempts with the committee's most voted de-augmented answers (ARC scoring)."""
    rng = np.random.default_rng(BENCH_SEED)
    model.to(device).eval()
    ok = []
    for task in tasks:
        augs = views(task, n_views, rng)
        members = [a.apply(task) for a in augs] if task.kind == "grid" else [task]
        encoded = [encode_task(m) for m in members]
        answers = []
        for group in budget_chunks(encoded, EVAL_PAIRS):
            chunk = [encoded[i] for i in group]
            ids = model(collate(chunk).to(device))[-1].logits.argmax(-1).cpu().numpy()
            answers += [decode_answer(e, row[:e.answer_pos.size]) for e, row in zip(chunk, ids)]
        tallied = tally([_undo(a, ans, task) for a, ans in zip(augs, answers)])
        ok.append(any(answers_equal(a, task.answer) for a, _ in tallied[:attempts]))
    return ok


def arc(model: HRMEQB, device: torch.device, directory: Path, limit: Optional[int] = None) -> SuiteResult:
    tasks = load_arc(directory)
    if limit:
        tasks = tasks[:limit]
    usable = [t for t in tasks if fits(t, MAX_TOKENS) and encode_task(t).target is not None]
    ok = committee_accuracy(model, usable, device)
    solved = sum(ok)
    return SuiteResult("arc", len(usable), len(tasks) - len(usable), solved / max(len(tasks), 1),
                       {"on_attempted": solved / max(len(usable), 1), "solved": float(solved)})


@torch.no_grad()
def sudoku(model: HRMEQB, device: torch.device, csv_path: Optional[Path] = None, limit: int = 200) -> SuiteResult:
    rng = np.random.default_rng(BENCH_SEED)
    own = [generate("sudoku", "sudoku", rng) for _ in range(limit)]
    ev_own = evaluate(model, own, device)
    detail = {"generated_solved": ev_own.accuracy,
              "generated_cells": _cell_accuracy(ev_own.predictions, own)}
    attempted, acc = len(own), ev_own.accuracy
    if csv_path is not None and Path(csv_path).exists():
        ext = load_sudoku_csv(Path(csv_path), limit)
        ev_ext = evaluate(model, ext, device)
        detail.update({"extreme_solved": ev_ext.accuracy, "extreme_cells": _cell_accuracy(ev_ext.predictions, ext)})
        attempted, acc = attempted + len(ext), ev_ext.accuracy
    return SuiteResult("sudoku", attempted, 0, acc, detail)


def _cell_accuracy(preds: Sequence[np.ndarray], tasks: Sequence[Task]) -> float:
    scores = []
    for p, t in zip(preds, tasks):
        blank = t.query == 0
        scores.append(float((p[blank] == t.answer[blank]).mean()) if p.shape == t.answer.shape else 0.0)
    return float(np.mean(scores))


def empty_brain(model: HRMEQB, device: torch.device, n_per_family: int = 50) -> SuiteResult:
    """Ablation: the same generated suite with every knowledge read returning zero."""
    silent = copy.deepcopy(model)
    silent.memory.silent = True
    ev = evaluate(silent, generated_tasks(n_per_family), device)
    return SuiteResult("empty_brain", len(ev.tasks), 0, ev.accuracy)


@torch.no_grad()
def schedule_invariance(model: HRMEQB, device: torch.device, n_per_family: int = 10) -> SuiteResult:
    """E3 on the trained model: answers with n_low = 1 and n_low = 6 must coincide."""
    tasks = generated_tasks(n_per_family)
    answers = []
    for n_low in (1, 6):
        variant = copy.deepcopy(model)
        variant.cfg = dataclasses.replace(model.cfg, n_low=n_low, solve_tol=1e-5, max_cycles=400)
        answers.append(evaluate(variant, tasks, device).predictions)
    a, b = answers
    same = [answers_equal(x, y) for x, y in zip(a, b)]
    return SuiteResult("schedule_invariance", len(tasks), 0, float(np.mean(same)))


def report(results: Sequence[SuiteResult], path: Path) -> None:
    Path(path).write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")

