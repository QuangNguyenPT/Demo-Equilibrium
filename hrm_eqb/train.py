"""Training and evaluation.

Loss per batch (deep supervision over segments s = 1..S):
    L = (1/S) Σ_s [ CE(answer logits_s, target) + λ_halt · BCE(halt_s, 1[answer_s exactly right]) ]
        + λ_coda · CE(coda(detached chain), description bytes)
Gradients reach the core through the last grad_cycles cycles from each equilibrium (E4/E7).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hrm_eqb.coda import PAD, encode_text
from hrm_eqb.field import (Batch, answers_equal, budget_chunks, collate, decode_answer, encode_task,
                           random_augmentation)
from hrm_eqb.model import HRMEQB, SegmentOutput
from hrm_eqb.tasks import FAMILIES, Task, generate

DEFAULT_DOMAINS = {"grid": 0.4, "sequence": 0.2, "text": 0.2, "sudoku": 0.2}
EVAL_PAIRS = 8_000_000   # inference keeps no autograd tensors, so it can afford a larger budget


@dataclass
class TrainConfig:
    """How long and on what to train."""

    steps: int = 2000
    batch_tokens: int = 16384          # padded tokens per batch (count x max_n)
    batch_pairs: int = 4_000_000       # padded attention pairs per batch (count x max_n^2): bounds VRAM
    max_batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 100
    minutes: Optional[float] = None
    seed: int = 0
    domains: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DOMAINS))
    only: Tuple[str, ...] = ()        # restrict to these "domain/family" names (curriculum, tests)
    extra_weight: float = 0.0          # probability of drawing a batch from `extra` tasks (e.g. ARC training)
    segments: Optional[int] = None     # deep-supervision segments per step; None = the model's max_segments
    solve_tol: float = 1e-2            # training solve tolerance
    halt_weight: float = 0.2
    coda_weight: float = 0.2
    ema: float = 0.999
    grad_clip: float = 1.0
    log_every: int = 50


def task_batches(rng: np.random.Generator, cfg: TrainConfig, max_side: int,
                 extra: Sequence[Task] = ()) -> Iterator[List[Task]]:
    """Endless batches within the token and attention-pair budgets, one source per batch; grids augmented."""
    names = sorted(cfg.domains)
    weights = np.array([cfg.domains[n] for n in names], dtype=np.float64)
    while True:
        if extra and rng.random() < cfg.extra_weight:
            draw = (lambda: extra[int(rng.integers(0, len(extra)))])
        elif cfg.only:
            domain, family = cfg.only[int(rng.integers(0, len(cfg.only)))].split("/")
            draw = (lambda: generate(domain, family, rng, max_side))
        else:
            domain = names[int(rng.choice(len(names), p=weights / weights.sum()))]
            fams = FAMILIES[domain]
            draw = (lambda: generate(domain, fams[int(rng.integers(0, len(fams)))], rng, max_side))
        picks: List[Task] = []
        n_max = 0
        while len(picks) < cfg.max_batch:
            task = draw()
            task = random_augmentation(rng).apply(task) if task.kind == "grid" else task
            grown = max(n_max, encode_task(task).content.size)
            size = len(picks) + 1
            if picks and (size * grown > cfg.batch_tokens or size * grown * grown > cfg.batch_pairs):
                break
            picks.append(task)
            n_max = grown
        yield picks


def exact(outputs: SegmentOutput, batch: Batch) -> torch.Tensor:
    """(B,) bool: every answer cell right."""
    wrong = (outputs.logits.argmax(-1) != batch.target) & batch.answer_valid
    return ~wrong.any(dim=1)


def answer_loss(outputs: List[SegmentOutput], batch: Batch, halt_weight: float) -> torch.Tensor:
    total = torch.zeros((), device=batch.content.device)
    for out in outputs:
        ce = F.cross_entropy(out.logits.transpose(1, 2), batch.target, reduction="none")
        ce = (ce * batch.answer_valid).sum() / batch.answer_valid.sum().clamp_min(1)
        halt = F.binary_cross_entropy_with_logits(out.halt, exact(out, batch).float())
        total = total + ce + halt_weight * halt
    return total / len(outputs)


def text_targets(tasks: Sequence[Task], max_len: int, device: torch.device) -> torch.Tensor:
    rows = [encode_text(t.description, max_len) for t in tasks]
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out.to(device)


def coda_loss(model: HRMEQB, outputs: List[SegmentOutput], batch: Batch, tasks: Sequence[Task]) -> torch.Tensor:
    tokens = text_targets(tasks, model.cfg.max_text, batch.content.device)
    answer_mask = torch.zeros_like(batch.valid)
    answer_mask.scatter_(1, batch.answer_pos, batch.answer_valid)
    memory = model.coda.memory([o.high for o in outputs], batch.valid, answer_mask)
    logits = model.coda(memory, model.coda.memory_mask(batch.valid), tokens[:, :-1])
    return F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], ignore_index=PAD)


def ema_update(ema: HRMEQB, model: HRMEQB, decay: float, step: int) -> None:
    """Exponential moving average with warmup: decay_t = min(decay, (1 + t) / (10 + t))."""
    rate = min(decay, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.lerp_(p, 1.0 - rate)


def train(model: HRMEQB, cfg: TrainConfig, device: torch.device, extra: Sequence[Task] = (),
          log: Callable[[str], None] = print) -> HRMEQB:
    """Train in place; returns the EMA copy (the weights that are evaluated and exported)."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    start, stream = time.time(), task_batches(rng, cfg, model.cfg.max_side, extra)
    for step in range(1, cfg.steps + 1):
        tasks = next(stream)
        encoded = [encode_task(t) for t in tasks]
        keep = [i for i, e in enumerate(encoded) if e.target is not None]
        batch = collate([encoded[i] for i in keep]).to(device)
        outputs = model(batch, segments=cfg.segments, tol=cfg.solve_tol)
        loss = answer_loss(outputs, batch, cfg.halt_weight)
        if cfg.coda_weight > 0:
            loss = loss + cfg.coda_weight * coda_loss(model, outputs, batch, [tasks[i] for i in keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        ema_update(ema, model, cfg.ema, step)
        if step % cfg.log_every == 0 or step == 1:
            acc = float(exact(outputs[-1], batch).float().mean())
            log(f"[TRAIN] step {step} loss {float(loss.detach()):.4f} task_acc {acc:.3f} "
                f"cycles {outputs[-1].cycles} bound {outputs[-1].error_bound:.1e} "
                f"t {time.time() - start:.0f}s")
        if cfg.minutes is not None and time.time() - start > 60 * cfg.minutes:
            log(f"[TRAIN] time budget reached at step {step}")
            break
    return ema


@dataclass
class Evaluation:
    """Exact-match accuracy overall and per (domain, family)."""

    correct: List[bool]
    tasks: List[Task]
    predictions: List[np.ndarray]

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.correct)) if self.correct else 0.0

    def by_family(self) -> Dict[str, float]:
        groups: Dict[str, List[bool]] = {}
        for ok, t in zip(self.correct, self.tasks):
            groups.setdefault(f"{t.domain}/{t.family}", []).append(ok)
        return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


@torch.no_grad()
def predict(model: HRMEQB, tasks: Sequence[Task], device: torch.device, pairs: int = EVAL_PAIRS,
            segments: Optional[int] = None) -> List[np.ndarray]:
    """Answer of the last segment for every task (identity view), in attention-pair-budgeted batches."""
    model.to(device).eval()
    encoded = [encode_task(t) for t in tasks]
    preds: List[np.ndarray] = [np.zeros((0, 0), dtype=np.int64)] * len(tasks)
    for group in budget_chunks(encoded, pairs):
        chunk = [encoded[i] for i in group]
        ids = model(collate(chunk).to(device), segments=segments)[-1].logits.argmax(-1).cpu().numpy()
        for i, enc, row in zip(group, chunk, ids):
            preds[i] = decode_answer(enc, row[:enc.answer_pos.size])
    return preds


def evaluate(model: HRMEQB, tasks: Sequence[Task], device: torch.device, pairs: int = EVAL_PAIRS,
             segments: Optional[int] = None) -> Evaluation:
    preds = predict(model, tasks, device, pairs, segments)
    return Evaluation([answers_equal(p, t.answer) for p, t in zip(preds, tasks)], list(tasks), preds)
