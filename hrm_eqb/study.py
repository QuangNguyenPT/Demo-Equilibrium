"""Self-study, influenced by karpathy/autoresearch: fixed budget, one frozen metric, keep or discard.

One experiment:
  1. the frozen harness (seeded held-out tasks + exact-match scorer, hash-locked) scores the
     current generation;
  2. curiosity picks the weakest family that is still learnable;
  3. a candidate copy practises for a fixed wall-clock budget, weighted toward that family;
  4. the candidate is kept only if its paired improvement clears the Hoeffding margin at
     δ / (experiments so far) (G1); otherwise it is discarded;
  5. one row is appended to results.tsv either way.
The harness never changes under a model: its hash is stored at the first experiment and a
mismatch refuses to run.
"""

import copy
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from hrm_eqb import hrmc
from hrm_eqb import knowledge as kn
from hrm_eqb.device import select_device
from hrm_eqb.gate import keep_margin, should_keep
from hrm_eqb.mind import CORE, KNOWLEDGE, STATE, Mind
from hrm_eqb.model import HRMEQB
from hrm_eqb.tasks import FAMILIES, Task, generate
from hrm_eqb.train import TrainConfig, evaluate, train

LEDGER = "results.tsv"
LEDGER_HEADER = "time\tgeneration\tfamily\tbaseline\tcandidate\tmean_diff\tmargin\tdecision\tminutes\tcore_sha256\n"
HARNESS_SEED = 4242
HOPELESS_AFTER = 2


def harness(n_per_family: int, max_side: int = 10) -> List[Task]:
    """The frozen study metric: seeded tasks from every generated family."""
    rng = np.random.default_rng(HARNESS_SEED)
    return [generate(d, f, rng, max_side) for d, fams in sorted(FAMILIES.items()) for f in fams
            for _ in range(n_per_family)]


def harness_hash(tasks: List[Task]) -> str:
    h = hashlib.sha256()
    for t in tasks:
        for x, y in t.demos:
            h.update(x.tobytes() + y.tobytes())
        h.update(t.query.tobytes() + t.answer.tobytes())
    return h.hexdigest()


def load_model(directory: Path) -> HRMEQB:
    """Core + knowledge as trainable parameters (for practice, not for serving)."""
    model, _ = hrmc.load(directory / CORE)
    data = torch.load(directory / KNOWLEDGE, weights_only=True, map_location="cpu")
    with torch.no_grad():
        for name in ("keys_a", "keys_b", "values"):
            getattr(model.memory, name).copy_(data[name])
    return model


@dataclass
class StudyResult:
    family: str
    baseline: float
    candidate: float
    mean_diff: float
    margin: float
    kept: bool


def study_once(directory: Path, minutes: float, n_per_family: int = 20, delta: float = 0.05,
               device: Optional[torch.device] = None, log=print) -> StudyResult:
    directory = Path(directory)
    device = device or select_device()
    state = json.loads((directory / STATE).read_text(encoding="utf-8"))
    tasks = harness(n_per_family)
    digest = harness_hash(tasks)
    if state.setdefault("harness_sha256", digest) != digest:
        raise RuntimeError("the frozen harness changed under this model; refusing to study")
    model = load_model(directory)
    base = evaluate(model, tasks, device)
    fam_acc = base.by_family()
    streaks: Dict[str, int] = state.setdefault("zero_streak", {})
    for fam, acc in fam_acc.items():
        streaks[fam] = streaks.get(fam, 0) + 1 if acc == 0.0 else 0
    learnable = {f: a for f, a in fam_acc.items() if streaks[f] <= HOPELESS_AFTER and a < 1.0}
    if not learnable:
        log("[STUDY] nothing learnable left in the harness")
        (directory / STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return StudyResult("", base.accuracy, base.accuracy, 0.0, 0.0, False)
    target = min(learnable, key=learnable.get)
    domain = target.split("/")[0]
    log(f"[STUDY] baseline {base.accuracy:.3f}; practising {target} ({fam_acc[target]:.2f}) for {minutes} min")
    candidate = copy.deepcopy(model)
    weights = {d: (3.0 if d == domain else 1.0) for d in FAMILIES}
    ema = train(candidate, TrainConfig(steps=10 ** 9, minutes=minutes, domains=weights, seed=int(time.time()),
                                       log_every=100), device, log=log)
    cand = evaluate(ema, tasks, device)
    diffs = torch.tensor([float(c) - float(b) for c, b in zip(cand.correct, base.correct)])
    experiments = state.get("experiments", 0) + 1
    kept = should_keep(diffs, delta, experiments)
    margin = keep_margin(len(tasks), delta / experiments)
    generation = state.get("generation", 0)
    sha = state.get("core_sha256", "")
    if kept:
        generation += 1
        archive = directory / "generations" / f"g{generation - 1}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in (CORE, KNOWLEDGE):
            shutil.copy2(directory / name, archive / name)
        sha = Mind.create(directory, ema, {"generation": generation, "parent": state.get("core_sha256"),
                                           "note": f"study: {target}"})
        state = {**json.loads((directory / STATE).read_text(encoding="utf-8")),
                 **{k: v for k, v in state.items() if k not in ("generation", "core_sha256")}}
    state.update({"experiments": experiments, "generation": generation, "core_sha256": sha})
    (directory / STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")
    ledger = directory / LEDGER
    if not ledger.exists():
        ledger.write_text(LEDGER_HEADER, encoding="utf-8")
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{generation}\t{target}\t{base.accuracy:.4f}\t"
                 f"{cand.accuracy:.4f}\t{float(diffs.mean()):.4f}\t{margin:.4f}\t{'keep' if kept else 'discard'}\t"
                 f"{minutes}\t{sha}\n")
    log(f"[STUDY] candidate {cand.accuracy:.3f} (mean diff {float(diffs.mean()):+.3f}, margin {margin:.3f}) "
        f"-> {'KEEP' if kept else 'discard'}")
    return StudyResult(target, base.accuracy, cand.accuracy, float(diffs.mean()), margin, kept)
