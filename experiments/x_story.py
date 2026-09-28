"""E3 variable binding, E4 context-dependent roles, E5 novel relation combinations, E8 counterfactuals,
plus contradiction and missing-information behaviour, against an end-to-end neural baseline.

Hypothesis: meaning lives in the situation (who has what, what changed), so an explicit
situation model transfers to unseen names, magnitudes, longer event chains and counterfactual
edits, while an end-to-end model trained on the same distribution does not.
Falsified if the baseline matches HRM-X on the shifted splits, or HRM-X fails them.

The generator and the truth simulator below are independent of hrm_x (own templates' filling,
own dictionary simulation). Both systems see only text.
  python -m experiments.x_story
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from experiments.common import record, table
from hrm_x.domains.story import TRAIN_ITEMS, TRAIN_NAMES, Parser, Story, StoryTask
from hrm_x.mind import Mind

SEED = 0
ANSWERS = 128        # baseline answers by classification over 0..127 (covers the training range)
EPOCHS = 150
UNSEEN_NAMES = ["Zoltan", "Quenby", "Xavier", "Wendell", "Priya", "Kofi", "Leilani", "Oskar", "Mirela", "Tomasz"]
UNSEEN_ITEMS = ["oranges", "pencils", "stamps", "shells", "cards"]
EVENT_TEXT = {
    "give": ["{a} gives {b} {n} {i}.", "{a} gives {n} {i} to {b}.", "{b} receives {n} {i} from {a}.",
             "{b} takes {n} {i} from {a}."],
    "buy": ["{a} buys {n} {i}.", "{a} finds {n} {i}."],
    "lose": ["{a} eats {n} {i}.", "{a} loses {n} {i}."],
}


@dataclass
class Case:
    sentences: List[str]
    question: str
    truth: Optional[int]          # None = the story is contradictory or does not fix the answer
    events: List[Tuple]           # for counterfactual edits


def truth_of(start: Dict[str, int], events: List[Tuple], ask: List[str]) -> Optional[int]:
    state = dict(start)
    for kind, a, b, n in events:
        if kind == "give":
            state[a] -= n
            state[b] += n
        else:
            state[a] += n if kind == "buy" else -n
        if min(state.values()) < 0:
            return None
    return sum(state[x] for x in ask)


def story(rng: random.Random, names: List[str], items: List[str], n_ent: int, n_events: int,
          lo: int, hi: int, allow_bad: bool = False) -> Case:
    while True:
        people = rng.sample(names, n_ent)
        item = rng.choice(items)
        start = {p: rng.randint(lo, hi) for p in people}
        events = []
        state = dict(start)
        for _ in range(n_events):
            kind = rng.choice(["give", "give", "buy", "lose"])
            a, b = rng.sample(people, 2) if len(people) > 1 else (people[0], people[0])
            cap = state[a] if kind in ("give", "lose") else hi
            if cap < 1 and not allow_bad:
                continue
            n = rng.randint(1, max(1, cap)) if not allow_bad else rng.randint(1, hi)
            events.append((kind, a, b, n))
            if kind == "give":
                state[a] -= n
                state[b] += n
            else:
                state[a] += n if kind == "buy" else -n
        if len(events) == n_events:
            break
    ask = rng.sample(people, rng.choice([1, 2]) if n_ent > 1 else 1)
    sentences = [f"{p} has {start[p]} {item}." for p in people]
    for kind, a, b, n in events:
        sentences.append(rng.choice(EVENT_TEXT[kind]).format(a=a, b=b, n=n, i=item))
    q = f"How many {item} does {ask[0]} have?" if len(ask) == 1 else \
        f"How many {item} do {ask[0]} and {ask[1]} have together?"
    return Case(sentences, q, truth_of(start, events, ask), events)


# ---------- end-to-end baseline: transformer regression from text to answer ----------

class Baseline(nn.Module):
    """Tokens (entities anonymized per story, numbers as <num> + value channel) -> answer."""

    def __init__(self, vocab: int, dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.value = nn.Linear(1, dim)
        self.pos = nn.Embedding(96, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.head = nn.Linear(dim, ANSWERS)

    def forward(self, ids, vals):
        pad = ids == 0
        h = self.embed(ids) + self.value(vals.unsqueeze(-1)) + self.pos(torch.arange(ids.shape[1]))
        h = self.enc(h, src_key_padding_mask=pad)
        return self.head((h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True))


def encode(case: Case, vocab: Dict[str, int], scale: float) -> Tuple[List[int], List[float]]:
    ids, vals, ents = [], [], {}
    words = " ".join(case.sentences + [case.question]).replace(".", " .").replace("?", " ?").split()
    for w in words:
        if w.isdigit():
            ids.append(vocab.setdefault(f"<n{w}>" if int(w) <= 40 else "<num>", len(vocab)))
            vals.append(int(w) / scale)
            continue
        if w[0].isupper() and w.lower() not in ("how",):
            w = ents.setdefault(w, f"<e{len(ents)}>")
        elif w.endswith("s") and w not in ("has", "does", "gives", "receives", "takes", "buys", "finds", "eats",
                                           "loses", "apples", "do"):
            w = "<item>"
        elif w in [t.lower() for t in TRAIN_ITEMS + UNSEEN_ITEMS]:
            w = "<item>"
        ids.append(vocab.setdefault(w.lower(), len(vocab)))
        vals.append(0.0)
    return ids, vals


def train_baseline(cases: List[Case], seed: int, epochs: int):
    torch.manual_seed(seed)
    vocab = {"<pad>": 0, "<num>": 1}
    scale = 40.0
    enc = [encode(c, vocab, scale) for c in cases]
    model = Baseline(len(vocab) + 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    width = max(len(i) for i, _ in enc)
    ids = torch.zeros(len(cases), width, dtype=torch.long)
    vals = torch.zeros(len(cases), width)
    for k, (i, v) in enumerate(enc):
        ids[k, :len(i)] = torch.tensor(i)
        vals[k, :len(v)] = torch.tensor(v)
    y = torch.tensor([min(c.truth, ANSWERS - 1) for c in cases])
    for epoch in range(epochs):
        perm = torch.randperm(len(cases))
        for s in range(0, len(cases), 64):
            b = perm[s:s + 64]
            loss = nn.functional.cross_entropy(model(ids[b], vals[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        train_acc = float((model(ids, vals).argmax(-1) == y).float().mean())

    @torch.no_grad()
    def predict(case: Case) -> int:
        i, v = encode(case, dict(vocab), scale)
        i = [t if t < model.embed.num_embeddings else vocab["<num>"] for t in i]
        return int(model(torch.tensor([i[:96]]), torch.tensor([v[:96]]))[0].argmax())
    return predict, sum(p.numel() for p in model.parameters()), train_acc


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000, help="baseline training stories")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--name", default="x_story")
    args = ap.parse_args()
    rng = random.Random(SEED)
    t0 = time.time()
    parser = Parser()
    parser.train(seed=SEED)
    parse_s = time.time() - t0
    train = [story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, rng.randint(1, 2), 1, 20) for _ in range(args.train)]
    train = [c for c in train if c.truth is not None]
    t0 = time.time()
    baseline, base_params, base_train_acc = train_baseline(train, SEED, args.epochs)
    base_s = time.time() - t0
    print(f"[BASELINE] {base_params} params, train accuracy {base_train_acc:.3f}, {base_s:.0f}s")

    splits = {
        "iid": [story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, rng.randint(1, 2), 1, 20) for _ in range(150)],
        "unseen names+items": [story(rng, UNSEEN_NAMES, UNSEEN_ITEMS, 2, rng.randint(1, 2), 1, 20) for _ in range(150)],
        "large numbers": [story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, rng.randint(1, 2), 100, 999) for _ in range(150)],
        "long chains (3 people, 4-6 events)": [story(rng, TRAIN_NAMES, TRAIN_ITEMS, 3, rng.randint(4, 6), 1, 20)
                                               for _ in range(150)],
        "contradictory": [c for c in (story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, 2, 1, 5, allow_bad=True)
                                      for _ in range(600)) if c.truth is None][:100],
    }
    dom = Story(parser)
    rows, metrics = [], {}
    for name, cases in splits.items():
        good_x = good_b = wrong_x = 0
        ops, cycles = [], []
        for c in cases:
            r = Mind().solve(dom, StoryTask(c.sentences, c.question))
            ops.append(r.ops)
            cycles.append(r.cycles)
            if c.truth is None:
                good_x += r.status == "abstain"
                wrong_x += r.status == "answer"
            else:
                good_x += r.status == "answer" and r.answer == c.truth
                wrong_x += r.status == "answer" and r.answer != c.truth
                good_b += baseline(c) == c.truth
        n = len(cases)
        m = {"n": n, "hrmx_correct": good_x / n, "hrmx_wrong_answers": wrong_x / n,
             "baseline_correct": good_b / n if name != "contradictory" else 0.0,
             "hrmx_mean_ops": float(np.mean(ops)), "hrmx_mean_cycles": float(np.mean(cycles))}
        metrics[name] = m
        rows.append([name, n, f"{m['hrmx_correct']:.3f}", f"{m['hrmx_wrong_answers']:.3f}",
                     f"{m['baseline_correct']:.3f}", f"{m['hrmx_mean_ops']:.0f}", f"{m['hrmx_mean_cycles']:.1f}"])

    # E4 minimal pairs (role swap) and E8 counterfactual edits
    pair_ok, pairs = 0, 0
    cf_ok, cfs = 0, 0
    for _ in range(100):
        a, b = rng.sample(TRAIN_NAMES, 2)
        na, nb, k = rng.randint(5, 20), rng.randint(5, 20), rng.randint(1, 4)
        base = [f"{a} has {na} apples.", f"{b} has {nb} apples."]
        q = f"How many apples does {b} have?"
        r1 = Mind().solve(dom, StoryTask(base + [f"{a} gives {b} {k} apples."], q))
        r2 = Mind().solve(dom, StoryTask(base + [f"{b} gives {a} {k} apples."], q))
        pairs += 1
        pair_ok += (r1.answer, r2.answer) == (nb + k, nb - k)
        c = story(rng, TRAIN_NAMES, TRAIN_ITEMS, 2, 3, 5, 20)
        if c.truth is None:
            continue
        idx = rng.randrange(len(c.events))
        kind, x, y, n = c.events[idx]
        new_events = list(c.events)
        new_events[idx] = (kind, x, y, n + 1)
        start = {s.split()[0]: int(s.split()[2]) for s in c.sentences if " has " in s}
        ask = [w for w in c.question.replace("?", "").split() if w in start]
        new_truth = truth_of(start, new_events, ask)
        if new_truth is None:
            continue
        edited = list(c.sentences)
        edited[len(start) + idx] = edited[len(start) + idx].replace(f" {n} ", f" {n + 1} ")
        before = Mind().solve(dom, StoryTask(c.sentences, c.question))
        after = Mind().solve(dom, StoryTask(edited, c.question))
        cfs += 1
        cf_ok += before.answer == c.truth and after.answer == new_truth
    metrics["E4 role-swap minimal pairs"] = pair_ok / pairs
    metrics["E8 counterfactual edits (both before and after correct)"] = cf_ok / max(cfs, 1)

    # boundary: a sentence frame never seen in training (passive voice)
    passive = Mind().solve(dom, StoryTask(["Alice has 5 apples.", "Bob has 1 apples.", "3 apples are given to Bob by Alice."],
                                          "How many apples does Bob have?"))
    metrics["boundary: unseen passive frame"] = {"status": passive.status, "answer": str(passive.answer),
                                                 "reason": passive.reason}
    print(table(rows, ["split", "n", "HRM-X correct", "HRM-X wrong", "baseline correct", "ops", "cycles"]))
    print(f"[E4] role-swap minimal pairs: {metrics['E4 role-swap minimal pairs']:.3f}")
    print(f"[E8] counterfactual edits: {metrics['E8 counterfactual edits (both before and after correct)']:.3f} ({cfs} stories)")
    print(f"[BOUNDARY] passive frame -> {passive.status} {passive.answer} ({passive.reason})")
    record(args.name, {"seed": SEED, "baseline_train_stories": len(train),
                       "parser_params": parser.params, "parser_train_seconds": parse_s,
                       "parser_data": "4000 template sentences, 20 names, 6 items, amounts 1-20",
                       "baseline_params": base_params, "baseline_train_seconds": base_s,
                       "baseline_train_accuracy": base_train_acc,
                       "baseline": f"3-layer transformer d=64, entities anonymized, numbers <= 40 as tokens + value "
                                   f"channel, classification over 0..{ANSWERS - 1}, {args.epochs} epochs",
                       "train_distribution": "2 people, 1-2 events, amounts 1-20, train names/items"}, metrics)


if __name__ == "__main__":
    main()
