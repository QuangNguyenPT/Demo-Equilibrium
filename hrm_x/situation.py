"""Situation model: quantities as a state vector, events as group actions learned one-shot (doc 03, s.1, 4).

State  q in Q^(E x I): quantity of item i held by entity e (exact Fractions).
Event  a sentence with frame f (its non-entity, non-number, non-item words, e.g. ('gives','to')),
       entities (E1, E2) in order of mention and amount n acts by the translation
           q  ->  q + n * (alpha_f e_E1 + beta_f e_E2)          (an element of the group Q^E)
       The frame's (alpha_f, beta_f) are UNKNOWN until taught by one worked example (before/after
       quantities). By Theorem 3 a coefficient is determined only if its entity's quantity after the
       example was observed; a frame never taught is undetermined -> abstain and ask for an example.
Law    alpha_f + beta_f = 0 marks the frame as a transfer (conserves the total); the verifier checks
       conservation with a different computation (totals, not per-entity replay).
Reader convention (hand-written here, the learned Reader of doc 04 in the full system): capitalised
       non-function words are entities, digit strings are amounts (read exactly), the word after the
       amount is the item. This part is NOT claimed as understanding.

Where this sits in the repo: this is the v0.5 story state model. It is the language-facing half of
HRM-X: every sentence is reduced to a group action on an exact-rational state vector, and a frame's
action is learned from a single worked example rather than being written into the program. The
arithmetic is deliberately exact and torch-free, so the story side of the system stays independent
of the tensor code in hrm_x.algebra.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

FUNCTION_CAPS = {"How", "What", "If", "The", "She", "He", "They", "Then", "After", "Now"}
TOKEN = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+|[?.,]")


@dataclass
class Parsed:
    entities: List[str]
    amount: Optional[Fraction]
    item: Optional[str]
    frame: Tuple[str, ...]


def parse(sentence: str) -> Parsed:
    toks = TOKEN.findall(sentence)
    ents, amt, item, frame = [], None, None, []
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            amt = Fraction(t)
            if i + 1 < len(toks) and toks[i + 1].isalpha():
                item = toks[i + 1].lower().rstrip("s")
                i += 1
        elif t[0].isupper() and t not in FUNCTION_CAPS:
            ents.append(t)
        elif t.isalpha():
            frame.append(t.lower())
        i += 1
    return Parsed(ents, amt, item, tuple(frame))


@dataclass
class FrameOp:
    alpha: Optional[Fraction]
    beta: Optional[Fraction]
    arity: int

    @property
    def conserves(self) -> bool:
        return self.arity == 2 and self.alpha is not None and self.beta is not None and self.alpha + self.beta == 0


@dataclass
class Memory:
    frames: Dict[Tuple[str, ...], FrameOp] = field(default_factory=dict)


STATE_FRAME = ("has",)


def teach(mem: Memory, before: Dict[str, Fraction], sentence: str, after: Dict[str, Fraction]) -> FrameOp:
    """One worked example -> the frame's coefficients (exact solve of one equation per observed entity)."""
    p = parse(sentence)
    assert p.amount is not None and p.amount != 0 and 1 <= len(p.entities) <= 2
    coef = []
    for e in p.entities:
        coef.append((after[e] - before.get(e, Fraction(0))) / p.amount if e in after else None)
    op = FrameOp(coef[0], coef[1] if len(coef) > 1 else Fraction(0), len(p.entities))
    mem.frames[p.frame] = op
    return op


@dataclass
class StoryAnswer:
    status: str
    answer: Optional[Fraction]
    reason: str
    ops: int
    verified: bool = False
    unknown_frames: List[Tuple[str, ...]] = field(default_factory=list)


def run_story(mem: Memory, sentences: List[str]) -> Tuple[Optional[Dict[Tuple[str, str], Fraction]], str, int, bool, List[Tuple[str, ...]]]:
    q: Dict[Tuple[str, str], Fraction] = {}
    ops, unknown = 0, []
    acquired = Fraction(0)                      # net change of the total from non-conserving events
    initial_total = Fraction(0)
    for s in sentences:
        p = parse(s)
        if p.frame == STATE_FRAME and len(p.entities) == 1 and p.amount is not None:
            key = (p.entities[0], p.item or "")
            initial_total += p.amount - q.get(key, Fraction(0))
            q[key] = p.amount
            continue
        op = mem.frames.get(p.frame)
        if op is None:
            unknown.append(p.frame); continue
        if op.arity != len(p.entities) or p.amount is None:
            return None, f"frame {p.frame} used with {len(p.entities)} entities, taught with {op.arity}", ops, False, unknown
        for e, c in zip(p.entities, (op.alpha, op.beta)):
            if c is None:
                return None, f"frame {p.frame}: effect on its role for {e} was never observed (undetermined)", ops, False, unknown
            key = (e, p.item or "")
            if key not in q:
                return None, f"initial quantity of {e} is not stated", ops, False, unknown
            q[key] += c * p.amount
            ops += 1
            if q[key] < 0:
                return None, f"contradiction: {e} would hold {q[key]} after '{s}'", ops, False, unknown
        if not op.conserves:
            acquired += sum(c for c in (op.alpha, op.beta) if c is not None) * p.amount
    if unknown:
        return None, "frames never taught (effect undetermined): " + ", ".join(" ".join(f) for f in unknown), ops, False, unknown
    # Independent check: conservation of the total under transfers (a different computation).
    verified = sum(q.values()) == initial_total + acquired
    return q, "ok", ops, verified, unknown


def answer(mem: Memory, sentences: List[str], question: str) -> StoryAnswer:
    q, why, ops, verified, unknown = run_story(mem, sentences)
    if q is None:
        return StoryAnswer("abstain", None, why, ops, False, unknown)
    p = parse(question)
    words = set(p.frame)
    item = next((w.rstrip("s") for w in p.frame if (w.rstrip("s") in {k[1] for k in q})), None)
    if item is None:
        return StoryAnswer("abstain", None, "question names no item in the story", ops)
    def val(e: str) -> Optional[Fraction]:
        return q.get((e, item))
    if {"how", "many", "more", "than"} <= words and len(p.entities) == 2:
        a, b = val(p.entities[0]), val(p.entities[1])
        res = None if a is None or b is None else a - b
    elif {"how", "many", "together"} <= words and len(p.entities) == 2:
        a, b = val(p.entities[0]), val(p.entities[1])
        res = None if a is None or b is None else a + b
    elif {"how", "many"} <= words and len(p.entities) == 1:
        res = val(p.entities[0])
    else:
        return StoryAnswer("abstain", None, f"question form {p.frame} not understood", ops)
    if res is None:
        return StoryAnswer("abstain", None, "the story does not fix the quantity asked about", ops)
    if not verified:
        return StoryAnswer("abstain", None, "conservation check failed", ops)
    return StoryAnswer("answer", res, "derived and conservation-checked", ops, True)
