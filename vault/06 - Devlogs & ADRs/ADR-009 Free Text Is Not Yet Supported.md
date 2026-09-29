---
title: "ADR-009: Free Text Is Not Yet Supported, and Verification Cannot See a Misreading"
type: adr
tags:
  - adr
  - hrm-x
  - perception
  - negative-result
status: accepted
date: 2026-09-28
agent: "opencode/big-pickle"
model: "big-pickle"
related:
  - "[[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]"
  - "[[ADR-008 Operator Inference over Structured Situations]]"
  - "[[Task Plan]]"
---

# ADR-009: Free Text Is Not Yet Supported, and Verification Cannot See a Misreading

## Status

Accepted. This ADR records a **negative result and a decision not to act yet**,
not a capability.

## Context

`hrm_x/domains/story.py` opens with a design claim worth repeating, because it is
the claim this ADR takes back:

> Perception (learned): a small transformer tags each token with a role (owner,
> source, destination, amount, item) and the sentence with an event schema. Names
> and numbers are never in its vocabulary; they are copied from the tokens by
> role, so unseen names and magnitudes cost nothing. Surface form ("gives B n",
> "gives n to B", "receives n from A") is convention, which is why this part is
> learned.

The second half of that — *surface form is convention, so learning it should
generalize* — was never tested. `Parser.train` generates from the 12 entries of
`TEMPLATES`, and every story split in `experiments/x_story.py` is worded from
`EVENT_TEXT`, which reuses those same twelve surface forms with different
fillings. Unseen **names** and **numbers** transferred; unseen **wording** never
had to.

`experiments/x_perception.py` measures it properly. The parser is trained on
`TEMPLATES` and nothing else, and its top reading of a held-out sentence is scored
against a hand-written regex+lexicon reference reader that shares no code, no
weights and no templates with it (only the `Frame` output type, so the two can be
compared). Held-out forms differ from training by exactly one thing, grouped by
what changed.

## Measurement

Parser: 40,908 parameters, trained 8 s on 4,000 template sentences.

**Sentence level** — is the top reading exactly right?

| held out | n | schema | parties | item | slots | **exact** |
|---|---|---|---|---|---|---|
| nothing (control, seen form) | 3 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| the verb only | 6 | 0.67 | 0.83 | 0.83 | 0.93 | **0.50** |
| the clause order only | 3 | 0.67 | 0.00 | 0.00 | 0.40 | **0.00** |
| the question word order | 3 | 1.00 | 1.00 | 0.67 | 0.93 | **0.67** |

**Story level** — 120 held-out stories per split, n counted in the JSON:

| split | correct | abstain | wrong | contains a misread | confident wrong on a misread |
|---|---|---|---|---|---|
| events held out | 0.25 | 0.73 | 0.02 | 0.76 (91/120) | 2/91 |
| events + state held out | 0.12 | 0.88 | 0.01 | 0.93 | 1/112 |
| events + question held out | 0.12 | 0.87 | 0.02 | 0.90 | 2/108 |

## Findings

**1. What fails is syntax, not vocabulary.** An unseen verb in the training word
order is read correctly half the time — `transfers`, `gets ... from`, `accepts
... from`, `collects`, `gathers`, `holds` all come out right, because the slots
land where training put them. Change only the clause order and exact accuracy is
**zero**: "Alice hands 3 oranges to Bob ." loses the source entirely,
"Bob is given 3 oranges by Alice ." is read as an *acquire* rather than a
transfer, and "Alice passes 3 oranges over to Bob ." loses both parties. The
tagger has learned template positions, not a compositional reader. Multi-word
verbs fail too ("throws away" is read with the item being "throw"), which is the
same failure wearing a different hat: the item is read by position, and a verb
that occupies two positions moves the item out from under it.

**2. The verifier is blind to all of this, and that is by construction.**
`Story.verify` calls `simulate(hyp.payload)`, which replays the *same* `Reading`
the tagger produced, through a different representation — a plain dict instead of
the e-graph. That is a genuine independent check of the *derivation* and no check
at all of the *perception*. A misread that is structurally complete is therefore
confirmed. Pinned deliberately in `tests/test_x_perception.py`:
"Alice has 10 apples. Alice eats 3 apples." read with `eats` as an acquire derives
13, verifies, and is confidently wrong; the text says 7.

**3. The safe failure rate is real, and it belongs to `reading_defect`, not to
verification.** Only 2 of 91 misread stories came back as a confident wrong answer,
because most misreads are *incomplete* — a missing party, a missing owner — and
`reading_defect` rejects those, so the mind abstains. This was not the prediction:
the expectation going in was that the blind spot would be the dominant failure
mode. It is not, and the reason is the structural guard.

**4. That guard was found incomplete twice, which is the real lesson.** A missing
amount reached `Fraction(None)` and raised `TypeError` inside `Mind.solve`
(2026‑09‑27). A question read with no owner — "Alice has how many apples ?" —
reached `q.owner[0]` in `derive` and raised `IndexError` (2026‑09‑28); left
unguarded, `simulate` would instead have summed over an empty tuple and returned
**0**, a confident wrong answer rather than a crash. An incomplete reading is a
*shape*, and checking one field at a time does not cover the shape space. It is
now pinned per schema, slot by slot, in `tests/test_x_perception.py`.

## Decision

1. **The story domain is not described as free-text capable**, here or in the
   spec. It reads a closed template family. Unseen names, items and magnitudes
   transfer; unseen wording does not.
2. **No second perception path is added to the mind yet.** The reference reader
   stays in `experiments/x_perception.py` as a scoring instrument. Wiring a
   hand-written reader into `Story` would make verification independent, which is
   the obvious next step, but it would also put a second non-learned reader in the
   loop and quietly turn the domain into a hybrid — an architecture decision that
   deserves its own evidence and its own falsifier, not a side effect of a
   measurement script.
3. **The safe-failure property is credited to `reading_defect`**, in the docs and
   in the reasoning, instead of to verification. Any future claim that the story
   domain "fails safe" must name the guard.

## Consequences

- Spec §26 falsifier (c) — "on free text, perception errors dominate" — is
  **confirmed**, and now quantified rather than argued.
- The most promising fix is not a bigger transformer. It is training data with
  varied word order, so positions stop being the only cue; a compositional reader
  (a dependency parse) is the alternative and is a much larger change. Either way
  the gate is *re-measured with `x_perception.py`*, which is now a working
  instrument: its reference reader and its held-out sets are unit-tested, because
  during development a spacing bug in the reference produced a clean-looking
  `misread = 0.00` and a wrong question label produced a bogus `misread = 1.00`.
  Both were caught only by asserting against hand-written expectations.
- Event-schema induction (Task Plan gate 3) stays open and is a separate problem:
  the schema head scored 0.67 even where roles failed, so the sentence classifier
  is the healthier half of the perception stack.

## Attribution

- Experiment: `experiments/x_perception.py`, result `experiments/results/x_perception.json`
- Tests: `tests/test_x_perception.py` (23 tests)
- Guard fix: `hrm_x/domains/story.py` `reading_defect` (question owner)
- opencode/big-pickle (big-pickle), 2026‑09‑28
