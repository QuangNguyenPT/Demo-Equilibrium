---
title: "HRM-X Specification v0.4 — Operator Inference over Structured Situations"
type: specification
tags:
  - architecture
  - specification
  - mathematics
  - hrm-x
  - cognition
status: implemented-poc
date: 2026-09-28
supersedes: "[[HRM-X Specification v0.3 - Understanding by Construction]]"
related:
  - "[[ADR-008 Operator Inference over Structured Situations]]"
  - "[[ADR-007 Grounded Algebra and Executable Concepts]]"
  - "[[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]]"
code: [hrm_x/workspace.py, hrm_x/rules.py, hrm_x/induction.py, hrm_x/mind.py, hrm_x/writer.py, hrm_x/domains/arith.py, hrm_x/domains/story.py, hrm_x/domains/grid.py, hrm_x/native/, experiments/]
---

# HRM-X v0.4 — Operator Inference over Structured Situations

> **Thesis.** A task is a *situation* plus an unknown or known *operator* plus a *question*.
> Understanding means three things: represent the situation so that what follows from it is
> decidable; hold a posterior over operators (programs) that explain the evidence; and derive
> the answer exactly, checking it in an independent representation. All of it runs under one
> control law that spends computation only where its expected value exceeds its cost. Learning
> is used where knowledge is conventional (words → roles) or statistical (which programs are
> usually right); exact algorithms are used where exactness is available.

Claim labels used below: **[E]** established in the literature, **[R]** empirical result in this
repository (with its file), **[H]** hypothesis, **[U]** unresolved.

---

## 1. Research diagnosis

**What the evidence says about the previous designs**

| # | Finding | Label | Source |
|---|---|---|---|
| 1 | The trained HRM-EQB learned a magnitude-like code for integers: similarity tracks \|a\|−\|b\| (r = 0.50) more than a−b (r = 0.29) | [R] | re-measured 2026-09-28 |
| 2 | Its additive effect is real but neighbourhood-level. Analogy retrieval hits a+b exactly 6% of the time (null 0.6%), with mean miss 5.3 (null 33.9). Under a relative-error metric there is no effect (1.39 vs null 1.31) | [R] | re-measured |
| 3 | Every HRM-EQB solve converges in about 5–7 cycles whatever the task (contraction rate 0.30). Termination is guaranteed; allocating compute is not | [R] | ADR-006 |
| 4 | HRM's neural H/L hierarchy adds little on ARC over a same-size transformer; the outer refinement loop and training on the evaluation tasks drive the score, and the program stays implicit | [E] | ARC Prize HRM analysis, Aug 2025 |
| 5 | A single 2-layer network recursing beats HRM on small puzzles; separate H/L networks are *worse* (82.4 vs 87.4 on Sudoku-Extreme) | [E] | TRM, Jolicoeur-Martineau 2025 |
| 6 | At the 1B scale on language, HRM's dual timescale beats Transformer, Looped Transformer, RINS and TRM under matched FLOPs. The paper argues for separating a compact reasoning core from factual storage | [E] | HRM-Text, arXiv 2605.20613 (in `docs/`) |
| 7 | Transformers extrapolate arithmetic only when task structure is built in (Abacus, position coupling: about 6× the training length); without it they fail beyond the training length | [E] | McLeish 2024; Cho 2024 |
| 8 | v0.3's own proposals had defects. Unfolding a definition costs O(n) (a missing abstraction). A learned MLP "compiler" can hide memorization. There was no representation of situations, entities or events | [R] | external audit, accepted |

**Conclusion.** Hierarchical *neural* recurrence is task-dependent: evidence for it on language
at scale (#6), against it on small symbolic puzzles (#4, #5). Neither gives exact, extrapolating
operations (#1–3, #7). What was missing is a representation where "follows from" is a
decidable relation, and a controller for which computation is a decision.

## 2. Functional understanding (operational definition)

A system *understands* an operator f on a domain D, relative to a family of tests, when:

| | Criterion | Measured by |
|---|---|---|
| U1 | **Extrapolation.** Accuracy does not fall outside the training range of magnitudes, sizes or lengths | E1, E5, E7 |
| U2 | **Composition.** f combines with other operators into structures never seen as tasks | E2, E6 |
| U3 | **Counterfactual consistency.** Changing a premise changes the answer by exactly the amount the mechanism implies | E8 |
| U4 | **Binding and role.** Renaming fillers changes nothing; swapping roles changes the answer the right way | E3, E4, E11 |
| U5 | **Knowing what it does not know.** Missing, contradictory or out-of-domain inputs produce "I don't know" with a reason, not a guess | E10, story contradictions |
| U6 | **Efficiency from abstraction.** Discovered structure shortens derivations | E12 |

"Understanding" is not claimed from similarity, clustering, probes or accuracy alone. Each
criterion is a behaviour under intervention.

## 3. Fundamental requirements (derived from §2)

- **R1 Exact identity and equality**, separate from similarity. U4 needs identity to survive
  renaming; U3 needs equality to propagate. *Similarity may propose; it must never assert.*
- **R2 Explicit state over time.** Needed for U3 and for multi-step events.
- **R3 Operators as inspectable objects** (programs), with a posterior. Needed for U1 and U2;
  the operator must be separable from its training data.
- **R4 Exact execution where the domain admits it.** U1 is impossible if arithmetic is approximate.
- **R5 A verifier that is structurally different** from the predictor. A self-check that repeats
  the same computation certifies nothing.
- **R6 Computation as a decision.** Stop, derive, verify, ask or propose, chosen by expected value
  minus cost. Needed for U5 and U6, and for the easy/hard asymmetry.
- **R7 Provenance.** Every conclusion must name the premises it rests on, so a contradiction
  can blame something (self-refinement as revision, not regeneration).

## 4. Competing architectures

| | D1 End-to-end recurrent net (HRM/TRM/HRM-EQB) | D2 Pure program synthesis (hrm/, DreamCoder w/o neural) | D3 Single vector space (VSA/FHRR everything, v0.3 direction) | **D4 Operator inference over structured situations (selected)** |
|---|---|---|---|---|
| Representational capacity | high, implicit | exact, limited to the DSL | high, approximate (crosstalk) | exact situations + DSL programs |
| Systematic generalization | weak off-distribution [R, E] | strong inside the DSL | within noise bounds | strong inside the DSL [R] |
| Exactness | none | exact | bounded error (A3 of v0.3) | exact (rationals, e-graph) |
| Interpretability | low | high | medium | high (traces, provenance) |
| Training difficulty | high (data-hungry) | none / search | medium | low (small parser; search) |
| Compute | fixed per input | exponential search | cheap per op | conditional (VOC) [R] |
| Language | strong at scale [E] | none | weak | templated now; learned Reader |
| Vision/grids | transductive [E] | good with objects | weak | good with objects [R] |
| Math | fails to extrapolate [R] | exact | approximate | exact [R] |
| Adaptive compute | ACT (learned halt) | none | none | value of computation [R] |
| Engineering | moderate | moderate | moderate | moderate (+ native core) |

**Why D4.** It is the only design meeting R1–R7 at once. The cost is that the hypothesis
languages are given, not grown (§26). D1 stays the right tool for *perception of language*
(HRM-Text); D3's FHRR algebra stays useful as an *independent verification* representation (§16).

## 5. The architecture

```
 input (text / grid / expression)
   │  perception: learned where conventional (tagger), exact where not (objects, parser)
   ▼
 WORKSPACE  = e-graph situation  S : typed terms, equality ≡ (congruence-closed), provenance
   │                       ▲
   │   L-level: saturate rules to least fixpoint (exact), extract cheapest term, execute
   ▼                       │
 BELIEF over operators  H = {(program/reading h, weight w_h, derived y_h, verified v_h)}
   ▲   │
   │   └─ H-level controller: argmax_a VOC(a) over {propose, derive-group, verify, ask, stop}
   │
 VERIFIER (different representation / code path) ──► accept, or refute and name the culprits
   │
 MEMORY: semantic (version spaces, laws, event schemas, parser .pt, search prior),
         episodic (tasks, traces), modulators (reliability, volatility)
   │
 SLEEP (self-loop): curiosity-driven study → law discovery → re-test on imagined values
   ▼
 WRITER: answer + explanation read off the trace (faithful by construction)
```

**Correspondence with the hand-drawn architecture (2026-09-28).** Each box in the sketch
has a place and a defined computation.

| Sketch | HRM-X v0.4 | Module |
|---|---|---|
| Input layer (train, chat, …) | tasks: `ArithTask`, `StoryTask`, `GridTask` | `hrm_x/domains/*` |
| Embedding (for chat) | learned token → role tagger (the only gradient-trained part) | `story.Parser` |
| Embedding (for train) | behaviour fingerprints of programs (observational equivalence) | `induction.behaviour_classes` |
| **Cognitive matrix** | the workspace e-graph: situation + equalities + provenance | `workspace.py` |
| **Main EQB matrix + transformer** | L-level: rule saturation to a **least fixpoint** (the meaningful "equilibrium") + extraction; transformer only in perception | `rules.py`, `arith._plan` |
| train & update / fix / refine → weight (.pt) → use info to solve | semantic memory (version spaces, laws, prior) + parser `.pt`; `Mind.save/load` | `mind.py`, `story.Parser.save` |
| **H level → think, output once sure** | VOC controller: answers when no action is worth its cost and U > 0 | `Mind.solve` |
| low level → output text | writer rendering the trace | `writer.py` |
| self loop (study / rethink) | `Mind.sleep`: curiosity study, law consolidation, law re-test | `arith.sleep` |
| visualize matrix (imagination) → train | re-testing laws on imagined values; dreamed task streams for the search prior (grid prior) | `arith._holds`, `grid._prior` |

## 6. Mathematical representation

**Derived representation (answer to the central question).** Not everything should be a vector
in one space.

| Kind | Represented as | Why this and not a vector |
|---|---|---|
| identity | e-class id (entity constant `const(E, "alice")`) | must survive renaming exactly (U4); a vector changes identity when the name changes |
| type | sort of a constant + e-class analysis (e.g. `is_nat`) | law domains need exact sorts (a law tested on ℕ×ℕ must not fire on ℚ) |
| value | exact constant (Fraction, symbol) | exactness condition for U1 |
| role | argument position of a term: `qty(t, owner, item)` | role is structure, not content |
| relation | function symbol; a fact is `term ≡ True` | composable by congruence |
| context | time index `t` and hypothesis worlds (assumption labels on every union) | contradictions must be attributable |
| state | the set of `qty(t, ·, ·)` classes with constants at time t | frame axiom keeps unaffected facts |
| event | transition rule `S_t → S_{t+1}` (effects + frame) | counterfactual = edit an event, re-derive |
| transformation / concept | program p in a typed DSL; behaviour class [p] | inspectable, executable, composable |
| goal | query term (a class whose constant is sought) | |
| uncertainty | posterior weights over readings/programs; reliability ρ; volatility v | |
| memory | version spaces, laws, event schemas, parser weights, search prior | |
| computation | trace of H-actions and L-ops | |
| similarity (learned) | parser embeddings; behaviour fingerprints | *proposal only*: never creates an equality |

**Workspace (e-graph).** Nodes are e-nodes `(op, child classes, payload)`. Classes form a
union-find partition. The invariants are tested in `tests/test_x_core.py` and `test_x_native.py`:
- **I1 congruence:** a ≡ b ⇒ f(a) ≡ f(b), enforced by `rebuild`.
- **I2 constant uniqueness:** a class holds at most one constant. A union that would merge two
  different constants is *refused* and recorded as a `Conflict`.
- **I3 provenance:** every union carries `Why(rule, assumptions, premises)`. `explain(a, b)`
  walks the unique path in the proof forest, expanding congruence edges into their arguments
  and rule firings into the premises they used. The culprit set of a conflict is the union of
  the assumptions on that path.

"Follows from": B follows from premises Γ under rules R iff B's sides share a class in
lfp(T_{Γ,R}), the least fixpoint of the monotone immediate-consequence operator. This is
decidable when saturation terminates, which the budget guarantees. **Similar ≠ follows:** no
code path turns a similarity into a union.

**Exact vs approximate.** Workspace, rules, arithmetic (Python `Fraction`, arbitrary precision)
and grid execution are exact. The approximate parts are the parser (a learned classifier),
learned priors (search order) and residue-phasor decoding in the verifier. That decoding is
exact under the A3 noise bound of v0.3 and was checked at σ ≤ 0.8.

## 7. Cognitive state and situation model

The mind's state at H-cycle t:

  s_t = (S_t, H_t, ρ_t, v_t, c_t)

- S_t: the workspace.
- H_t: the hypotheses with weights w_h, derived answers y_h and verification v_h.
- ρ_t: learned reliability.
- v_t: volatility.
- c_t: operations spent.

**Situation example (story).** "Alice has 4 apples. Bob has 2 apples. Alice gives Bob 3 apples.
Bob buys 5 apples." becomes:
```
qty(0,alice,apple) ≡ 4 [s0]      qty(0,bob,apple) ≡ 2 [s1]
qty(1,alice,apple) ≡ sub(qty(0,alice,apple), 3) [s2]   qty(1,bob,apple) ≡ add(qty(0,bob,apple), 3) [s2]
qty(2,bob,apple)   ≡ add(qty(1,bob,apple), 5) [s3]     qty(2,alice,apple) ≡ qty(1,alice,apple) [frame s3]
```
Constant folding saturates this to `qty(2,bob,apple) ≡ 10`. With "Alice has 2 apples" instead,
`qty(1,alice,apple)` would be ≡ −1. The non-negativity check marks this a contradiction, and
`explain` blames exactly **{s0, s3}** in the test story (tests: `test_contradiction_is_refused…`).
Context changes interpretation without touching the concept: "gives" and "receives" map to the
same TRANSFER schema with roles swapped by the parser.

## 8. Inference and abstraction

- **Deduction / equivalence:** rule saturation with guards, the L-level.
- **Variable substitution:** e-matching modulo equality.
- **Constraint propagation:** conflicts on constant clash; domain constraints (non-negativity,
  demo replay).
- **Hypothesis comparison:** the posterior over behaviour classes of programs,
  p(h | D) ∝ Σ_{p ∈ h} 2^{−|p|} · 1[p ⊨ D].
- **Abstraction (law discovery, QuickSpec-style [E]):**
  1. Enumerate terms up to size k over {f, add} with variables x, y ∈ ℕ and z ∈ ℚ.
  2. Group terms by their values on n probes.
  3. Order candidate equations by size, then by generality.
  4. Accept a candidate only if congruence closure over instances of the laws already accepted
     does not already derive it.

  Guarantee: a law that is false on a fraction ε of its domain survives n probes with
  probability at most (1 − ε)^n. Laws carry their tested domain as guards.
- **Using abstractions:** rewrite the question with the laws and place-value notation
  (hypothesis-checked), then extract the cheapest term under the cost semiring. The cost of
  `f(a, b)` is the loop count of the concept's program.
- **Counterfactuals:** edit an event's amount and re-derive. The answer changes by exactly the
  implied amount (E8: 1.000).

## 9. Mathematics mechanism (demonstration A)

**Substrate (given, exact):** rationals, `add`, `iter(n, body, init)` for natural n. The concept
`×` is *not* given.

**Evidence** 2×2=4, 2×3=6. Enumerating programs up to size 6 and grouping by behaviour leaves
four consistent classes:
- `y + y` (2^-3; the simplest);
- repeat x times [a := a + y] from 0;
- repeat y times [a := a + x] from 0;
- repeat y times [a := a + 1] from y.

They agree on the evidence and disagree on 10 × 3.1: 6.2, 31, undefined, undefined.

**Identifiability [R].** The two examples do *not* identify multiplication; the simpler `y + y`
fits them. With no teacher, the mind answers 6.2. That is the Occam-correct answer given the
evidence (`test_two_examples_do_not_identify…`).

**With a teacher**, the mind first derives each reading (6.2, 31, undefined, undefined). It then
asks the one question worth its cost, 1 × 1: the `y + y` readings say 2 and the counting
readings say 1. The teacher answers 1, both doubling readings are refuted, and the reading
that counts by y is undefined for 10 × 3.1, so the answer mass rests on the count-by-x program
(10 additions of 31/10 = **31**), which is verified in residue-phasor space. Trace: propose →
derive (6.2) → verify → derive (31) → derive (undefined × 2) → ask → verify; 7 cycles, 4,257 ops.

**Four ways to multiply, compared (E1, `experiments/results/x_arith.json`):**

| Method | in range | large integers | large decimals | 10 × 3.1 |
|---|---|---|---|---|
| pattern interpolation (1-NN, 2000 pairs) | 0.41 | 0 | 0 | 0 |
| neural operator (MLP, 2000 pairs) | 0 (rel. err 1e-3) | 0 (rel. err 1.0) | 0 | 0 |
| neural operator with log bias (log-MLP) | 0 (9e-4) | 0 (1.0) | 0 (0.96) | 0 |
| hybrid HRM-X (2 examples + 1 question) | **1.000** | **1.000** | **1.000** | **1** |
| exact hand-coded (upper bound) | 1.0 | 1.0 | 1.0 | 1 |

Nested expressions at depths 1–3 score 1.000 (E2).

**Where exact computation happens:** `domains/arith.run` (Fraction), inside the extracted plan.

**Abstraction reduces work (E12):**

| Question | ops, definition only | ops, with laws |
|---|---|---|
| 100 × 3.1 | 115 | 43 |
| 1000 × 3.1 | 1,021 | 192 |
| 10000 × 3.1 | 10,020 | 411 |
| 100000 × 3.1 | *refused as not worth the cost* | 779 |

(The with-laws column includes the rewrite work: saturation, the place-value rule and the
exact folding of cheap sub-applications that planning does before extraction.)

With laws, 1000 × 3.1 is computed as `f(10, f(10, f(10, 3.1)))`, using place value plus the
discovered associativity. Laws discovered in 1.6 s (term size ≤ 7) include:
- commutativity of `f` on ℕ×ℕ;
- associativity `f(x, f(y, z)) = f(f(x, y), z)` with z ∈ ℚ;
- both distributivities.

## 10. Visual reasoning (demonstration C)

Pipeline: pixels → objects (4-connected components; exact) → a program over objects → rendered
output.
- A step is (selector ∈ {all, colour c, largest, smallest}, action ∈ {keep, delete, recolour,
  move, flip}) or a whole-grid symmetry.
- Hypotheses are the programs that replay every demo, grouped by their prediction for the test
  (observational equivalence on the question).
- Prior: 2^{−|p|} × a usage prior learned from solved tasks. Description length is
  all = 1 < colour = largest = smallest = 2. *This ordering was set after the seed-0 failures
  and re-validated on the unseen seed 1.*
- Verification is an applicability check on the new input:
  - every step acts in at least one demo (evidence for it);
  - no selector is ambiguous (a tie for "largest");
  - no step moves an object off the grid or makes objects collide.

**Results (independent generator, confound-controlled; seeds 0 / 1):**

| Suite | HRM-X | wrong | pixel LUT | no objects | colours permuted |
|---|---|---|---|---|---|
| single rule, same size | 0.984 / 1.000 | 0 | 0.29 / 0.27 | 0.52 | = HRM-X |
| single rule, new arrangement (grid 11–14, 4–6 objects) | 0.984 / 1.000 | 0 | 0.25 / 0.27 | 0.52 | = HRM-X |
| two-step composition, never a task | 0.931 / 0.833 | 0.034 | 0.03 | 0.00 | = HRM-X |

The residual failures are `delete_smallest + move_all`. Both orders fit every demo, but on the
test a move can make objects merge, which changes which one is smallest. That is genuine
ambiguity, and the mind abstains.

## 11. Memory

| Store | Holds | Retrieval | Written by |
|---|---|---|---|
| working | workspace S_t, belief H_t | direct | the loop |
| episodic | (domain, task, Result with trace) | by task | every solve |
| semantic | concept **version spaces** (surviving programs + weights + evidence); laws keyed by (symbol, program); event schemas; grid usage prior | by symbol / structure (exact keys), never by similarity alone | `remember`, `sleep` |
| parameters | parser weights (40,908, `.pt`) | — | gradient descent |
| meta | reliability ρ (Beta), volatility v | by key | verification, feedback |

A concept is stored as a version space, not a single winner, because two readings can agree on
all evidence yet differ elsewhere (count-by-x vs count-by-y). Keeping both is what makes
curiosity possible.

## 12. Transformer role

Used for **conventional mapping**: surface form to roles and schema ("gives B n" / "gives n to
B" / "receives n from A" all mean TRANSFER). It is a 2-layer, d = 48 encoder that tags tokens.
Names and numbers are never in its vocabulary; they are copied by role. That is why unseen names
and magnitudes cost nothing (E3: 1.000; E5: large numbers 1.000).

**Not used for reasoning.** The end-to-end transformer baseline (3 layers, 120k parameters)
reached 0.65 in-distribution with 16k stories. It scored 0.00 on large numbers and 0.04 on long
chains, and never refused a contradiction (`x_story_16k.json`). **[H]** At scale, the
Reader/Writer should be an HRM-Text-style dual-timescale transformer (evidence #6).

## 13. HRM / hierarchical recurrence role

- **H** makes a few value-based decisions per task: propose, derive-group, verify, ask, stop.
- **L** fires rules to a least fixpoint and executes the extracted plan.

A cycle is a meaningful state transition, not a fixed iteration. Recurrence here is *iterative
revision of a belief*, which depth alone cannot express, because the number of useful revisions
depends on the evidence.

**[R] Test:** a flat controller that decided at every L-operation would take 500–4,000
decisions per task, against 2.5–3.5 H-cycles (§14 table).

**[E]** Neural H/L separation is not what matters for small puzzles (#4, #5). HRM-Text (#6)
suggests it matters for language at scale. This design therefore keeps hierarchy as a
*computational* separation. It does not claim that separate H/L networks help reasoning.

## 14. Adaptive-compute controller

The probability that answer y is correct is

  q(y) = Σ_h (w_h / Σw) · r_h · [y_h = y],   where r_h = 1 if verified, else ρ.

The value of the state is

  U = max(0, (1 + L) max_y q(y) − L)

(+1 for a right answer, −L for a wrong one, 0 for abstaining). Each cycle the controller takes

  a* = argmax_a { E[U | s, a] − U − λ · cost(a) }

and stops when that value is ≤ 0. This is myopic value of computation [E, Russell & Wefald 1991]
with two corrections found in this work [R]:
- **Derive-group macro-action.** Hypotheses predicted to agree are derived together. Otherwise
  two 50% hypotheses that agree look worthless one at a time.
- **Plan-then-cost.** Rewrite and extract before estimating execution cost. Otherwise the
  controller refuses computations that laws make cheap. Before extraction, applications whose
  arguments are known and whose cost is ≤ 200 are computed exactly, so loop counts that other
  plans depend on are known (a guessed count once chose a 99,070-step plan over a 415-step one).
- **Answerability [R, stated assumption].** A question that is asked presupposes an answer.
  Readings under which it is undefined leave the answer mass: q is normalized over the readings
  that answer. If none answers, the mind abstains and says so.
- **Prediction without derivation.** Grouping needs a cheap guess of each reading's answer.
  Evaluating the question would be a derivation without the laws and erase their savings, so
  the guess is abstract interpretation over kinds {natural, non-natural, undefined}. A
  sub-application's kind is the kind the reading produced on every probe of its argument kinds;
  sums of literals are computed. An undefined root predicts "undefined". Otherwise readings are
  grouped by their fingerprint on the probes of the question's kinds. Readings grouped this way
  are *expected*, not asserted, to agree [H]. The answer comes only from derivation and is
  checked by the verifier.

| Question type | Value computed |
|---|---|
| verify | ρ·U(pass) + (1−ρ)·U(fail) − U (it has value only if failing changes the action) |
| ask | expected U over the teacher's possible answers, grouped by hypothesis predictions |
| propose | learned P(success at that level) − λ·cost |

**E9 results (`x_control.json`; 45 easy and 40 hard mixed tasks from arithmetic, stories and grids):**

| Policy | easy acc / ops | hard acc / ops | wrong |
|---|---|---|---|
| VOC (this controller) | 1.000 / 248 | 0.975 / 4,346 | 0 |
| greedy (derive the best reading, answer) | 1.000 / 134 | 0.750 / 1,700 | 0 |
| fixed (derive and verify everything) | 1.000 / 3,836 | 1.000 / 10,182 | 0 |
| no verifier | 1.000 / 139 | 0.975 / 4,071 | 0 |
| no questions | 1.000 / 248 | 0.975 / 4,341 | 0 |

VOC spends 15× fewer ops than the fixed policy on easy tasks and 2.3× fewer on hard ones, for
one abstention in 40 hard tasks. Greedy is cheaper but loses a quarter of the hard tasks.
The verifier ablation shows no difference *here* because no fault is injected; E10 is where it
matters.

### Neuromodulation: the "hormone" layout, made computational

The sketch's hormone system was a layout, not a mechanism. v0.4 keeps a signal only where it
has a defined job in the controller above, following the computational reading of
neuromodulators [E, Doya 2002; Yu & Dayan 2005]. No other hormones are added.

| Signal | Role here | Computation | Status |
|---|---|---|---|
| ACh-like: expected uncertainty | reliability ρ of unverified derivations | Beta mean of verify passes, per domain | [R] used in q and VOC |
| DA-like: prediction error | outcome feedback | `Mind.feedback`: Beta update of ρ on external right/wrong | [R] |
| NE-like: unexpected uncertainty | volatility v | surprise −log p(outcome) traced by an EMA; v = excess over the mind's own expected surprise H(ρ), clipped to [0, 1]; on each observation old counts are discounted toward the prior by 1 − 0.6·v | [R] tested (below) |
| 5-HT-like: patience | cost of computation λ | fixed constant | [U] no learning rule proposed; left fixed on purpose |

**Why volatility is needed [R].** With a stale ρ ≈ 0.99, verification looks worthless, so a
fault that enters memory is never caught. When a false law is planted after 100 tasks, with
verification expensive, answers go wrong until surprise re-opens verification. The results are
2 wrong answers with the signal on and 6 with it off (`x_metacog.json`).

### Curiosity

Curiosity is *expected information gain* over a concept's version space during sleep, weighted
by how often the concept is used. The mind asks the teacher the question whose answer
distribution over the surviving readings has the highest entropy, and stops when the readings
no longer disagree on the question pool. For `×` seeded with two examples, curious study
identifies the concept with 1.0 question against 4.8 for random questions. Later unaided
accuracy is 0.86, against 0.71 after random study and 0.67 with no study; with no study, a third
of the answers are wrong (`x_metacog.json`).

## 15. Self-refinement

A candidate that fails is revised by blame, not regenerated:
- **Conflicts:** the explanation's assumption set names the sentences, laws or parse choices
  responsible. The refuted reading gets weight 0 and the others keep their mass.
- **Arithmetic:** the residue verifier failing on a derivation blames the laws on its rewrite
  path. A false law was blamed and refused (`test_residue_verifier_catches_a_false_law…`).
- **Grids:** a failed applicability check names the step and the demo.

Multiple hypotheses coexist until evidence (a question, a check, a constraint) separates them.

## 16. Verification

Each check is structurally different from the predictor:

| Domain | Predictor | Verifier | Different in |
|---|---|---|---|
| arithmetic | law-rewritten plan executed with Fractions | the *definition* evaluated in residue-phasor space, with repeated addition as exponentiation (A2) | representation, algorithm, and it ignores laws |
| story | e-graph saturation | plain dictionary replay + conservation of items | code path, representation |
| grid | program replay | applicability conditions (support, ambiguity, bounds, collisions) | checks conditions, not outputs |

**Limit [R].** The story verifier replays the *same reading*, so it cannot catch perception
errors. It catches reasoning faults; parse faults are caught only if they produce a
contradiction (E10).

## 17. Learning objectives and curriculum

| Problem | Method | Why |
|---|---|---|
| perception (roles, schemas) | supervised gradient descent, CE on tags and schema | conventional knowledge |
| concepts / operators | Bayesian program induction (enumerate, observational equivalence, posterior) + active questions | exactness, identifiability |
| relations / events | given schemas (TRANSFER, ACQUIRE, CONSUME); parser learns to map to them | [U] schema learning |
| laws (abstractions) | theory exploration (testing + congruence-closure pruning) | shortens derivations |
| search prior | counts of actions in solved programs | reduces search 50 → 30 programs (E12) |
| controller | Beta estimates of verification reliability and proposal success; volatility | computation as a decision |
| execution | none: exact algorithms | a guarantee beats a learned approximation |

**Curriculum:** simple examples → questions → sleep (curiosity, laws) → harder tasks reuse
memory. The measured cost is 7 cycles / 4,257 ops to learn `×`, then 4 cycles / 58 ops for the
next question (6 × 7: both surviving readings are derived and verified).

## 18. Anti-memorization evaluation

- Generators are independent of the model (their own scene code, flood fill, transformations,
  story simulator; truth from Fraction ops).
- Splits include:
  - unseen magnitudes (to 10⁴) and decimals;
  - unseen names and items;
  - longer event chains;
  - larger grids with more objects;
  - compositions that were never tasks;
  - colour permutation;
  - role-swap minimal pairs;
  - counterfactual edits;
  - contradictions and missing information.
- Confound control: grid demos are regenerated until the property selector is identified.
- Baselines are strong enough to learn their training distribution; the 16k-story baseline
  reached 0.65 in-distribution.

## 19. Complexity

| Part | Cost |
|---|---|
| e-graph add / union / find | ~α(n) amortized; rebuild O(n) per pass |
| e-matching | O(classes × pattern) |
| saturation | bounded by a firing budget |
| extraction | O(iterations × e-nodes) |
| program induction | O(\|programs ≤ k\| × \|probes\|): size ≤ 6 gives about 4k evaluations |
| law discovery | about 0.41M evaluations at term size 7; pruning O(accepted × classes) |
| execution | the extracted plan's cost, e.g. 10·log₁₀(n) with place value vs n naive |
| grid induction | about 400 single steps; beam 24 × 400 two-step programs |
| controller per cycle | O(\|H\| × \|questions\|) |

Native backend (§23): grid induction 3.8× faster, arithmetic 1.5×. The arithmetic ceiling is
Amdahl's law applied to exact Fraction arithmetic, which is kept exact on purpose.

## 20. Failure modes (observed or predicted)

1. **Hypothesis language limits [R].** No program in the arithmetic DSL multiplies with a
   fractional *count*, so 2.5 × 4 is "undefined for me" and the mind abstains. [U] Extending a
   concept beyond its defining domain (e.g. by a law used outside its tested domain) is open.
2. **Perception boundary [R].** An unseen sentence frame (passive voice) gives a
   mis-parse → contradiction → abstain. Honest, but not understanding.
3. **Verifier blind spot [R].** A reading-level error in perception passes replay (E10).
4. **Myopic VOC [R].** One-step lookahead misses multi-step value; patched with macro-actions,
   not solved in general.
   Related **[H]:** two readings that agree on every probe of the question's kinds may still
   differ on the question. The controller then derives both, finds different answers, and
   abstains or asks. That costs work, but it is not a wrong answer.
5. **Genuine ambiguity [R].** Order-dependent compositions; the mind abstains.
6. **Probe domain [R].** Laws involving two rationals are not found, because probes use one
   rational variable.
7. **DSL given, not grown [U].** Most dangerous (§26).

## 21–22. Minimal prototype and the main loop (exact pseudocode)

```
solve(domain, task):
  H ← ∅; ops ← 0
  loop (≤ max_cycles):
    q, U ← belief(H, ρ)
    options ← []
    if mass(H) = 0 and not exhausted: options += propose(level)    value p_found − λ·cost
    groups ← group underived h by domain.predict(h)                 (kind analogy; if ≥ 2 alive)
    options += derive(group)                                         value U(q ⊕ group·ρ | answering mass) − U − λ·cost
    options += verify(h) for derived, unverified h                   ρU(pass) + (1−ρ)U(fail) − U − λ·c_v
    options += ask(x) for teacher-answerable x                       E_o[U | o] − U − λ·(c_ask + c)
    a ← argmax VOC;  if VOC(a) ≤ 0: break
    execute a (L-level work inside derive / verify); ops += spent; trace += a
  answer argmax_y q(y) if U > 0 else abstain(reason from notes, culprits)
  domain.remember(task, result)          # version space, usage prior
  return result                          # writer.explain(result) renders it

sleep(domain, teacher):                  # self-loop
  for concept by (ambiguity × uses): study(curious) with the teacher
  consolidate laws; re-test laws on imagined values; retract failures
```

## 23. Repository and module architecture

```
hrm_x/
  workspace.py   EGraphAPI (shared semantics) + Workspace (Python reference) + make_workspace()
  rules.py       Rule, e-matching (Python or native), saturate (L-level), close (pruning)
  induction.py   behaviour classes, posterior, law discovery (QuickSpec)
  mind.py        belief, VOC controller (H-level), Modulators, sleep, save/load
  writer.py      trace -> answer + explanation
  domains/       arith.py (concepts, laws, residue verifier), story.py (parser + situation), grid.py (objects + programs)
  native/        include/hrmx.h (C ABI) · src/grid.cpp · src/egraph.cpp · __init__.py (ctypes, NativeWorkspace)
experiments/     x_arith, x_story, x_grid, x_control, x_metacog, x_native (+ results/*.json)
tests/           test_x_core, test_x_domains, test_x_native (differential), test_x_mind
scripts/build_native.ps1
```

**C++ backend.** Python is the blueprint and reference semantics; C++ executes the profiled
hot paths behind a pure C ABI with zero-copy NumPy buffers (the conventions of ADR-001 and the
C-ABI note).

| Workload | Python hot path | Share | What moved to C++ | Speedup |
|---|---|---|---|---|
| grid induction | `objects()` | 67% | components + object step | 3.8× |
| arithmetic | e-graph `nodes` / `_match` / `find` | ~50% | union-find, hash-cons, rebuild, e-matching, explanations | 1.5× |

- **Kept in Python on purpose:** exact rational arithmetic (arbitrary precision is the
  guarantee) and guards and builders (domain callables).
- **Parity** is tested on 200 component labelings; 60 grids × every step × 3 strictness levels;
  40 random e-graph histories (partition, values, node order, op counts, conflicts,
  explanations); e-matching; and whole arithmetic solves under both backends.
- **Correctness paid off:** porting surfaced two latent reference bugs, now fixed and
  regression-tested:
  - union-by-size swapped the roots but not the constants, dropping a value;
  - a congruence conflict did not name its argument premises.

## 24. Implementation plan (next gates)

| Gate | Pass condition |
|---|---|
| G1 mathematical | invariants I1–I3 and law error bound (done) |
| G2 representation | information recoverable: explanations and culprits exact (done) |
| G3 capability | four domains' tasks in the PoC (done) |
| G4 generalization | unseen magnitudes, compositions, arrangements (done: §9, §10, §12) |
| G5 anti-shortcut | independent generators, confound control, strong baselines (done) |
| G6 efficiency | laws and search prior reduce work (done) |
| **next** | learned Reader/Writer on free text (HRM-Text-style); library learning that *grows* the DSL (DreamCoder compression) on ARC-AGI training tasks, measured on evaluation; schema induction for events |

## 25. Experiments and ablations (all recorded in `experiments/results/`)

| # | Hypothesis | Result |
|---|---|---|
| E1 | exact extrapolation of an induced operator | HRM-X 1.000 on every split; kNN 0.41 → 0; MLPs 0 |
| E2 | composition of learned × with + | 1.000 at depths 1–3 |
| E3 | variable binding (unseen names and items) | 1.000 (baseline 0.60 at 16k) |
| E4 | role-swap minimal pairs | 1.000 |
| E5 | novel relation combinations (3 people, 4–6 events) | 0.993–1.000, no wrong answers (baseline 0.04) |
| E6 | rule composition (grid, never a task) | 0.931 / 0.833 (pixel 0.03, no-objects 0.00) |
| E7 | structural transformation on new arrangements | 0.984 / 1.000 (pixel 0.25) |
| E8 | counterfactual edits | 1.000 (89–93 stories) |
| E9 | adaptive compute | VOC 1.000 / 248 ops (easy), 0.975 / 4,346 (hard) vs fixed 3,836 / 10,182; greedy 0.75 on hard (§14) |
| E10 | verification under injected faults | false law: 0 wrong with the verifier, 9 without; swapped transfer roles (perception): 10 wrong either way (blind spot, §16) |
| E11 | representation change (colour permutation) | identical accuracy |
| E12 | efficiency from abstraction | 5–24× fewer ops with laws at 10³–10⁴; 10⁵ computed only with laws; search rank 47–50 → 30 with learned prior |
| C | curiosity (expected information gain vs random study) | × identified with 1.0 question (random 4.8); later accuracy 0.86 vs 0.71 (no study 0.67, 33% wrong) |
| V | volatility (NE-like) under a regime change | 2 wrong answers before verification resumes vs 6 without |
| N | native backend parity and speed | identical results; 3.8× / 1.5× |

## 26. Open questions and the most dangerous assumption

**What would show the whole idea is wrong:**
- **(a)** A transformer with task-structure embeddings (Abacus / position coupling) and a
  comparable budget matches HRM-X on extrapolation, *counterfactual consistency* and
  contradiction refusal. Then structure is unnecessary.
- **(b)** Library learning cannot grow the operator languages needed for a new domain (e.g.
  ARC-AGI evaluation) from these primitives. Then D4 does not scale beyond hand-given DSLs.
- **(c)** On free text, perception errors dominate (§20.2–3), so the exact core rarely receives
  a correct situation.

**Most dangerous unresolved assumption: that the hypothesis languages (arithmetic primitives,
object selectors and actions, event schemas) can be learned and grown rather than hand-given.**
Everything exact in this design is exact *relative to* those languages. The PoC gives them;
DreamCoder-style compression is the proposed mechanism for growing them, and it is the next
experiment.

**Further open questions:**
- Concept extension beyond the defining domain (fractional counts).
- Non-myopic value of computation.
- Learning event schemas from text.
- Two-rational law probing.
- Whether an HRM-Text-style neural hierarchy helps the Reader/Writer at small scale.
