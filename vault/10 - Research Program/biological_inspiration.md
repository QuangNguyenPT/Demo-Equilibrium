---
title: Biological inspiration — MaleCNS, and what survived testing
type: research-biology
tags: [research, connectome, malecns, mushroom-body, kimi-k3]
created: 2026-09-30
related: ["[[research_plan]]", "[[hypotheses]]", "[[ablation_results]]"]
---

# Biology → computational principle → mechanism → test

Chain used throughout: **observation → principle → mechanism → prediction → experiment**. The
analogy is never the result; the experiment is.

## 1. LOCAL DATA — Janelia male CNS v1.0, minconf 0.5 (`C:/Users/Admin/Downloads/flybrain`)

Inventory (read from the files, `experiments/connectome_malecns.py`):

| file | rows | contents |
|---|---|---|
| connectome-weights | 151,856,684 | body_pre, body_post, weight (synapse count) |
| body-annotations | 211,577 | superclass, class, type, instance, soma side, status, MANC match, … |
| body-neurotransmitters | 1,835,518 | predicted / consensus transmitter per body |
| body-stats | 88,384,522 | per-body pre/post counts by partner |
| syn-partners | 311,833,243 | synapse coordinates (pre/post), confidence, ROI |
| tbar-neurotransmitters | 45,656,140 | T-bar coordinates, per-transmitter probabilities, ROI |

No morphology skeletons; coordinates exist at synapse level (not used here).

**[RESULT] re-measured from the raw tables** (`experiments/results/connectome_malecns.json`):

| | this repo | delivery doc 14 |
|---|---|---|
| proofread core | 166,668 neurons, 25,578,988 edges, 124.2 M synapses | 164,679 / 25,563,433 / 124.0 M |
| reciprocity ≥ 1 synapse | 0.299 vs configuration null 0.0055 (54×) | 0.299 vs 0.0055 |
| reciprocity ≥ 5 synapses | 0.157 vs 0.0026 (60×) | 0.157 vs 0.0026 |
| Kenyon cells / claws per KC | 4,064 / median 5, mean 5.11; 283 without PN input | same |
| APL → KC | 196,200 synapses, 4,063 of 4,064 KCs | same |

(The core differs by 1,989 bodies: this repo excludes Orphan/Glia/Unimportant only.)

**[RESULT] new — M3, one code, many gated readouts:**
- each KC is read (≥ 3 synapses) by a median **9** of the 37 MBON types; 99.8 % by ≥ 2, 96.4 % by ≥ 5;
- the KC sets of different MBON types overlap little (mean pairwise Jaccard 0.135; median 689 KCs per type);
- **every** MBON type receives DAN input (median 5 DAN types);
- KC output synapses: 20.6 % to MBONs, 51.4 % to other KCs, 12.5 % to DANs, 9.4 % to APL.

## 2. LITERATURE [FACT]

- Male CNS connectome: Berg et al., bioRxiv 10.1101/2025.10.09.680999 (as cited in doc 14).
- KCs sample ~5 PNs, not uniformly at random: Zheng et al., Curr. Biol. 2022.
- MB compartments pair DAN inputs with MBON outputs; dopamine gates plasticity per compartment:
  Aso et al., eLife 2014.
- Ring attractor in the central complex: Kim et al., Science 2017 (doc 14 C3).
- Kimi K3 in C (`docs/kimi-k3-in-c-main.zip`, README): 16 of 896 experts per layer active
  (3.7 % of parameters per token); "the MoE routing bias steers selection only; the combining
  weights come from the unbiased scores"; the output is byte-identical from 8 GB to 224 GB —
  memory is a speed dial, not a quality dial.

## 3. INTERPRETATION → HYPOTHESIS → TEST

| observation | principle | mechanism here | prediction | test | verdict |
|---|---|---|---|---|---|
| KCs = random conjunctions of ~5 inputs (M2) | cover a space by random features | M2: sample constructions randomly instead of enumerating | more targets per budget beyond exhaustive reach | E15b | **rejected** — 9 % vs 55.5 % of targets at 80k evaluations, 4.6× fewer distinct classes |
| APL inhibits all KCs (M2); kWTA raises effective dimension (doc 14 X1b) | sparse code for one-shot association | M4: FHRR signature → kWTA code → readouts ordering construction | fewer ops to the right feature | E16 | **rejected here** — held-out top-1 5–7.5 % (chance 2.5 %), kWTA no better than dense ReLU, routing costs ~⅓ of exhaustive prefiltering |
| one KC code read by ~9 gated MBON types (M3) | a new association is a new *readout* of an existing code, not a new representation | concept memory: a concept is a new entry over the shared primitive code, admitted only after a second verification (the "second teaching signal") | stored concepts transfer to related tasks at low cost | E17/E19 | **supported** — memory off: 60 → 44 held-out correct; E17 rows needed fall only for related targets |
| 30 % reciprocity, 59 % lateral synapses (M1) | depth from recurrence, not layers | the H/L loop iterates shallow exact operations; depth grows only when VOC pays for it | ops grow with difficulty | E19 (ops by category) | **supported** — mean ops 2.5k written → 31k constructible → 152k novel → 229k outside |
| Kimi K3: routing steers selection only; byte-identical output across memory | selection must never change the answer | construction is exhaustive within a level; memory can shorten a description but a stored concept is still verified exactly on every task | memory changes cost and evidence needed, never a verified answer's correctness | E19 wrong-rate with memory on/off | **supported after the distinct-surplus fix** — 0 wrong with and without memory (before it: 2 wrong with memory) |

## 4. How people seem to do it — [SPECULATION], used only as a source of testable ideas

- People learn new concepts by composing old ones and need fewer examples for a concept built
  on a familiar one → tested as library compression (E15b, E17): **supported** in this domain.
- People name what they have abstracted and can then use the name without examples → the
  naming path of the language interface (`test_x_construct.py`): implemented, trivially works;
  not evidence of anything beyond memory lookup.
- People say "I don't know" or ask rather than invent → BOTTOM + question horizon: measured in E19.

## 5. What biology did *not* tell us

Nothing in the connectome says how a fly (or a person) invents a new primitive; the primitive
alphabet here is still given. Both biology-derived search/ordering mechanisms (M2, M4) failed their
tests in this domain; the organisational principle that survived (M3: shared code, many cheap
gated readouts) is the memory design, not the search.
