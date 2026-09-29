---
title: "Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup"
type: session
tags:
  - session
  - hrm-x
date: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
code:
  - hrm_x/__init__.py
  - hrm_x/qlinalg.py
  - hrm_x/concepts.py
  - hrm_x/consolidation.py
  - hrm_x/situation.py
  - hrm_x/algebra.py
  - hrm_x/operators.py
  - hrm_x/reader.py
  - hrm_x/domains/concept.py
  - experiments/x_v05.py
  - tests/test_x_concept.py
  - tests/test_simplify_graph.py
  - scripts/simplify_graph.py
  - scripts/update_knowledge_graph.ps1
  - scripts/knowledge_map.py
  - .gitignore
  - .graphifyignore
  - .obsidian/workspace.json
---

# Session — 2026-09-29: HRM-X v0.5 structural induction, and the vault cleanup

Two things happened. The Obsidian vault was cleaned up because it was presenting
someone else's model as this project's system knowledge, and the HRM-X v0.5
research delivery was ported into the codebase — with a measurement that came out
better on one axis and **worse** on another, and is recorded as such.

## 1. Vault: what was wrong

`vault/Knowledge Graph/System Knowledge Graph.canvas` was nine text cards, eight
of which described **Kimi K3** — a 2.78-trillion-parameter model with 896 routed
experts, MXFP4 quantisation and 5.9 GB/s `O_DIRECT` NVMe streaming. It was filed
under *this project's* Knowledge Graph, and
`vault/Knowledge Graph/00 - Knowledge Graph Index.md` linked to it as "the system
knowledge graph", with four empty wiki links to notes that did not exist and
admitted as much in the body. Six of its ten concept notes were about Kimi K3.

Underneath that, `vault/_graph-view/` held 651 derived notes plus a 654-node
canvas, and 207 of those notes were named after things this repository never
defines: `Any.md`, `Any_1.md` … `Any_24.md`, `ndarray.md`, `textwrap.md`,
`fixture.md`, `parametrize.md`. `.obsidian/workspace.json` carried twenty of
those dead paths in its recent-files list.

### What was done

- **The pruner now resolves provenance** (`scripts/simplify_graph.py`). A code
  node survives if graphify attributed a `source_file` to it, or if the
  repository's own source defines its label. An external module stub needs a
  file of that name in the tree. Real project symbols the extractor failed to
  attribute (`Rule`, `Solution`, `Workspace`) are recovered by the lookup;
  imported names are not.
  **Measured on the real graph: 2,950 → 321 nodes (10.8%), and the Obsidian view
  651 → 402 notes with a 402-node canvas.** `Any*`, `ndarray`, `textwrap`,
  `fixture` and `parametrize` are gone.
  The first attempt at the definition regex was too loose — it captured the first
  identifier on any line, so `from pathlib import Path` and `@pytest.fixture`
  reintroduced 14 `Path` notes. The keyword is now required, and
  `test_no_kept_node_is_an_import_the_repository_never_defines` asserts the
  property against the real graph.
- **The Kimi K3 notes moved** to
  `vault/Knowledge Graph/External Research (Kimi K3)/` with an index that says
  what transfers to this project and what cannot (all of it is a serving result
  about someone else's checkpoint). Four dangling `[[Semantic Relationship
  Matrix]]` links, to a note deleted in an earlier cleanup, were repointed.
- **The canvas was rewritten** as the project's own system: the model that runs,
  the baselines it is measured against, the specs and ADRs that say what is and
  is not claimed, and the concepts behind our design — with the external research
  drawn deliberately outside the picture. Every node is a `file` node pointing at
  a note that exists.
- **The index was rewritten** and no longer over-claims what the canvas maps.
- **Housekeeping**: the 0-byte `data.md` at the repository root is deleted; the
  45 MB vendored `docs/kimi-k3-in-c-main/` tree and its 32 MB zip are gitignored
  (already excluded from the graph); Obsidian's recent-files list went from 42
  entries, 10 of them dead, to 10 live ones.

## 2. Codebase: the v0.5 port

The delivery at `C:\Users\Admin\Downloads\HRM-X_v0.5_complete_delivery` supplies
fourteen research documents and a 2,038-line reference PoC that benchmarks itself
against this repository by importing it. What came in:

| new module | what it is |
|---|---|
| `hrm_x/qlinalg.py` | exact RREF / rank / solve / row-space membership over ℚ |
| `hrm_x/concepts.py` | 68 law-closed families, exact fit, Prop. 4 weighting, Def. 3 BOTTOM, the coverage rule, Theorem 3 determinacy |
| `hrm_x/consolidation.py` | corpus-MDL macro admission; the PoC's `library.py`, renamed because `hrm_x/library.py` already exists at the program level |
| `hrm_x/situation.py` | `Q^(E×I)` state, one-shot verb frames, conservation check |
| `hrm_x/algebra.py` | FHRR binding, characters, Prop. 8 cleanup bound |
| `hrm_x/operators.py` | `D4 × Sym(colours)` grid operators, solved not searched |
| `hrm_x/reader.py` | Place-Value Reader (Thm 5) — torch, and only here |

`hrm_x/domains/concept.py` is new and is the actual decision: the mathematics as
a `Domain`, driven by the existing `Mind.solve`. The delivery's standalone
`concepts.decide()` loop was **not** imported. Two controllers in one repository
is one too many, and a head-to-head is only fair when both arms share the same
value-of-computation loop, modulators, patience and trace writer.

Two deviations from the delivery, both recorded in the code:
- `operators.py` used `scipy.ndimage.label`; scipy is not installed here, so the
  component finder is `hrm_x.domains.grid.objects`, the repository's own. One
  component finder, not two.
- the PoC's `apply_global` called `np.nindices`, which does not exist. Every call
  raised `AttributeError`, so the whole global route could not execute. It is
  `np.indices`, which returns identical arrays.

## 3. The measurement, including the part that went against the delivery

`experiments/x_v05.py` runs both routes over the same operators with the same
`Mind`. **26 solves per condition**, recorded to `experiments/results/x_v05.json`.

| E13a, 21 queries | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| concept v0.5 | **14** | 7 | 0 | **751.5** |
| arith v0.4 | 9 | 4 | 8 | 4247.2 |

| split | concept v0.5 | arith v0.4 |
|---|---|---|
| in-language (15) | **13 / 2 / 0** at 593 ops | 7 / 4 / 4 at 4265 ops |
| out-of-language (6) | 1 / **5** / 0 at 1147 ops | 2 / **0** / 4 at 4202 ops |

**In-language the route wins**: +6 correct, −2 wrong, −4 abstentions, and 7.2×
fewer ops, because deriving stopped being a search.

**Out-of-language it is worse than the program route**: 5 confident wrong answers
out of 6, against 0 wrong and 4 abstentions. The delivery's README reports the
opposite direction. Two reasons this repository measures differently, both
recorded: these numbers come from `Mind.solve` and not from the delivery's
standalone loop, and the operator lists are this repository's.

The mechanism is now understood and pinned. On `a² + b` with evidence
`(2,3) → 7, (4,3) → 19`, the full bi-affine family fits both rows. It is not
symmetric, so the coverage rule refuses it at `(5, 7)`. Then the controller asks
one question, the answer brings a second value of `y` into the evidence, `y` has
now "varied", and the rule stops firing. The dual confirms. The mind answers 47
where the truth is 32. **Asking is what defeats the coverage guard.** Two tests
pin it.

Two more measured limits:

- **The closed-world ablation is nearly vacuous here** (14 ops out of 1618 on the
  task list designed to exercise it). The machinery is correct and, on these
  operators, close to inert.
- **L = 4 is not independently choosable.** The delivery calibrated its answer
  threshold at `L = 4`; `Controller` ships `L = 1`. At `L = 4` the route abstains
  on **21 of 21**, because before any verification `rho` is its Beta(4,1) prior
  of 0.8 and `U = (1 + 4)(0.8) − 4 = 0` exactly, so nothing has positive value.

## 4. Files changed, with agents

All by **opencode/stealth/space-bunny-alpha** (`space-bunny-alpha`), 2026-09-29,
except where a line says otherwise.

| file | change |
|---|---|
| `hrm_x/__init__.py` | version and pointer to the v0.5 spec and ADR-010 |
| `hrm_x/qlinalg.py`, `concepts.py`, `consolidation.py`, `situation.py`, `algebra.py`, `operators.py`, `reader.py` | ported from the delivery; `scipy` and `np.nindices` deviations noted in-file |
| `hrm_x/domains/concept.py` | **new.** The mathematics as a `Domain`: propose/derive/verify/ask/remember, with verification by the dual solve |
| `experiments/x_v05.py` | **new.** E13: both routes, one mind, five conditions, two task lists |
| `tests/test_x_concept.py` | **new.** 26 tests: determinacy, Prop. 4, BOTTOM, coverage, dual verification, consolidation, phasor algebra, grid operators, situation model, and the three negative results |
| `tests/test_simplify_graph.py` | updated for the provenance signature; four new tests for it |
| `scripts/simplify_graph.py` | provenance resolution (`Definitions`, `is_proven`, `project_definitions`, `--root`) |
| `scripts/update_knowledge_graph.ps1` | passes `--root` to the pruner |
| `scripts/knowledge_map.py` | new `HRM-X Concept Induction (v0.5)` subsystem, so `test_knowledge_map.py` passes |
| `.gitignore`, `.graphifyignore` | the vendored Kimi tree is gitignored as well as graph-ignored |
| `vault/Knowledge Graph/System Knowledge Graph.canvas` | rewritten: the project's own system, every node a real note |
| `vault/Knowledge Graph/00 - Knowledge Graph Index.md` | rewritten; no dangling links, no third-party model as our own |
| `vault/Knowledge Graph/External Research (Kimi K3)/` | **new.** Six notes moved here, plus an index |
| `vault/06 - Devlogs & ADRs/ADR-010 Structural Operator Inference.md` | **new.** The decision, the measurement, the negative results |
| `vault/01 - Architecture Blueprint/HRM-X Specification v0.5 ...md` | **new.** The normative design, including the gates that are *not* met |
| `data.md` | deleted (0 bytes) |
| `.obsidian/workspace.json` | recent files pruned to 10 live entries |

## 5. Not done, deliberately

- **The rival-margin rule** (doc 13's P3) is the identified fix for the
  coincidental-fit channel and is not implemented. It is a design with its own
  evidence requirement and its own ADR.
- **The strongest neural baseline** (Abacus, arXiv 2405.17399) has still not been
  run. Every comparison here is symbolic against symbolic.
- **Library growth of new concept *kinds*** remains the project's most dangerous
  assumption and is still untested.

## Index

[[00 - Session Index]] · [[ADR-010 Structural Operator Inference]] ·
[[HRM-X Specification v0.5 - Structural Operator Inference]]
