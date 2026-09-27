---
title: "Session: 2026-09-27 — Unified Model Build"
type: session-log
tags:
  - session
  - build
  - pem-port
  - knowledge-store
  - domains
date: 2026-09-27
status: in-progress
---

# Session: 2026-09-27 — Unified Model Build

Plan: [[ADR-003 Unified Cognitive Model]]. Every phase ends with tests green → graphify update → this log.

## User decisions (this session)

| Question | Answer |
|---|---|
| PEM code | Port and develop further — PEM is one section of the model, not all of it |
| Old C++ engine | Evaluate fit; fix if it fits, otherwise rewrite the backend → **did not fit** (see ADR-003), moved to `legacy/` |
| LLM | All three backends (stub, HF/torch, Ollama); LLM is a *native section* of the model that thinks in language, not a translator |
| Promotion | Both modes (auto behind gate / staged); autoresearch-style self-study driven by knowledge gaps and input intent |

## Phase log

| Phase                              | Result                                                                                                                                                                                                                       | Tests |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| P0 triage                          | `legacy/` holds old engine + blueprint; `pyproject.toml`; `.graphifyignore` excludes legacy                                                                                                                                  | —     |
| P1 substrate                       | FHRR algebra, name-derived deterministic codebook (read-only cached symbols), backend validation gate                                                                                                                        | 47    |
| P2 knowledge                       | SSD store: memmap fp16 keys + sqlite; copy-on-write rows; savepoints; crash-tested; disk cap eviction                                                                                                                        | 68    |
| P3 domains + verifier              | `Domain` protocol; grid domain (PEM port); sequence domain (new); independent verifier                                                                                                                                       | 100   |
| P4 two-level reasoning             | fast (recognition, zero ticks) → deep (equilibrium) → synthesis → abstain; confidence = agreement of exact explanations over the whole vocabulary; reasoner is read-only                                                     | 119   |
| P5a model bundle + online learning | `Model` = one SSD bundle (manifest, strict config, core generations, knowledge); episodes + answer log; human feedback quarantines / relearns from corrections                                                               | 132   |
| P5b sleep cycle                    | hash-locked frozen benchmark (51 tasks, `data/benchmark/benchmark-v1.json`); gate = no regression + zero decoy false-accepts + store integrity; auto/staged promotion; reversible graduation; rollback; ledger `results.tsv` | 148   |
| P5c metacognition + self-study     | competence map (learning progress), curiosity (skips mastered/stuck), intent regression matrix (asks when unsure), `perceive()`, autoresearch-style `study()`                                                                | 167   |

| P6 language section | inner speech in reasoning, strict program-language parser, text notation + LLM rewrite, intent text columns, 3 backends (hashing / transformers / ollama), `explain()`, CLI `python -m hrm` | 197 |
| P7 native backend | deferred by profiling; numpy caching + closed-family pruning instead (see ADR-003) | — |

## Real-scale measurements (D=4096, benchmark v1)

Solve rate 93.3% (gravity 0% by design), 0/6 decoy false accepts, 207 s. Self-study 30 tasks: 27 correct, 0 wrong, 3 abstained; one kept cycle consolidated `dihedral(r180)`, `affine(a=-3, b=0)`, `reverse`, `cumsum`, `diff` and graduated them off the SSD. Each sleep cycle costs ~7 min at D=4096 (two benchmark passes) — fine overnight, slow interactively.

## Design decisions made during the build

- **Reasoner never writes knowledge**; the learning section does — inference cannot corrupt the store.
- **Copy-on-write key rows** → rollback and crash recovery reduce to "the database decides which rows are live".
- **Graduation is reversible**: graduated items keep their keys (evicted first under the disk cap); a core rollback reactivates them.
- **Human outranks self**: only human feedback quarantines; self-study ground truth adds failure evidence and relearns.
- **Q-head training deliberately not wired**: it would change core weights outside the benchmark gate. Stays zero (PEM's rule-based behaviour) until it can be gated.
- **Message schema**: `task_id` names a new task; `refers_to` points at an earlier one (they were conflated — caught by a test).

## Known limitations (next work)

- **Retrieval for non-algebraic programs** (reverse, scale, dihedral): task keys are data-dependent, so SSD retrieval finds them only while the store is small. Fix: invariant descriptors in `task_key` (PEM's invariant/variant decomposition).
- Language section (P6) and native C++ backend (P7) not yet built.

## PEM defects found while porting (fixed in the port, not copied)

1. `shared` task vector bundled `Z_EX`-tagged pairs → only demo 0 contributed coherently; signature refinement targeted tagged vectors. Fixed: bundle raw pairs.
2. Trace recorded `"swapped"` instead of the failed segmentation name → failed parses were never down-weighted.
3. Anderson acceleration could never run (`trace = trace or Trace()` made the disable-flag always true) → not ported.
4. `eps_kinetic` scaled with √(2D) although the residual is already per-element RMS.
5. Conservation / error-spread checks cannot change a verdict once exact replay is required → verifier reduced to exact replay + determinism + anti-memorization.
6. D4 signatures were powers of `Z_TIME`/`SIZE` (scene-attribute bases) → named symbols per element.

## Bugs caught by this session's own tests

- Compaction copied a full chunk into a shorter destination slice on the last partial chunk (`ValueError: could not broadcast (6,512) into (8,512)`).
- Synthesis could not reach `affine(2,0) . reverse`: second factor drawn only from the top singles by energy → now every base program extends the beam; sequence similarity gained an order-free multiset term.
- Overconfidence: an ambiguous task (`[1,1]→[2,2]`, six exact explanations) reported confidence 1.0 → confidence now measured over the whole candidate vocabulary.
- Intent tie between answer and feedback when a new task was named → `refers_to` split out of `task_id`.

## Related

- [[ADR-003 Unified Cognitive Model]]
- [[Session - 2026-09-26 - Knowledge Graph Expansion]]
- [[Concept - PEM Phasor Equilibrium Machine]]
