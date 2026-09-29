---
title: "ADR-003: Unified Cognitive Model — Reasoning + Language + SSD Knowledge + Metacognition"
type: adr
tags:
  - adr
  - architecture
  - hrm
  - pem
  - llm
  - knowledge
  - metacognition
  - self-study
status: accepted-baseline
date: 2026-09-27
supersedes: "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
---

# ADR-003: Unified Cognitive Model

> [!NOTE] Status: accepted **as the baseline**, not as the model
> This design was built (P0–P6, 197 tests) and is measured, so the record stands —
> but `hrm/` is now the **program-search baseline** that later work is compared
> against, not the design target. It was superseded as *the model* by ADR-005
> (HRM-Φ / HRM-EQB) and then by [[ADR-008 Operator Inference over Structured Situations]]
> (HRM-X v0.4, in `hrm_x/`), whose specification was normative, and by
> [[ADR-010 Structural Operator Inference]] (HRM-X v0.5), which is the current target.
>
> Two path corrections to the text below:
> - The root `src/`, `include/` and `apps/` paths this ADR discusses under
>   "Decision — C++ backend" **no longer exist at the repository root**. They were
>   moved to `legacy/` by the same decision, and were never live.
> - Phase **P7 (native C++ backend)** was deferred by measurement here (see
>   "P7 native backend — deferred by measurement") and did **not** land as a
>   rewrite of this design. It landed later, scoped to the e-graph kernels of a
>   different model, as `hrm_x/native/` under ADR-008.

## Context

Goal (user, 2026-09-27): a model that **understands** input (not only pattern-matches), for ARC/AGI tasks and beyond.
- **Reasoning section** (PEM-derived) does the thinking: two levels — **H = deep/slow**, **L = fast**.
- **Language section** (LLM) is a *native section* of the same model, not a translator: the model partly *thinks in language* (inner speech, hypothesis proposals), and language is where pattern knowledge is useful.
- **Knowledge** lives in a **dynamic model file on SSD** (not RAM), embedded in the model's forward pass, updated continually; consolidated knowledge flows into the core ("into the neurons").
- **Self-directed learning** (autoresearch-style): the model estimates what it knows / does not know, studies its own knowledge holes, and accepts feedback from itself and from the user.
- **Metacognition**: it infers the *intent* of each input (answer? learn? feedback?) with a learned regression matrix.

Sources: PEM (`C:\Users\Admin\Desktop\Code\AI\PEM`, 71 tests green), kimi-k3-in-c (local copy), karpathy/autoresearch, HRM paper.

## Operational definition of "understanding"

An input is *understood* when the model finds a compact, executable explanation (program / rule / fact binding) that **exactly reproduces every observation**, prefers the simplest one (MDL), and uses it to answer. If no explanation passes verification, the model **abstains** — or, on language-only paths with no executable check, labels the answer **unverified (pattern-based)**. This is the honest boundary: verification is only as strong as the domain's executable check.

## Decision — model sections

| Section | Package | Origin | Role |
|---|---|---|---|
| Substrate | `hrm/substrate` | PEM port | FHRR phasor algebra: zero-parameter computation |
| Perception | `hrm/perception` | PEM grid encoder + new `Domain` protocol | Input → structured scene (objects, attributes, relations) |
| Reasoning | `hrm/reasoning` | PEM engine port | **L = fast** resonator ticks + memory hits; **H = deep** Birkhoff-mixed KM equilibrium; controller (conjunctive halt, annealed restart, abstain, Q-head) |
| Verification | `hrm/verify` | PEM port, generalized | Independent deterministic judge: exact replay + MDL |
| Knowledge | `hrm/knowledge` | PEM Engram + kimi-k3 streaming lessons | SSD model file: memmap fp16 keys + sqlite items/feedback + manifest generations; frequency-aware hot cache |
| Language | `hrm/language` | new | Native LLM section: encode, propose hypotheses (inner speech), verbalize; latent bridge ↔ phasor space. Backends: stub (tests), HF/torch (latents), Ollama (text-level) |
| Metacognition | `hrm/metacognition` | new | Intent head (softmax regression), self-model (competence map per region), curiosity (learning-progress goal selection) |
| Learning | `hrm/learning` | PEM distiller/trainer/runner + autoresearch | Surprise synthesis, consolidation, feedback ledger, frozen harness, self-study loop, promotion (auto or staged) |

## Decision — C++ backend

Existing engine (`src/`, `include/`, `apps/`) evaluated 2026-09-27 — **does not fit, rewrite**:
- no complex/phasor kernels (reasoning section needs them)
- fp32-only tensors: no fp16/bf16/quantized weights, no safetensors, MHA-only KV cache that copies the full cache per `get_k` (language section needs these)
- global `-ffast-math` / `/fp:fast` breaks the deterministic FP contract that numpy cross-validation (PEM) and byte-identical output (kimi-k3) depend on
- did not compile: `hrm_c_api.cpp:227`, `apps/cli/main.cpp:87` (stale `run_hrm_reasoning_loop` signature); tests passed only against a stale DLL

Old engine moved to `legacy/` (reversible). New backend (phase P7) follows PEM's csrc pattern: C ABI, self-test, loader gate that cross-validates every kernel against the numpy reference before activation. **Numpy reference first** — it is the oracle the native code is checked against.

## Loops (loop-design-check)

| Loop | Type | Done-criterion (machine-decidable) | Boundary (must NOT) | Fallback / damping |
|---|---|---|---|---|
| **A. Online** (every input) | servo | program reproduces **all** demos exactly | never write unverified knowledge; verifier immutable | abstain; restart budget |
| **B. Self-study / sleep** (autoresearch-style) | servo per cycle, regulator overall | frozen benchmark solve-rate ≥ baseline **and** no family regresses > tol **and** 0 false accepts on decoy set **and** store integrity check passes | harness + benchmark hash-locked (loop refuses to run on mismatch); loop cannot edit verifier/metric | rollback (sqlite savepoint + core snapshot); stop after N consecutive discards and report; max cycles per session |
| **C. Development** (this build) | plan/build/judge | `pytest` green, test count never decreases, no test weakened | no speculative code; no deleted tests | stop and report after 3 failed attempts |

Human keeps judgment: promotion mode `auto` (behind gate B) or `staged` (human runs promote); only the human may change the gate; human feedback outranks self-feedback and can quarantine knowledge.

## Phases (each ends: tests green → graphify update → vault note)

| Phase | Deliverable | Acceptance |
|---|---|---|
| P0 | legacy move, `pyproject.toml`, `hrm/config.py` strict config | pytest green; unknown/missing keys rejected; save/load round-trip |
| P1 | substrate port | algebra property tests (unbind∘bind = id, similarity bounds, quantization error ≤ 1e-3, seed determinism) |
| P2 | knowledge model file (SSD) | reopen-from-disk round-trip; unverified write refused; crash between key append and commit leaves store consistent; savepoint rollback; quarantine excluded from retrieval; keys are memmapped |
| P3 | domains + verifier | ported PEM grid tests; sequence domain; verifier refuses non-reproducing programs |
| P4 | two-level engine + solver | low-R/high-E never ACCEPT; Q-head cannot grant ACCEPT; fast path vs deep path; abstain on decoys |
| P5 | learning + metacognition | surprise rate declines on re-presented families; regression → rollback; hash mismatch → refuse; human "wrong" quarantines; intent head learns; curiosity targets gaps |
| P6 | language section | inner-speech hypotheses join candidate set; HF/Ollama backends (skip when absent) |
| P7 | native C++ backend | kernels cross-validated against numpy; loader refuses a broken kernel |

## Outcome (2026-09-27)

P0–P6 built and gated: 197 tests. Real-scale measurements (D=4096, benchmark v1, 51 tasks): solve rate 93.3% (all families 100% except `gravity`, outside the vocabulary by design), 0/6 decoy false accepts, 207 s; 30-task self-study: 27 correct / 0 wrong / 3 abstained, one kept sleep cycle consolidated 5 search-discovered programs into the core and graduated them off the SSD with no regression.

### P6 language — as built
- Inner speech inside reasoning (after fast recognition fails): proposals in the domain program language, strictly parsed (`parse_program`, bounded literals, injection-tested), verified like any hypothesis. A fluent wrong proposal cannot become an answer (tested).
- Text perception: arrow notation parsed deterministically; free language rewritten by the language section, then parsed strictly.
- Intent matrix gains three text columns (similarity to intent prototypes in the language embedding space).
- **Deferred**: learned latent bridge (LLM hidden states ↔ phasor space). A random projection would be meaningless; the next step is a contrastive objective aligning embeddings of program descriptions with program signatures, trained on the knowledge store's verified items.

### P7 native backend — deferred by measurement
Profile of a D=4096 solve mix (fast / synthesized / abstained / decoy):
- resonator ≈51%: waste in the ported code (conjugate signatures rebuilt up to 4×/tick, constant task scores recomputed every tick, per-stream matvecs). **Fixed in numpy**: cached `conj_t`, `task_scores` once per run, batched streams → resonator 4.55 s → 2.45 s, identical outputs.
- program evaluation ≈37%: per-program Python overhead in synthesis. **Fixed algorithmically**: compositions inside closed families pruned → evaluations 4620→2570 (gravity), 1108→204 (scale+reverse).
- What remains is BLAS matmuls (already native) and interpreter overhead. Ported FHRR elementwise kernels would not move either.
Triggers to build native code (validation gate `set_backend` is ready): knowledge store > ~50k items (fp16 SSD scan with O_DIRECT + prefetch, kimi-k3 style), synthesis dominating wall time (batched DSL interpreter), or GPU residency for large candidate sets.

## Risks

- Generality: "understands everything" is a research target. Verified understanding exists only where a domain provides an executable check; the Domain protocol is the extension point.
- Language-only answers cannot be verified exactly → always flagged `unverified`.
- Self-modifying loops need stricter review, not looser: every generation is kept, every cycle logged to `results.tsv`.

## Related

- [[ADR-004 Discriminating Benchmark and Reasoning Ceiling]] — builds the instrument that tests this design, finds 4 defects in it, and shows the deep engine is not yet load-bearing
- [[Concept - PEM Phasor Equilibrium Machine]]
- [[Concept - The Four Memory Reductions]]
- [[Session - 2026-09-27 - Unified Model Build]]
- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]]
