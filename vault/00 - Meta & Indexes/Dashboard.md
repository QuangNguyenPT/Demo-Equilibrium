---
title: System Architecture & Knowledge Dashboard
type: dashboard
tags:
  - meta
  - dashboard
  - overview
status: active
created: 2026-09-26
updated: 2026-09-29
---

# 🧠 HRM-X Knowledge Dashboard

> [!IMPORTANT] What this vault describes today
> The live model is **HRM-X v0.5 — structural operator inference** (spec v0.5,
> [[ADR-010 Structural Operator Inference]]), in `hrm_x/`. A hypothesis is a law-closed family
> fitted exactly over ℚ, not a program found by enumeration. **Gates G4 and G5 are not met** —
> read ADR-010 before quoting a headline. The notes under `01`–`05` were written for the
> ADR-001 "Python blueprint + C++ engine" design, which **was never built**; ADR-003 moved it to
> `legacy/`. They are kept as design history, not as a description of the code. For the code
> that actually exists, use [[Code Map]].

Welcome to the internal engineering knowledge base for **HRM-X**: a reasoner that infers
*operators* from evidence and applies them exactly to *structured situations*, checking each
answer in an independent representation.

---

## 🧬 Three generations, all present in the repo

| Generation | Path | Role | Standing |
| :--- | :--- | :--- | :--- |
| `hrm_x/` | ADR-010, spec v0.5 | **Current model.** Law-closed hypothesis families fitted exactly over ℚ, determinacy, open-world coverage, VOC control, dual-solve verification. The v0.4 program-search route (`domains/arith.py`, `library.py`) stays inside it as the comparison baseline. C++ hot paths in `hrm_x/native/`. | live |
| `hrm_eqb/` | ADR-005, ADR-006 | HRM-Phi equilibrium baseline: neural H/L trained on algebra. It measured no arithmetic (ADR-006), which is why it was replaced. | baseline |
| `hrm/` | ADR-003 | The first unified cognitive model: program search over a phasor substrate. | baseline |
| `legacy/` | ADR-001, ADR-002 | The C++ engine plus Python blueprint design. **Never integrated with the live model.** | historical |

---

## ⚡ Current architecture

```mermaid
flowchart TD
    subgraph HRMX["hrm_x/ — CURRENT (ADR-010, spec v0.5)"]
        direction TB
        subgraph V5["v0.5 concept core — structural operator inference"]
            QL["qlinalg.py — exact RREF / rank / solve over Q"]
            CO["concepts.py — 68 law-closed families,<br/>Thm 3 determinacy, Def. 3 BOTTOM"]
            CN["consolidation.py — corpus-MDL macro admission"]
            CD["domains/concept.py — the same mathematics<br/>as a Domain, under the same Mind"]
        end
        subgraph PY["Python: the blueprint and the reference semantics"]
            WS["workspace.py — e-graph with provenance"]
            RU["rules.py — laws and e-matching"]
            IN["induction.py — programs, version spaces (v0.4 route)"]
            LB["library.py — program-level DSL mining (v0.4 route)"]
            MI["mind.py — L: saturation + exact execution<br/>H: value of computation"]
            WR["writer.py — answer in words"]
            DOM["domains/ — arith.py, story.py, grid.py"]
        end
        subgraph OTH["other v0.5 components"]
            ST["situation.py — Q^(ExI) state, one-shot frames"]
            AL["algebra.py — FHRR binding, Prop. 8"]
            OP["operators.py — D4 x Sym(colours), solved"]
            RD["reader.py — Place-Value Reader (REQUIRES TORCH)"]
        end
        subgraph NAT["hrm_x/native/ — C++ hot paths behind the hrmx.h C ABI"]
            EG["src/egraph.cpp — union-find, hash-cons,<br/>rebuild, e-matching, explanations"]
            GR["src/grid.cpp — connected components,<br/>object-level steps"]
        end
        QL --> CO
        CO --> CD
        CN --> CO
        ST --> CD
        OP --> DOM
        AL -.-> CD
        RD -.-> CD
        CD --> MI
        CO -.-> MI
        WS --> MI
        RU --> MI
        IN --> MI
        LB -.-> IN
        WS <--> EG
        DOM <--> GR
        MI --> WR
    end

    subgraph BASE["Baselines, kept for comparison"]
        EQB["hrm_eqb/ — HRM-Phi equilibrium (ADR-005/006)"]
        HRM["hrm/ — program-search substrate model (ADR-003)"]
    end

    LEG["legacy/ — ADR-001 C++ engine + Python blueprint:<br/>designed, never integrated"]

    EQB -.-> HRMX
    HRM -.-> HRMX
```

Exact `Fraction` arithmetic stays in Python on purpose (ADR-008): arbitrary precision is the
guarantee, so the C++ accelerates only the e-graph and the grid, and parity with the Python
reference is tested differentially (`tests/test_x_native.py`).

---

## 📊 Quick metrics (measured, `experiments/results/*.json`)

- **Tests**: `python -m pytest` → **436 passed, 1 skipped**.
- **Exact extrapolation (E1)**: **1.000** on every split — in-range, 10²–10⁴ integers, large decimals, 10×3.1.
- **Composition (E2)**: **1.000** at depths 1–3.
- **Native backend**: results identical to Python; grid **3.79×**, arithmetic **1.47×**.
- **v0.5 vs v0.4, one mind (E13)**: in-language **13/15 correct at 593 ops** against 7/15 at 4,265 ops (7.2x cheaper); out-of-language **5 confidently wrong of 6**, where v0.4 had 0 wrong and 4 abstentions. See [[ADR-010 Structural Operator Inference]].
- **Experiment result files**: **12** in `experiments/results/` (`x_v05.json` added 2026-09-29).

---

## 🧭 Navigation & Maps of Content (MOCs)

| Knowledge Hub | Focus Area | Direct Link |
| :--- | :--- | :--- |
| **Master Index** | Complete vault graph and topic index | [[Master MOC]] |
| **Code Map** | Which document is implemented by which file | [[Code Map]] |
| **Current spec** | HRM-X v0.5, structural operator inference (normative) | [[HRM-X Specification v0.5 - Structural Operator Inference]] |
| **Decision Records** | ADR-010 is current; the ADRs it replaced are kept verbatim | [[ADR-010 Structural Operator Inference]] · [[ADR-011 Reintroducing EQB Ideas into HRM-X]] |
| **HRM-X v0.5 concepts** | One note per v0.5 module | [[HRM-X Concept Induction (v0.5)]] |
| **Python subsystems** | One note per module of `hrm_x/`, `hrm_eqb/`, `hrm/` | [[HRM-X Workspace & Rules]] |
| **The only C++** | E-graph and grid kernels behind `hrmx.h` | [[HRM-X Native Backend]] |
| **Build & Toolchain** | `scripts/build_native.ps1`, `python -m pytest` | [[Build & Toolchain Guide]] |

### Historical — the ADR-001 design that was never built

| Knowledge Hub | Focus Area | Direct Link |
| :--- | :--- | :--- |
| **System Philosophy** | Why Python blueprint + C++ engine | [[Hybrid System Philosophy]] |
| **HRM+LLM Fusion** | Research brief, architectures A–E | [[HRM+LLM Fusion Architecture]] |
| **Tensor Data Flow** | Shape mappings $(B, T) \to (B, D) \to (B, V)$ | [[Tensor Flow & Data Pipelines]] |
| **C++ Engine & Kernels** | Arena, tensor views, thread pool, kernels | [[Engine Architecture & Memory Model]] |
| **Python Blueprint** | Declarative layout API, graph mutators, configs | [[Model Blueprint & Graph Layout]] |
| **C-ABI & Interop** | The `hrm_c_api.h` that was never written | [[C-ABI & Interop Specification]] |
