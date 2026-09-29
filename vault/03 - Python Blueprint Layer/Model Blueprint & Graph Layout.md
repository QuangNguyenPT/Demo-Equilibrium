---
title: Model Blueprint & Graph Layout
type: blueprint
tags:
  - python
  - blueprint
  - topology
status: superseded-not-implemented
created: 2026-09-26
updated: 2026-09-28
superseded_by: "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[Hybrid System Philosophy]]"
---

# 🐍 Model Blueprint & Graph Layout

> [!WARNING] Not implemented as described
> This note describes the ADR-001 C++/Python hybrid engine, which was **never built**; ADR-003 moved it to `legacy/`. The live C++ is only `hrm_x/native/` (`egraph.cpp`, `grid.cpp`) behind the `hrmx.h` C ABI.
>
> **Superseded as the design target 2026-09-29.** At the time of writing this note the current model was HRM-X v0.4 (spec v0.4, ADR-008); the current target is **HRM-X v0.5** (`hrm_x/`), spec v0.5 + ADR-010. Kept as design history.
> Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.

> [!IMPORTANT] Specifically
> There is no `python/hrm_blueprint/` package in this repo. The module named in the import below
> lives at `legacy/python/hrm_blueprint/`, and nothing in `hrm/`, `hrm_eqb/` or `hrm_x/` imports
> it. `ModelBlueprint`, `LayerLayout`, `HybridHRMLLMBlueprint`, `blueprint.summary()`,
> `total_parameters`, `modify_reasoning_depth` and `export_json` **do not exist**. `python/` at
> the repo root is an **empty directory**. The transformer-graph layer counts in the example
> (23 layers, 12,456,194 parameters) were never computed against a trained model, so treat them
> as illustrative, not as measurements.

**What replaced it.** The current model needs no graph layer, because nothing downstream
consumes a topology description. `hrm_x/` builds its structure directly in Python:

| Concern | Where it lives now |
| :--- | :--- |
| hyperparameters | `hrm/config.py` (`ModelConfig` + per-subsystem dataclasses), `hrm_eqb/config.py` (frozen `ModelConfig`) |
| the "graph" | the e-graph itself — `hrm_x/workspace.py` |
| node/mutation API | `hx_eg_add` / `hx_eg_lookup` / `hx_eg_union` / `hx_eg_rebuild` in `hrm_x/native/`, called over `hrmx.h` |
| domain programs | `hrm_x/domains/{arith,story,grid}.py` |
| a parameter count | one convention-layer tagger, 40,908 parameters (ADR-008) — not a 23-layer decoder |

The general lesson is worth keeping: a declarative layer pays for itself only if something
executes it. Nothing did.

---

## 🎨 Design Pattern: Pure Declarative Topology

In our architecture, Python does not execute low-level tensor matrix multiplies. Instead, Python acts as a **blueprint architect**:
1. Creates `LayerLayout` nodes representing layers and operations.
2. Assembles them into a `ModelBlueprint` graph.
3. Automatically computes weight dimensions, memory requirements, and total parameter counts.
4. Allows researchers to dynamically modify hyperparameters (e.g. reasoning steps, layer counts, dimensions) in simple Python statements.
5. Passes the configuration to the C++ engine for execution.

---

## 💻 Code Example: Creating and Mutating a Blueprint
> [!CAUTION] This code cannot run
> `from hrm_blueprint import ...` raises `ModuleNotFoundError` from the repo root. Preserved as the
> API the ADR-001 design intended.

```python
from hrm_blueprint import (
    ModelConfig,
    TransformerConfig,
    HRMConfig,
    HybridHRMLLMBlueprint,
)

# 1. Define configuration
config = ModelConfig(
    vocab_size=32000,
    transformer=TransformerConfig(
        num_layers=12,
        hidden_dim=768,
        num_attention_heads=12,
        intermediate_dim=2048,
    ),
    hrm=HRMConfig(
        hidden_dim=768,
        max_reasoning_steps=8,
        act_threshold=0.6,
    ),
)

# 2. Build declarative blueprint
blueprint = HybridHRMLLMBlueprint.create(config)

# 3. Print architectural summary
print(blueprint.summary())
print(f"Total Parameters: {blueprint.total_parameters:,}")

# 4. Rapidly tweak reasoning depth without touching C++!
HybridHRMLLMBlueprint.modify_reasoning_depth(blueprint, new_max_steps=12)

# 5. Export topology graph to JSON
blueprint.export_json("exported_blueprint.json")
```

---

## 📊 Summary Output Format
> [!CAUTION] Synthetic
> The table below is illustrative. No run of `blueprint.summary()` ever produced it; the shapes and
> the parameter count are hand-written placeholders, not measurements.

Calling `blueprint.summary()` was to yield a clean ASCII table of all layers:
```text
=======================================================================
 Model Blueprint Layout: HRM-LLM-Hybrid-Base
 Total Layers: 23 | Total Parameters: 12,456,194
=======================================================================
Layer Name                   Op Type          Out Shape      Params      
-----------------------------------------------------------------------
token_embeddings             embedding        -1x-1x256        4,096,000
layer_0.attn_norm            rms_norm         -1x-1x256              256
layer_0.attention            flash_attention  -1x-1x256          262,144
layer_0.ffn_norm             rms_norm         -1x-1x256              256
layer_0.swiglu_mlp           swiglu_mlp       -1x-1x256          589,824
...
hrm.context_pooler           context_pool     -1x256              65,536
hrm.low_level_executor       recurrent_executor -1x256             393,984
hrm.high_level_planner       recurrent_planner -1x256             393,984
hrm.act_qhead                act_halting_head -1x2                   514
lm_decoder_head              linear_head      -1x16000         4,096,000
=======================================================================
```

---

## 🔗 Related Notes
- [[Hybrid System Philosophy]] — Why this design was chosen. *(historical)*
- [[Configuration & Hyperparameters]] — the real config surfaces. *(the schema below is historical too)*
- [[C-ABI & Interop Specification]] — How blueprints were to be dispatched to C++. *(never implemented)*
- [[HRM-X Workspace & Rules]] — **the graph layer that exists** (e-graph with provenance).
- [[Template - Model Blueprint]] — Template for authoring new blueprints.
