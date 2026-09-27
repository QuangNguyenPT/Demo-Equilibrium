---
title: Model Blueprint & Graph Layout
type: blueprint
tags:
  - python
  - blueprint
  - topology
status: active
created: 2026-09-26
---

# 🐍 Model Blueprint & Graph Layout

The Python Blueprint Layer (`python/hrm_blueprint/`) provides an intuitive, high-level API for creating and modifying neural network topologies.

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

Calling `blueprint.summary()` yields a clean ASCII table of all layers:
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
- [[Hybrid System Philosophy]] — Why this design was chosen.
- [[Configuration & Hyperparameters]] — Configuration schema reference.
- [[C-ABI & Interop Specification]] — How blueprints are dispatched to C++.
- [[Template - Model Blueprint]] — Template for authoring new blueprints.
