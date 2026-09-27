---
title: Configuration & Hyperparameters
type: blueprint
tags:
  - python
  - configs
  - hyperparameters
status: active
created: 2026-09-26
---

# ⚙️ Configuration & Hyperparameters

Configurations in this project are strongly typed using Python `dataclasses` in `python/hrm_blueprint/configs.py` and serialized to YAML/JSON under `configs/`.

---

## 📋 Configuration Schema Reference

### 1. `TransformerConfig` (LLM Backbone)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_layers` | `int` | `6` | Number of Transformer blocks. |
| `hidden_dim` | `int` | `512` | Embedding & hidden dimension ($D$). |
| `num_attention_heads` | `int` | `8` | Number of query attention heads ($H$). |
| `num_kv_heads` | `Optional[int]` | `None` | Key/Value heads (set for GQA, `None` for MHA). |
| `intermediate_dim` | `int` | `1536` | SwiGLU MLP hidden dimension (usually $\sim \frac{8}{3} D$). |
| `max_seq_len` | `int` | `2048` | Maximum sequence context length. |
| `norm_eps` | `float` | `1e-6` | Epsilon for RMSNorm numerical stability. |
| `rope_base` | `float` | `10000.0` | Base frequency constant for RoPE angles. |

### 2. `HRMConfig` (Reasoning Module)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hidden_dim` | `int` | `512` | Reasoning state vector dimension ($D$). |
| `max_reasoning_steps` | `int` | `8` | Maximum reasoning cycles ($R_{\text{max}}$). |
| `low_level_substeps` | `int` | `2` | Number of fast executor iterations per planner step. |
| `act_threshold` | `float` | `0.5` | Halting probability threshold to terminate early ($\tau$). |
| `use_act_halting` | `bool` | `True` | Enable or disable dynamic Adaptive Computation Time. |
| `inject_mode` | `str` | `"residual_sum"` | How context is injected (`residual_sum` or `linear_proj`). |

### 3. `EngineConfig` (C++ Runtime)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arena_capacity_mb` | `int` | `128` | Pre-allocated bump arena size in Megabytes. |
| `num_threads` | `int` | `4` | Worker threads in C++ ThreadPool. |
| `enable_avx2` | `bool` | `True` | Compile with AVX2 vectorization flags. |

---

## 🔗 Related Notes
- [[Model Blueprint & Graph Layout]] — Using configs to instantiate blueprints.
- [[HRM+LLM Fusion Architecture]] — Architectural context for hyperparameters.
