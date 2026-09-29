---
title: Configuration & Hyperparameters
type: blueprint
tags:
  - python
  - configs
  - hyperparameters
status: superseded-not-implemented
created: 2026-09-26
updated: 2026-09-28
superseded_by: "[[HRM-Phi Design Document v0.1]]"
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[Model Blueprint & Graph Layout]]"
---

# ⚙️ Configuration & Hyperparameters

> [!WARNING] Not implemented as described
> This note describes the ADR-001 C++/Python hybrid engine, which was **never built**; ADR-003 moved it to `legacy/`. The live C++ is only `hrm_x/native/` (`egraph.cpp`, `grid.cpp`) behind the `hrmx.h` C ABI. The current model is [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] (ADR-008), in `hrm_x/`. Kept as design history.

> [!IMPORTANT] Specifically
> `python/hrm_blueprint/configs.py` does not exist (it is `legacy/python/hrm_blueprint/configs.py`,
> unused), so the `TransformerConfig`, `HRMConfig` and `EngineConfig` tables below describe no
> class in the live codebase. **`configs/` at the repo root is an empty directory** — the only
> YAML that ever existed there (`hrm_base.yaml`, `small_toy.yaml`) is now `legacy/configs/`. No
> model is configured by YAML in this project.

---

## 🟢 The real configuration surfaces

Configuration is plain Python dataclasses, and there are exactly two config modules, one per
non-`hrm_x` generation. `hrm_x/` (the current model) has no config module: its thresholds are
module constants in `hrm_x/mind.py` and `hrm_x/induction.py`, deliberately close to the code that
consumes them.

### 1. `hrm/config.py` — the program-search baseline (ADR-003)

`ModelConfig` nests one dataclass per subsystem: `SubstrateConfig` (phasor codebook), 
`KnowledgeConfig` (SSD store caps, retrieval), `PerceptionConfig` (grid/sequence/vector search
bounds), `VerifyConfig` (`max_program_complexity`, `alpha_mdl` for MDL), `ReasoningConfig`
(`n_streams`, `ticks_per_cycle`, `max_ticks`, the `eps_kinetic` / `eps_energy` / `tau_verify`
rails, `restart_gamma`, `restart_budget`), `LearningConfig` (`min_successes`,
`consolidate_amount`, `promotion`), `MetaConfig` (mastery, patience, explore) and
`LanguageConfig` (backend `none | hashing | transformers | ollama`).

Two properties that **do** still hold, and are worth keeping as a convention:
- **Loading is strict.** A saved config with a missing, unknown or mistyped field raises
  `ConfigError` instead of being silently defaulted: "a model running under a half-understood
  config is architecturally wrong, not merely slower."
- **Round-trip is JSON**, via `ModelConfig.to_dict()` / `ModelConfig.load(path)` / `save(path)`.

### 2. `hrm_eqb/config.py` — the equilibrium baseline (ADR-005/006)

A frozen `ModelConfig` with a strict JSON round-trip: `dim`, `heads`, `prelude_layers`,
`max_side`, `n_low`, `max_segments`, `solve_tol`, `max_cycles`, `grad_cycles`, `route_iters`,
`memory_side`, `memory_topk`, `key_dim`, `value_quaternions`, `coda_dim`, `coda_layers`,
`thought_tokens`, `max_text`. `__post_init__` rejects invalid combinations (e.g. `dim` must be a
multiple of 4 and of `heads`). No unknown or missing key is accepted on load.

### 3. `hrm_x/` — the current model (ADR-008)

No dataclass config file. The control-law constants live next to the controller that reads
them, and the measured behaviour is in `experiments/results/*.json` (8 files) rather than in a
config: e.g. E1 exact extrapolation 1.000 across splits, E2 composition 1.000 at depths 1–3,
native parity with grid 3.79× / arithmetic 1.47×.

### 4. Legacy

`legacy/configs/hrm_base.yaml` and `legacy/configs/small_toy.yaml` — YAML for the ADR-001 engine.
Nothing loads them. The historical schema follows.

---

## 📋 Configuration Schema Reference *(historical, ADR-001 — classes that were never built)*

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
> [!CAUTION] No such class
> `EngineConfig`, `arena_capacity_mb`, `num_threads` and `enable_avx2` were never implemented.
> There is no arena, no thread pool and no AVX2 build switch in `hrm_x/native/`, which is
> single-threaded and unoptimised by hand. Its only build option is `-Debug` in
> `scripts/build_native.ps1`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arena_capacity_mb` | `int` | `128` | Pre-allocated bump arena size in Megabytes. |
| `num_threads` | `int` | `4` | Worker threads in C++ ThreadPool. |
| `enable_avx2` | `bool` | `True` | Compile with AVX2 vectorization flags. |

---

## 🔗 Related Notes
- [[Model Blueprint & Graph Layout]] — Using configs to instantiate blueprints. *(never implemented)*
- [[HRM+LLM Fusion Architecture]] — Architectural context for hyperparameters. *(historical)*
- [[Build & Toolchain Guide]] — the one real build option, `-Debug`.
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] — the current model's parameters and where they live.
