# HRM+LLM Hybrid AI Model Research Project

## Session-Start Rules (MANDATORY — read before any action)

Every session MUST complete these steps before touching code:

1. **Read vault session index** → `vault/08 - Chat Context & Sessions/00 - Session Index.md`
   - Understand the decision chain: what was decided, why, what changed last session
2. **Check ADR status** → `vault/06 - Devlogs & ADRs/` — know current phase (ADR-003..008; ADR-008 = HRM-X v0.4, current)
3. **Run test suite** → `python -m pytest --tb=short -q` — baseline before any change
4. **Verify training state** → if `models/eqb-small/` absent, training may need resuming (check `runs/train-eqb-small.log`)
5. **Update graph after every session** → `.\scripts\update_knowledge_graph.ps1 -Mode code-only`

**Why:** Two BSOD crashes taught us: never start blind. Context from previous sessions prevents re-doing fixed bugs, re-breaking invariants, or racing training runs.

### Per-Prompt Checklist (before implementing any change)

- [ ] Does this change a certified map, routing normalization, or σ? → `test_eqb_theorems.py` must stay green
- [ ] Is this a CUDA command? → only through `select_device()` (70% VRAM cap)
- [ ] Is this a write to knowledge? → only through `mind.learn()` with a `VerificationReport`
- [ ] Is this editing a frozen benchmark or `hrm/learning/harness.py`? → STOP, make a new version instead
- [ ] Does this affect training? → check checkpoint exists, use `--pause 0.15` on GPU
- [ ] Does this touch `hrm_x/native/` or a Python function it mirrors? → rebuild (`.\scripts\build_native.ps1`) and keep `tests/test_x_native.py` (differential parity) green

### Information Path (traceability)
```
Question → Research (vault/Knowledge Graph/) → Decision (vault/06 ADRs) 
→ Implementation (hrm_eqb/) → Test (tests/) → Session log (vault/08)
→ Knowledge graph (graphify-out/) → vault/ sync
```

---

## Project Overview

Engineering hybrid AI architecture combining Hierarchical Reasoning Model (HRM) with modern LLMs. Reference: Executive Summary + detailed system design doc in `/docs`.

**Core Goal:** Merge deep structured reasoning (HRM: explicit multi-step iteration) with broad language understanding (LLM: next-token prediction on massive data). Test whether architectural fusion beats chain-of-thought prompting alone.

## Repo Structure (as built, ADR-003)

```
hrm/                       the model, numpy-first (torch only for the transformers backend)
  config.py                ModelConfig: one dataclass per section, strict load
  substrate/               FHRR phasor algebra, name-derived codebook, backend validation gate
  perception/              Domain protocol; grid/ (PEM port), sequence.py; program language parser
  verify/                  independent judge: exact replay + determinism + anti-memorization
  knowledge/store.py       SSD store: memmap fp16 keys + sqlite (items, outcomes, episodes, answers)
  reasoning/               fast (L) -> language -> deep (H equilibrium) -> synthesis -> abstain
  language/                LanguageSection backends, inner speech, text notation
  metacognition/           self-model (competence), curiosity, intent regression matrix
  learning/                online learner, generators, frozen harness, sleep cycle, self-study
  core.py                  core weights <-> JSON (per generation)
  model.py                 Model facade = one bundle directory; human controls
  __main__.py              CLI: python -m hrm create|ask|feedback|explain|sleep|study|report|...
hrm_eqb/                   HRM-EQB (torch): the model (ADR-005, spec v0.2 incl. §14)
  algebra.py               rotation-vector text (complex planes) + quaternion knowledge (4-D)
  certified.py             Cayley x sigmoid-diag weights, sub-stochastic routing, radial ReLU
  equilibrium.py           L/H equilibrium core; certificate (Γ, r_n) from SVD norms; halting
  tasks.py / field.py      generated families + ARC/Sudoku loaders; token field, augmentation
  model.py                 prelude -> certified segments -> answer/halt heads; coda.py (language)
  memory.py / knowledge.py quaternion product-key memory; knowledge.pt (in-place rows, WAL, seal)
  hrmc.py                  model.hrmc: full core, recomputed certificate, probes, forgery refused
  train.py / mind.py       training (checkpointed); runtime Answer-or-Question, verified writes
  study.py / bench.py      autoresearch-style keep/discard (G1); benchmark suites
  device.py                select_device(): VRAM cap (see GPU Safety note)
  __main__.py              CLI: python -m hrm_eqb create|info|ask|learn|study|bench|seal
tests/                     pytest; test_eqb_theorems.py (§2-8), test_eqb_model.py + test_eqb_runtime.py (§14); test_x_*.py (HRM-X)
data/benchmark/            frozen benchmark v1 (never overwrite; make v2)
hrm_x/                     HRM-X v0.4 (ADR-008): operator inference over structured situations, CPU only
  workspace.py / rules.py  e-graph with provenance (I1-I3), e-matching, saturation; make_workspace() picks backend
  induction.py             programs -> behaviour classes (posterior), QuickSpec law discovery
  mind.py                  belief + VOC controller (H), Modulators (rho, volatility), sleep, save/load
  writer.py                trace -> answer + explanation
  domains/                 arith (concepts, laws, residue verifier), story (parser .pt + situation), grid (objects)
  native/                  C++ backend: include/hrmx.h (C ABI), src/egraph.cpp, src/grid.cpp; ctypes wrapper
experiments/               x_arith|x_story|x_grid|x_control|x_metacog|x_native -> results/*.json
scripts/                   make_benchmark.py, update_knowledge_graph.ps1, build_native.ps1 (clang++ -> hrm_x/native/bin, gitignored)
vault/                     Obsidian vault; session logs in "08 - Chat Context & Sessions"
legacy/                    superseded C++ GRU engine + blueprint (reference only, graph-ignored)
```

## Key Insights from Research

### Architecture Candidates (Preferred: A)

**Status note (2026-09-27).** The sections below that describe a transformer
stack, RoPE, FlashAttention, attention residuals, a 6-stage pretraining pipeline
and MXFP4 compression are *design targets carried over from the original
research brief*. Updated 2026-09-27 (ADR-005): `hrm_eqb/` now has a small
transformer prelude, rotation-bound (RoPE-as-binding) positions, Attention
Residuals over the thinking chain, and a certified two-level equilibrium core
trained with deep supervision. There is still no LLM pretraining pipeline,
FlashAttention kernel, MoE or MXFP4, and the ACT Q-learning head is a supervised
"is this answer right" halting head instead. `hrm/` remains the numpy
program-search model of ADR-003/004.
- **A (Separate HRM Module):** LLM encoder → HRM reasoning → decoder. Modular, reusable, trains as add-on.
- **B (Hierarchical Inside Transformer):** Recurrent loops embedded in layers. Hard to train, breaks parallelism.
- **C (Latent Reasoning Loop):** Extra hidden-state updates between tokens. Simple, but slow inference.
- **D (Adaptive Compute/ACT):** Variable steps per input. Complex training, unstable batching.
- **E (Mixture of Experts):** Specialist experts routed per token. High complexity, rarely needed.

**Recommended starting point:** Architecture A + ACT-style halting (variant D for cost control).

### Data Flow (Tensor Shapes)
```
Input: (B, T) token IDs
  ↓ Embedding + RoPE/learned pos: (B, T, D)
  ↓ Transformer encoder (L layers): (B, T, D)
  ↓ Pool/project to state: (B, D)
  ↓ HRM loop (R reasoning steps, only last gradded):
      - Low-level: z_L ← L_level(z_L, injected_input) → (B, D)
      - High-level: z_H ← H_level(z_H, z_L) → (B, D)
      - ACT Q-head: q ← Q_head(z_H) ∈ [0,1]  (halt if q > threshold)
  ↓ Final: z_H_final (B, D)
  ↓ Language head (linear): (B, V) logits
  ↓ Softmax + sample: next tokens
```

### Training Pipeline (6 Stages)

1. **Language Pretraining** – Transformer backbone, cross-entropy next-token (billions of tokens)
2. **Instruction Tuning** – Supervised fine-tune on instruction→output (ChatGPT-style)
3. **Reasoning Curriculum** – Multi-step tasks (puzzles, math). Train HRM on (input, solution) pairs.
4. **Integrated Fine-Tuning** – LLM + HRM jointly on mixed language + reasoning data.
5. **Alignment** – RLHF or classification loss (optional, for safety/preference).
6. **Compression** – Quantize (INT8), distill to smaller student (optional).

**Critical:** ACT halting loss requires target Q-values (e.g., max(halt_Q, continue_Q)); HRM code shows Q-learning pattern.

### Compute & Memory Reality

| Reasoning Steps | Compute | Memory (Inference) | Feasibility |
|---|---|---|---|
| 1 | ~1.0× LLM cost | KV cache (main) | ✅ Always |
| 5 | ~1.2–1.5× | + hidden states | ✅ Good |
| 10 | ~2× | Moderate growth | ✅ Acceptable |
| 50 | ~5–6× | Significant | ❌ Long dialogue only |

**Rule:** Keep R ≤ 10 for interactive use. Use ACT to halt early on easy queries.

### Performance Optimizations

- **FlashAttention** (must-have) – Fused Q·K·softmax, tiling. Reduces memory I/O by 10×.
- **Multi-Query Attention (MQA)** or **Grouped-Query (GQA)** – Speed inference KV cache load.
- **Triton/CUDA kernels** – Custom HRM loop, fused LayerNorm+residual, etc.
- **Mixed precision (FP16/BF16)** – 2× throughput on modern GPUs (Ampere/Hopper).
- **Gradient checkpointing** – Trade compute for memory (essential for large R).
- **DDP + ZeRO** – Data parallel (small teams), fully-sharded (memory-constrained).

### Open Problems

- **Emergent vs explicit:** Does architecture alone beat prompting? Needs empirical validation.
- **Efficient iteration:** Train halting stably without supervision → unsolved.
- **Modular independence:** How much can reasoning & language share weights? Tension exists.
- **Interpretability:** Validate intermediate reasoning steps (probes, auxiliary losses).
- **Scaling laws for reasoning:** Unknown if bigger = better reasoning or hit ceiling.

## Code Style & Conventions

**From companion Python project (C:\Users\Admin\Desktop\Code\Python\Project1):**

### Logging Tags
Use `[TAG]` format in prints for clarity:
```python
print("[MOUSE] Connected successfully")
print("[ERROR] Config reload failed: {e}")
print("[HID] Attempting reconnect...")
```

### Type Hints
Always use typing module:
```python
from typing import Optional, Tuple, List

def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def load_config(path: str = "config.json") -> dict:
    ...
```

### Docstrings
One-liner or short (class-level). No multi-line comment blocks unless reasoning WHY is non-obvious:
```python
class HIDMouseController:
    """Thread-safe HID device controller with reconnection support."""
    
def move(self, x: int, y: int) -> bool:
    """Send movement packet."""
    ...
```

### Threading & Locks
Use `threading.Lock()` for shared state, avoid race conditions:
```python
self.lock = threading.Lock()

with self.lock:
    self.nonce_counter += 1
    # ... critical section
```

### Configuration
Load from JSON, validate early, use dicts for flexibility:
```python
config = load_config("config.json")
fov = int(config["FOV"])
magnet = bool(config["Magnet"])
```

### Error Handling
Explicit exceptions, print context, retry logic where appropriate:
```python
for attempt in range(3):
    try:
        mouse = HIDMouseController(vid, pid)
        break
    except Exception as e:
        print(f"[MOUSE] Attempt {attempt + 1}/3 failed: {e}")
        if attempt < 2:
            time.sleep(2)
```

### Avoid
- Bare `except:` — catch specific exceptions.
- Commented-out code that won't be used again — delete it.
- Global mutable state without locks — use classes or thread-safe patterns.
- Hard-coded magic numbers — extract to named constants or config.

## Development Workflow

```powershell
python -m pytest                                   # full suite (~45 s)
python scripts/fetch_benchmarks.py                 # ARC-AGI-1 + Sudoku-Extreme -> data/external/ (gitignored)
python -m experiments.x_arith                      # HRM-X experiments (also x_story, x_grid, x_control, x_metacog, x_native)
.\scripts\build_native.ps1                          # HRM-X C++ backend; HRMX_BACKEND=python|native|auto selects it
python -m hrm_eqb create models/eqb-small --minutes 120 --arc-train data/external/arc/training
python -m hrm_eqb bench models/eqb-small --suites generated,language,sudoku,arc,empty_brain,schedule
python -m hrm_eqb ask models/eqb-small task.json   # Answer, or a Question with candidates + bits
python -m hrm_eqb study models/eqb-small --minutes 10   # keep/discard gate, results.tsv ledger
python -m hrm create models/main --benchmark data/benchmark/benchmark-v1.json
python -m hrm study models/main --tasks 100 --minutes 480   # self-study, sleep cycles, ledger
python -m hrm report models/main                            # competence map + generations
.\scripts\update_knowledge_graph.ps1 -Mode code-only        # after every phase
```

Rules that keep the loops honest (ADR-003):
- Reasoner never writes knowledge; `hrm/learning` does. Knowledge writes need a passed `VerificationReport`.
- Never edit a frozen benchmark or `hrm/learning/harness.py` casually: it breaks every model's harness lock (by design). Make a new benchmark version; a human re-signs.
- Core weights change only through the sleep gate (or human promote/rollback). Q-head training is intentionally not wired until it can be gated.
- Measure before optimizing: `cProfile` a D=4096 solve mix first (see ADR-003 P7).

### Ablation
Always test:
- HRM disabled (LLM-only baseline) → measure reasoning δ
- ACT disabled (fixed R steps) → measure cost savings from early halt
- Reasoning curriculum disabled → measure language performance δ

## Integration with Claude Code

### Running Claude Code Operations
Use caveman mode when appropriate. Tag modifications with `[caveman]` in commit messages or comments when using terse style.

### File Edits
- Read first with `Read` tool.
- Use `Edit` for surgical changes; `Write` only for new files or complete rewrites.
- Batch independent tool calls (no sequential delays).

### Testing Before Commit
```bash
python -m pytest --tb=short
```

## Memory & References

- **Research doc:** `/docs/Executive Summary.pdf` (comprehensive design, architectures A–E, training pipeline, cost analysis)
- **HRM paper:** sapientinc/HRM GitHub (code reference: HierarchicalReasoningModel_ACTV1 class)
- **Transformers:** Vaswani et al., 2017 (original architecture)
- **FlashAttention:** Dao et al., 2022 (critical for long contexts)
- **RoPE (rotary pos-emb):** Su et al., 2021 (handles variable sequence lengths)
- **ACT (Adaptive Computation Time):** Graves et al., 2015 (learned halting)

## Checkpoints & Versioning

**Naming convention:** `checkpoint-{stage}-{epoch}.pt`
- `checkpoint-pretrain-100.pt` — Stage 1, epoch 100
- `checkpoint-reasoning-50.pt` — Stage 3, epoch 50
- `checkpoint-final-integrated.pt` — Stage 4, final

**Restore:**
```python
model.load_state_dict(torch.load("checkpoint-reasoning-50.pt"))
```

**Save best on dev set:**
```python
if val_loss < best_loss:
    torch.save(model.state_dict(), f"checkpoint-best-{stage}.pt")
    best_loss = val_loss
```

## Debugging & Monitoring

### Gradient Norms
Watch for explosion/vanishing:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### Attention Patterns
Visualize to catch issues (e.g., all tokens attending to first, or dead patterns):
```python
# Extract attn weights from a layer, plot heatmap
import matplotlib.pyplot as plt
plt.imshow(attn_weights[0, 0].detach().cpu(), cmap='viridis')
plt.colorbar()
```

### HRM Convergence
Monitor hidden-state changes per step:
```python
for step in range(max_steps):
    z_h_prev = z_h.clone()
    z_h, z_l, q = hrm(z_h, z_l, input_emb)
    delta = (z_h - z_h_prev).norm()
    print(f"Step {step}: delta={delta:.4f}, q={q[0]:.3f}")
    if q[0] > threshold:  # Halt
        break
```

## When to Use Caveman Mode

Terse responses preferred for:
- Implementation task execution (build features, not explain)
- Code changes justified by test passing or simple fact
- Bug fixes (state the issue + fix without preamble)
- Configuration/deployment commands

Use normal mode for:
- Architecture decisions or trade-offs that need weighing
- Security warnings or irreversible actions
- Multi-step sequences where order matters

---

**Last updated:** 2026-09-28  
**Status:** Unified model built (ADR-003 P0–P6): reasoning, SSD knowledge, metacognition, learning loops, language section. Native backend deferred by measurement (P7).
**Current (ADR-008, 2026-09-28):** HRM-X v0.4 in `hrm_x/` is the measured PoC; spec `vault/01 - Architecture Blueprint/HRM-X Specification v0.4 - Operator Inference over Structured Situations.md`. Python is the reference; C++ (`hrm_x/native/`) runs profiled hot paths only, exact Fraction arithmetic stays in Python. Hormones = only signals with a computational role (rho, feedback, volatility; lambda fixed); do not add others without an ablation.
**Earlier direction (ADR-005):** HRM-EQB replaces program search as the model. The spec is `vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md` (§14 = construction lemmas). N0–N7 are built and tested; N8 (training + benchmarks) is recorded in ADR-006. `hrm/` is teacher, judge and baseline.
**Rule:** anything that changes a certified map, the routing normalization or σ must keep `tests/test_eqb_theorems.py` green. The proofs depend on those invariants.
**GPU rule (two crashes, 2026-09-27):** all CUDA work goes through `hrm_eqb.device.select_device()` (70 % VRAM cap) with pair/token-budgeted batches; probe peak memory on a worst-case batch before any long run; stop background jobs only by task ID. See `vault/05 - Engineering & Guides/GPU Safety - VRAM Oversubscription and DPC Watchdog.md`.
**Git:** the repo is under version control (branch `feat/hrm-eqb-full`); commit per phase with the full suite green.

## Knowledge Graph — Auto-Update Rule

**Run after every modification session:**
```powershell
.\scripts\update_knowledge_graph.ps1
```

Options:
- `.\scripts\update_knowledge_graph.ps1 -Mode deep` — richer semantic extraction (needs API key)
- `.\scripts\update_knowledge_graph.ps1 -Mode code-only` — code-only, no key needed
- `.\scripts\update_knowledge_graph.ps1 -NoObsidian` — skip vault sync

Graph written to `graphify-out/` and synced to `vault/` automatically.

## PEM Integration Notes (2026-09-26) — superseded by ADR-003

These notes describe the legacy C++ engine (now in `legacy/`). The PEM port lives in `hrm/`:
conjunctive halt, false-attractor restart and the 9-feature Q-head are in
`hrm/reasoning/controller.py` and `hrm/reasoning/qhead.py`.

**Key patterns ported from PEM (`C:\Users\Admin\Desktop\Code\AI\PEM`):**

1. **9-Feature Q-Head** (`hrm/reasoning/qhead.py`, not `hrm.act_qhead` — that path does not exist): Q operates on 9-dim controller features, not raw z_h (D-dim). Features: residual, residual_slope, energy, diversity, memory_confidence, verification_margin, plateau_count, restarts_used, bias. W_q shape: (2, 9). Q-head training is intentionally not wired (see S3, deferred).

2. **Conjunctive Halt (Hard Rails)**: ACCEPT = kinetic (residual < ε_k) AND semantic (energy < ε_e) AND verification (halt_p ≥ τ). Simple threshold insufficient — prevents reward hacking.

3. **False-Attractor Detection**: kinetic converged but semantic hasn't → RESTART. Prevents low-|Δz_h| + high-uncertainty trap.

4. **Annealed Restart**: sigma = sigma0 × gamma^k. Perturbs z_h + z_l. Budget-limited (restart_budget). Exhausted → ABSTAIN.

**Attention Residuals (RealFormer / Kimi K3 pattern):**

Pre-softmax score passthrough between transformer layers:
```
score_l = QK^T/√D + prev_scores_{l-1}
```
Enabled via `TransformerConfig.use_attn_residuals = True`.
- Block variant (`attn_residual_block_size` > 1): O(L/block) memory vs O(L).
- Fixes Pre-LN dilution (magnitude grows O(L) with depth).
- 1.25× compute advantage (arxiv:2603.15031, Kimi Linear).

**Key Decisions Locked (updated):**
6. **Q-Head**: 9-feature linear Q (PEM-inspired), not full z_h projection. Hard rails for ACCEPT/ABSTAIN.
7. **Attention Residuals**: RealFormer cross-layer score passthrough (optional, `use_attn_residuals` flag).

## Session Logging & Chat Context

**Every session writes a log to `vault/08 - Chat Context & Sessions/`.**

This captures: key decisions, research findings, files modified, open questions — making chat context queryable via knowledge graph.

### Session Log Location
```
vault/08 - Chat Context & Sessions/
├── 00 - Session Index.md    ← decision chain + information path
├── Session - YYYY-MM-DD - <topic>.md
└── ...
```

### Information Path
Traceability: Question → Research → Decision → Implementation → Validation → Node

See `vault/08 - Chat Context & Sessions/00 - Session Index.md` for the full decision chain.

### How to Query Sessions
```powershell
# Project-scoped config lives in .env (see .env.example); no personal key needed.
.\scripts\update_knowledge_graph.ps1 -Mode deep
graphify query "what was decided about Q-head design"
graphify query "Kimi K3 memory reduction techniques"
```

### New Research (2026-09-26): Kimi K3 in C
`github.com/FareedKhan-dev/kimi-k3-in-c` — 2.78T params on 8.24 GB RAM, pure C99:
- **MXFP4**: 4-bit experts, 0.53125 bytes/param → 3.56× size reduction
- **MoE sparsity**: 896 experts, 16 active → 13.7× (only dense trunk in RAM)
- **KDA**: O(1) constant-memory recurrent attention (no T² score matrix)
- **MLA**: KV → 512-dim latent vector → further KV cache compression
- **Trunk streaming**: 93 layers packed contiguous, `O_DIRECT` from NVMe
- **Byte-identical**: 8 GB and 224 GB produce same tokens — memory = speed dial only
- **Attention residuals**: "layers that look back" (section 8) = RealFormer-style

Port candidates for this project: KDA for long HRM sequences, MXFP4 in Stage 6 compression.
