---
title: "HRM-Φ — Phasor Hierarchical Reasoning Model: Complete Design Document v0.1"
type: design
tags:
  - design
  - hrm
  - pem
  - latent-reasoning
  - parametric-memory
  - model-format
status: superseded
superseded_by: "[[HRM-EQB Specification v0.2 - Theorems and Proofs]]"
date: 2026-09-27
related:
  - "[[ADR-005 Neural HRM-Phi Replaces Program Search]]"
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
  - "[[ADR-003 Unified Cognitive Model]]"
---

# HRM-Φ — Phasor Hierarchical Reasoning Model
## Complete Design Document — v0.1

> **Superseded (2026-09-27)** by [[HRM-EQB Specification v0.2 - Theorems and Proofs]].
> v0.2 closes this document's open convergence gap (§4.4 here). It iterates in Euclidean
> space with contraction by construction, adds proofs for every claim, and checks each
> one numerically. Kept for history.

**Status:** research design specification, not yet built. Every mechanism is grounded in
a published result (HRM, TRM, recurrent-depth LMs, FHRR/VSA, RoPE, QK-norm, product-key
memory, Titans, DEQ Jacobian regularization) and in PEM v1.1. The composition is
unvalidated. Numbers marked (est.) are engineering estimates to be confirmed by §12.

---

## 0. Reflection: why the current build is not the target

**What `hrm/` is.** A symbolic program search. Domains enumerate a hand-written
vocabulary (translate, recolor, affine, rotate…); a verifier replays each candidate
against the demonstrations; beam synthesis composes up to three steps. The phasor
substrate only *ranks* candidates. ADR-004 measured the consequence: removing the
H-engine, the phasor ranking, and the priors/SSD knowledge left benchmark v2 at 100%.
The reasoning layers were not load-bearing, because exhaustive search subsumes them.
The language section was attached from outside and contributed 0/6 useful proposals.
It does not *learn to reason*; it enumerates what a human wrote down.

**What HRM actually is, and what the evidence says.**

| Source | Finding | Consequence for this design |
|---|---|---|
| HRM (Wang et al., 2025) | Two recurrent transformer modules (H slow, L fast), 1-step gradient, deep supervision, ACT; 27M params; Sudoku-Extreme 55%, ARC-AGI-1 40% | the target family: **learned latent iterative refinement** |
| ARC Prize analysis (Aug 2025) | the H/L hierarchy ≈ a same-size transformer; the **outer refinement loop** drove the gains (+13 pp from 1→2 loops, 2× from 1→8); training *with* refinement matters more than inferring with it; HRM is **transductive** (a `puzzle_id` embedding, no few-shot context) | refinement + deep supervision are core; the task must be read **from its demonstrations**, not from an ID |
| TRM (Jolicoeur-Martineau, Oct 2025) | one 2-layer net (7M) with a latent `z` and an answer `y` beats HRM: Sudoku 87.4%, Maze 85.3%, ARC-1 44.6%, ARC-2 7.8%; full backprop through the last recursion, no fixed-point theorem; fewer layers generalize better | one shared network; a working stream and a solution stream; no reliance on a fixed-point theorem |
| Mechanistic study of HRM (2026) | the solution lives in z_H; z_L is transient working state | z_H = solution stream, z_L = working stream |
| Recurrent-depth LM (Geiping et al., 2025) | prelude → recurrent core → coda; latent reasoning emits tokens; input injection each step; truncated backprop over the last k iterations | the shape for "thinking in latent, speaking through a coda" |
| Memory Layers at Scale (Berges et al., 2025) | trainable product-key memory, millions of slots, sparse reads, >2× factual accuracy at equal compute | the knowledge file's mechanism |

**Target.** A *learned* latent refinement machine in the HRM/TRM line. It must be
inductive (it conditions on demonstrations), use PEM's algebra as its native geometry,
keep parametric knowledge in an SSD-resident memory file, and speak through a small
transformer coda that reads its thoughts. The existing symbolic system becomes the
teacher, judge and baseline (§7.4), not the model.

---

## 1. Overview and invariants

HRM-Φ has five parts, all built from one operator (§2.5):

1. **Prelude P** (perception): tokens → a phasor field.
2. **Core Φ** (reasoning): one shared block with two roles, L (working state) and H
   (solution state), iterated inside an outer refinement loop with deep supervision and
   adaptive halting.
3. **Knowledge M** (memory): a product-key parametric memory whose values live in an
   SSD-resident `.pt` file.
4. **Structured head**: decodes the solution state by cleanup against the input
   codebook.
5. **Semi-embedder Σ + coda C** (language): compresses the thinking chain into thought
   tokens; a small causal transformer verbalizes them.

**Invariants.**

- **I1 One algebra.** Every read is attention on the torus, every position is a
  binding, every mixture is a bundle. There is no second representation to convert to
  (Propositions 1–3).
- **I2 Thought is latent; language is a readout.** Words never carry the reasoning
  state; the coda verbalizes, it does not decide.
- **I3 Two files, one mind.** The core file holds *how to think* (operators). The
  knowledge file holds *what is known* (memory values). They are paired by a schema
  hash. The core without knowledge is an empty brain that still reasons in context.
- **I4 Nothing unverified persists.** A test-time memory write reaches the SSD only if
  the model then reproduces every demonstration exactly; core promotion only passes
  through the frozen benchmark gate.
- **I5 Claims are measured, not assumed.** Contraction, convergence and "the hierarchy
  helps" are measured, stored in the core file as a certificate, and ablated.

---

## 2. Substrate: the plane algebra shared by PEM and the transformer

### 2.1 Planes and phasors

Split $\mathbb{R}^{2m}$ into $m$ planes: $\mathbb{R}^{2m}=\bigoplus_{j=1}^{m}\mathbb{R}^2$.
Identify plane $j$ with $\mathbb{C}$, so a vector is $v\in\mathbb{C}^m$. A **phasor** has
unit modulus in every plane; the phasors form the torus $T^m=(S^1)^m$. A state is a field
of $n$ tokens, $Z\in T^{n\times m}$.

### 2.2 Operations

- **Binding** is plane-wise rotation: $(a\circledast b)_j=a_j b_j$. For $b=e^{i\theta}$ this
  is the block-diagonal orthogonal map $R(\theta)=\mathrm{diag}(R_{\theta_1},\dots,R_{\theta_m})$.
  It is exactly invertible: $a\circledast b\circledast\bar b=a$.
- **Fractional power**: $(X^x)_j=e^{i x\theta^X_j}$ for any real $x$.
- **Bundling** is superposition then projection, $\Pi(a+b+\dots)$, with per-item SNR
  $\approx\sqrt{m/N}$ (PEM §2.2).
- **Similarity**: $\mathrm{sim}(a,b)=\mathrm{Re}\langle a,b\rangle/m\in[-1,1]$.
- **Smoothed projection**: $\Pi_\varepsilon(v)_j=v_j/\sqrt{|v_j|^2+\varepsilon^2}$. It is
  differentiable everywhere and globally $1/\varepsilon$-Lipschitz, and it equals the exact
  torus projection up to $O(\varepsilon^2/|v_j|^2)$.

### 2.3 Proposition 1 — RoPE is fractional binding

RoPE rotates pair $j$ of a query or key at position $p$ by angle $p\,\omega_j$. In plane
notation, with the phasor $P_j=e^{i\omega_j}$:

$$\mathrm{RoPE}(q,p)=q\circledast P^{p}.$$

Hence

$$\mathrm{Re}\langle q\circledast P^{p},\,k\circledast P^{p'}\rangle=\mathrm{Re}\sum_j q_j\bar k_j e^{i(p-p')\omega_j}=\mathrm{Re}\langle q\circledast P^{p-p'},\,k\rangle,$$

so relative position is unbinding. For coordinates $p\in\mathbb{R}^a$ (x, y, z, example
index…), with one phase vector $\omega^{(\ell)}$ per axis on all planes:

$$\mathrm{RoPE}_{a}(q,p)=q\circledast X^{p_x}\circledast Y^{p_y}\circledast Z^{p_z}\circledast\cdots$$

This is exactly PEM's coordinate encoding. **One operator**, so HRM-Φ encodes position
only through binding at the query/key interface; content travels as the payload.

The spectrum matters. The relative-offset kernel is
$k(\Delta)=\frac1m\sum_j\cos(\langle\Delta,\omega_j\rangle)$. Geometric frequencies
(RoPE) give a smooth, multi-scale locality kernel; uniformly random phases (PEM) give
$k(\Delta)\approx\delta_{\Delta,0}$ for integer offsets, i.e. exact identity. HRM-Φ splits
the planes: half geometric, half random, so
$k(\Delta)\approx\tfrac12 k_{\text{geo}}(\Delta)+\tfrac12\delta_{\Delta,0}$. The split ratio
is a hyperparameter.

### 2.4 Proposition 2 — QK-norm is torus projection, and attention on the torus is L2 attention

Normalize queries and keys plane-wise, $q,k\in T^m$. Then $\|q\|^2=\|k\|^2=m$ and

$$\|q-k\|^2=2m-2\,\mathrm{Re}\langle q,k\rangle\;\Longrightarrow\;\mathrm{softmax}_k\big(\beta\,\mathrm{Re}\langle q,k\rangle\big)=\mathrm{softmax}_k\big(-\tfrac{\beta}{2}\|q-k\|^2\big),$$

because the constant $2m$ is shared by all keys and cancels. So dot-product attention on
the torus *is* L2 attention. Kim et al. (2021) showed that unconstrained dot-product
self-attention is not Lipschitz: its Jacobian grows with activation magnitude. Here the
logits satisfy $|\beta\,\mathrm{Re}\langle q,k\rangle|\le\beta m$, so the Jacobian is bounded
by quantities the model controls ($\beta$, $m$, number of keys), never by how large
activations grow. Modern LLMs add QK-norm for training stability (ViT-22B, OLMo 2);
plane-wise QK-norm is its phasor form.

### 2.5 Proposition 3 — one operator reads everything

Define torus attention over a codebook of keys $K$ and values $V$:

$$\mathcal{A}(q;K,V)=\sum_c \mathrm{softmax}_c\big(\beta\,\mathrm{Re}\langle q,k_c\rangle\big)\,v_c.$$

- **PEM's resonator step** (cleanup toward candidate rules) is $\mathcal{A}(z;S,S)$ with
  $S$ the rule signatures. It is attention whose keys and values are a fixed codebook.
- **Self-attention** is $\mathcal{A}$ with the token field as its own codebook:
  relational inference.
- **Product-key memory** (§5) is $\mathcal{A}$ with a learned codebook of $N$ entries,
  restricted to a sparse top-$k$: knowledge recall.
- **The semi-embedder** (§6) is $\mathcal{A}$ with learned query phasors: thought
  compression.

Relational inference, hypothesis cleanup, knowledge recall and thought compression are
one operation with different codebooks. This is invariant I1.

### 2.6 Real implementation

Planes are stored as interleaved (cos, sin) pairs. Binding is a batched 2×2 rotation and
$\Pi_\varepsilon$ is a per-pair scale. Phases are never parameterized directly (no
wrap-around gradients): codebooks are learned as unconstrained pairs projected by
$\Pi_\varepsilon$. Everything is plain PyTorch autograd, with bf16 autocast on the matmuls
and fp32 inside the projection.

---

## 3. Perception: the inductive task field

### 3.1 Tokens

A token has content $c$, coordinates $p=(x,y,z,e)$ (grid position, depth, example index)
and a role $r\in\{\text{IN},\text{OUT},\text{QUERY},\text{ANSWER}\}$. Its value is

$$u=E[c]\circledast\rho_r,\qquad E\in T^{|\mathcal{V}|\times m}\text{ (learned codebook)},\;\rho_r\in T^m\text{ (role phasors)}.$$

Coordinates enter only through $\mathrm{RoPE}_a$ inside attention (Proposition 1).

- **Discrete content** (colors, digits, bytes) uses codebook rows.
- **Continuous content** (a vector component $v$) uses fractional power encoding,
  $E(v)=C^{v}$ (PEM §2.2). It is decoded by cleanup over a value lattice.

### 3.2 The task field is inductive

All demonstration pairs, the query input, and blank answer slots (content $\varnothing$)
form one token set $X$ of size $n$. There is no task identifier: the model must infer
the transformation from the demonstrations. This answers ARC-Prize finding #3 (HRM's
`puzzle_id` makes it transductive and unable to handle unseen tasks). It costs
difficulty: HRM's authors report that puzzle embeddings beat few-shot context on small
ARC data. §12 A8 measures both protocols honestly.

### 3.3 Augmentation is a group action

$G=D_4\times S_{\text{colors}}$ (× axis permutations in 3-D) acts on coordinates (the
$\mathrm{RoPE}_a$ argument) and on content (a permutation of codebook rows). Training
samples $g\in G$ per episode; the ARC Prize found about 300 augmentations suffice. At test
time the model predicts under $|G'|$ views, de-augments, and votes. **Confidence** is the
fraction of views agreeing with the winner. It is agreement between independent
derivations, not a self-reported score.

### 3.4 Prelude

$e=\Pi_\varepsilon\big(P(u)\big)\in T^{n\times m}$, with $P$ = $l_P$ blocks of torus
attention and a SwiGLU MLP.

---

## 4. The reasoning core Φ

### 4.1 Two streams

- $z_H\in T^{n\times m}$: the **solution** stream (TRM's $y$; the mechanistic study
  finds the solution lives here).
- $z_L\in T^{n\times m}$: the **working** stream (TRM's $z$; transient).

Both are initialized to a fixed phasor field (learned, input-independent), plus small
noise during training (path independence, Anil et al. 2022).

### 4.2 One block, two roles

Let $F_\theta$ be a block (torus self-attention + SwiGLU, pre-$\Pi_\varepsilon$), and let
$\rho_L,\rho_H\in T^m$ be role phasors:

$$\textbf{L-tick:}\quad z_L\leftarrow\Pi_\varepsilon\Big((1-\alpha)\,z_L+\alpha\,F_\theta\big(\rho_L\circledast\Pi_\varepsilon(z_L+z_H+e)\big)\Big)$$

$$\textbf{H-tick:}\quad z_H\leftarrow\Pi_\varepsilon\Big((1-\beta)\,z_H+\beta\,\big[F_\theta\big(\rho_H\circledast\Pi_\varepsilon(z_H+z_L)\big)+\gamma\,\mathcal{M}(z_H)\big]\Big)$$

- **Bundling of the inputs**, $\Pi_\varepsilon(z_L+z_H+e)$, is TRM's `x + y + z`: TRM's
  input combination *is* VSA superposition.
- **Role binding** lets one set of weights compute two functions: $F_\theta$ is not
  rotation-equivariant, so $\rho_L$ and $\rho_H$ present it with differently rotated
  inputs. TRM found one shared network generalizes better than HRM's two; binding is how
  one network holds two roles.
- $\alpha,\beta\in(0,1)$ are learned (sigmoid, clamped to $[0.05,0.95]$). $\gamma$ is a
  learned gate on the memory read.
- **Units.** A cycle is $n_L$ L-ticks and one H-tick; a segment is $T$ cycles; an
  episode is up to $N_{\text{sup}}$ segments with adaptive halting.

### 4.3 What the hierarchy is — honestly

The ARC Prize found HRM's hierarchy worth little over a same-size transformer. TRM found
that what matters is separating a working state from a solution state and refining
recurrently. HRM-Φ keeps exactly that separation. The timescale ratio $n_L$ is a measured
hyperparameter, and §12 A2 decides whether "hierarchy" means more than two streams.

### 4.4 Convergence is measured, not assumed

Let $\Phi$ be one cycle and define the residual and the empirical contraction rate:

$$r_t=\frac{\|\Phi(Z_t)-Z_t\|}{\sqrt{nm}},\qquad \hat\rho=\Big(\prod_{t\in\text{tail}}\frac{r_{t+1}}{r_t}\Big)^{1/|\text{tail}|}.$$

**Proposition 4 (averaged update).** In Euclidean space, $U(z)=(1-\alpha)z+\alpha F(z)$
has $\mathrm{Lip}(U)\le 1-\alpha+\alpha\,\mathrm{Lip}(F)$.
- If $\mathrm{Lip}(F)\le1$, $U$ is $\alpha$-averaged, and Krasnoselskii–Mann iteration
  converges to a fixed point when one exists.
- If $\mathrm{Lip}(F)<1$, $U$ is a contraction with rate $1-\alpha+\alpha\,\mathrm{Lip}(F)$.

**The projection breaks the theorem.** On planes of pre-projection magnitude $\ge r$,
$\Pi_\varepsilon$ is only $1/r$-Lipschitz. Averaging two unit phasors can produce
magnitudes as small as $|1-2\alpha|$, so the projected iteration is not provably
non-expansive. PEM v1.1 claimed provable convergence on the torus; that claim silently
ignored the projection. HRM-Φ therefore claims no theorem. Instead:

- **Training** uses truncated backpropagation through the last cycle (TRM, Geiping).
  There is no implicit-function-theorem gradient.
- **Contraction is encouraged, not assumed**, with a Jacobian penalty
  $\lambda_J\,\mathbb{E}\|J_\Phi v\|^2$ (Hutchinson estimate, Bai et al. 2021). A
  contractive core is path-independent and extrapolates to more test-time compute.
- **Contraction is certified per file.** The core file stores $\hat\rho$, the residual
  curve, and the spectral-norm estimate of $J_\Phi$ at the final state (power iteration
  with JVPs) on bundled probe inputs, and the loader re-measures them (§8.2).

### 4.5 Halting and the four rails

- **Halting head** (TRM single pass): $q=\sigma(w^\top\mathrm{pool}(z_H))$, trained with
  BCE against $\mathbb{1}[\hat y=y]$.
- **Leave-one-out energy.** $E_{\text{loo}}$ counts demonstrations whose output the model
  mispredicts when that output is hidden and treated as answer slots. It is exact,
  label-free at test time, and it is the model's own verification.

| Action | Condition |
|---|---|
| ACCEPT | $q>\tfrac12\ \wedge\ r<\varepsilon_r\ \wedge\ E_{\text{loo}}=0$ (the last term when demonstrations exist) |
| CONTINUE | otherwise, while the segment budget lasts |
| RESTART | $r<\varepsilon_r\ \wedge\ E_{\text{loo}}>0$ (false attractor): $Z\leftarrow\Pi_\varepsilon(Z\circledast e^{i\sigma_k\xi})$, $\sigma_k=\sigma_0\gamma^k$ |
| ABSTAIN | restart budget exhausted with $E_{\text{loo}}>0$ |

ACCEPT and ABSTAIN stay rails in code (PEM R-impl-10); the learned head only times
CONTINUE. Tasks without demonstrations, such as free text, have no exact energy. Their
answers carry the flag *unverified*.

### 4.6 Structured head

$$\mathrm{logits}_{i,c}=\beta_o\,\mathrm{Re}\langle (z_H)_i,\;E[c]\circledast\rho_{\text{ANSWER}}\rangle.$$

Output is cleanup against the **same** codebook as the input: tied, no separate output
matrix. Continuous values decode by cleanup over the value lattice $C^{v}$.

---

## 5. Knowledge M: parametric memory on SSD (the second file)

### 5.1 Product-key addressing

Per solution token $i$: $q=\Pi_\varepsilon(W_q (z_H)_i)\in T^{d_k/2}$, split into halves
$q^{(1)},q^{(2)}$. There are two sub-codebooks $K^{(1)},K^{(2)}\in T^{s\times d_k/4}$ and
$N=s^2$ slots. Addressing:

$$I^{(a)}=\mathrm{top}_k\big(\beta_k\,\mathrm{Re}\langle q^{(a)},K^{(a)}\rangle\big),\qquad \mathcal{I}=\mathrm{top}_k\big\{s^{(1)}_a+s^{(2)}_b:(a,b)\in I^{(1)}\times I^{(2)}\big\},$$

$$\mathcal{M}(z_H)_i=W_o\sum_{(a,b)\in\mathcal{I}}\mathrm{softmax}(s)\,V[a\,s+b].$$

Cost per token is $O(s\,d_k+k^2+k\,d_v)$ instead of $O(N d_v)$. Only $k$ value rows are
touched, so $V$ can live on SSD and be paged on demand.

### 5.2 Values are parameters

Training updates only the touched rows (sparse gradients, row-wise sparse Adam). The
sub-keys and $W_q, W_o$ are small, dense, and belong to the core.

### 5.3 Test-time learning, PEM's gate at parameter level

$E_{\text{task}}$ is the cross-entropy at demonstration OUT tokens, available at test
time. Surprise is $S=\|\partial E_{\text{task}}/\partial\mathcal{M}\|$. If
$S>\tau_S$, form a tentative update in a RAM overlay:

$$\Delta V_{\mathcal{I}}\leftarrow-\eta\big(\mu\,\Delta V_{\mathcal{I}}^{\text{prev}}+\nabla_{V_{\mathcal{I}}}E_{\text{task}}\big)\odot(1-\iota_{\mathcal{I}}),$$

where $\iota$ is a per-row importance built from usage and confidence (an EWC-like shield
for well-established knowledge). Re-run the episode:
- If $E_{\text{loo}}=0$, **commit** the rows to the SSD file (journaled).
- Otherwise discard the overlay.

Guesses never persist (I4). This is Titans' surprise-gated memorization with PEM's
verified-write rule.

### 5.4 Sleep consolidation (gated)

Rows with high usage and verified success may be distilled into the core: fine-tune the
core on replay with those rows masked, then zero the rows and reset their importance.
Promotion passes only through the existing frozen, hash-locked benchmark gate
(`hrm/learning/harness.py`), with keep/discard logged to the ledger. Without
consolidation the memory simply grows fuller; $N$ is fixed, so nothing grows on disk.

### 5.5 The empty brain

$V\equiv0\Rightarrow\mathcal{M}\equiv0\Rightarrow$ Φ reasons purely in context. The core
without its knowledge file is a complete reasoner that knows nothing it was not shown.
That is I3, and it is testable.

### 5.6 Capacity

$N\,d_v\cdot2$ bytes: $65{,}536\times512\times2=64$ MiB by default, and $2^{20}$ slots
$=1$ GiB. Capacity is a dial fixed at creation.

---

## 6. Language: the semi-embedder and the coda, subordinate to Φ

### 6.1 The thinking chain

$\mathcal{Z}=(z_H^{(1)},\dots,z_H^{(S)})$ is the solution stream at each segment
boundary: the model's latent thoughts, in order.

### 6.2 Semi-embedder Σ

It uses $k_\tau$ learned query phasors $Q\in T^{k_\tau\times m}$ (a resonator over the
token field, Proposition 3):

$$\tau^{(s)}=W_\Sigma\,\mathcal{A}\big(Q\circledast\Theta^{s};\,z_H^{(s)},\,z_H^{(s)}\big)\in\mathbb{R}^{k_\tau\times d_c},$$

where $\Theta^s$ binds the thought-time index (RoPE over segments). It is *semi* in two
ways: it keeps the phase geometry until the last linear map $W_\Sigma$, and it compresses
$n$ tokens to $k_\tau$ thought tokens per segment. The result is a short sequence of
$S\cdot k_\tau$ vectors in the coda's width.

### 6.3 Coda C

$l_C$ causal transformer layers of width $d_c$ (RMSNorm, SwiGLU, RoPE over text
positions), each with cross-attention to the thought tokens. Output logits are tied to
the prelude's byte codebook, so text enters and leaves through one vocabulary. This is
the recurrent-depth LM shape (prelude → recurrent core → coda) with HRM-Φ as the core.

### 6.4 Gradient policy

The text cross-entropy trains Σ and C. It reaches Φ only through the final segment,
scaled by $\lambda_\ell\in[0,1]$ (A5). Language may shape *how thoughts are readable*,
never the dynamics that produce them.

### 6.5 Why the model does not think in words

The (est.) figures below use the default config ($n\approx300$ tokens,
$P_{\text{core}}\approx6$M) against a 7B LLM producing 2,000 chain-of-thought tokens.

| | latent tick (HRM-Φ) | chain-of-thought token (LLM) |
|---|---|---|
| FLOPs per step | $2P_{\text{core}}n\approx3.6$ GFLOP | $2P_{\text{LLM}}+4L d\,t$, growing with $t$ |
| memory while thinking | constant: $2nm$ floats ≈ 0.6 MB | KV cache grows $2Ld$ per token (≈1 GB at 2k tokens) |
| information per step | $n\cdot m$ continuous phases; a superposition of hypotheses (SNR $\sqrt{m/N}$) | one sampled token, $\le\log_2|\mathcal{V}|\approx17$ bits, one committed path |
| episode | 336 ticks ≈ 1.2 TFLOP (est.) | ≈ 28 TFLOP (est.) |

Thinking in words pays the full model per step, discretizes after every step, and grows
memory with length. The coda speaks *once*, after thinking.

### 6.6 Authority rule

The coda verbalizes; it never decides. Any answer stated in text is checked against the
structured head. Disagreement is scored as a coda error, never as a model answer.

---

## 7. Training

### 7.1 Objective

For each episode, run segments $s=1..N_{\text{sup}}$:
- cycles $1..T-1$ under `no_grad`, cycle $T$ with gradients;
- then the loss, then detach both streams.

$$\mathcal{L}=\sum_{s}\Big[\mathrm{CE}(\hat y^{(s)},y)+\lambda_q\,\mathrm{BCE}(q^{(s)},\mathbb{1}[\hat y^{(s)}=y])\Big]+\lambda_\ell\,\mathrm{CE}_{\text{coda}}+\lambda_J\,\widehat{\|J_\Phi\|}_F^2+\lambda_d\,\mathcal{L}_{\text{InfoNCE}}$$

$\mathcal{L}_{\text{InfoNCE}}$ is PEM's anti-collapse term: final states of different
tasks are pushed apart, and two augmented views of one task are pulled together.

**Leave-one-out training.** Each episode hides one demonstration output as an extra
answer target, so rule induction from $J-1$ examples is trained directly, and
$E_{\text{loo}}$ at test time is in-distribution.

### 7.2 Optimization

AdamW ($\beta_2=0.95$), 2k-step warm-up, EMA 0.999 (TRM), bf16 autocast, stablemax loss
optional. Recurrence depth is sampled per batch (Geiping): $N_{\text{sup}}$ from a
heavy-tailed distribution so inference can extrapolate.

### 7.3 Curriculum

| Stage | Data | Purpose / exit |
|---|---|---|
| C0 | substrate property tests | Propositions 1–4 hold numerically |
| C1 | generated exact families: grid, sequence, vector3d (infinite, verified, depth-labelled) | learns latent refinement; depth-4/5 families the symbolic system cannot reach (ADR-004) |
| C2 | Sudoku-Extreme, Maze-Hard (HRM's datasets) | direct comparison with HRM 55% / TRM 87% on the same protocol |
| C3 | ARC-AGI-1 training (inductive protocol, with augmentation) | the honest target; compared with transductive in A8 |
| L | program → description pairs from the symbolic teacher; a small byte-level corpus | coda fluency and faithfulness |

### 7.4 The symbolic system's new role

The current `hrm/` keeps four jobs:
- **Data engine**: exact, depth-controlled tasks without end.
- **Judge**: the frozen, hash-locked harness and gate.
- **Baseline**: program search against learned refinement on benchmark v2's depth bands.
- **Language teacher**: verified programs render into descriptions.

It is repositioned, not deleted.

---

## 8. The two files

### 8.1 Core file `*.hrmc` — why a custom format

A standard checkpoint stores tensors. This model has four things that are not tensors,
and the format exists to carry them:

1. **The law, not the table.** Codebook phases for coordinates and roles come from a
   seeded generator plus a spectrum spec. The file stores the generator, and the loader
   regenerates the phases bit-exactly.
2. **A certificate.** $\hat\rho$, the residual curve and the Jacobian norm are measured
   on bundled probe inputs, and the loader re-measures them. A file that no longer
   thinks the way it did when saved is refused.
3. **Pairing.** It holds the knowledge schema hash that its knowledge file must match.
4. **Lineage.** Generation, parent hash, benchmark lock and gate result.

It is also **pickle-free**: loading an untrusted core cannot execute code (a `.pt` relies
on a restricted unpickler for the same guarantee).

| offset | content |
|---|---|
| 0 | magic `HRMCORE\0` (8 B), `u16 major`, `u16 minor`, `u32 header_len` |
| 16 | header: canonical UTF-8 JSON, strict schema (unknown or missing keys refused): architecture, substrate generator, tensor directory `name → {dtype, shape, offset, nbytes, sha256}`, knowledge schema + hash, certificate, lineage |
| next 4096 boundary | tensors, each 4096-aligned (mmap- and O_DIRECT-friendly, kimi-k3 style) |
| end − 32 | SHA-256 of all preceding bytes |

### 8.2 Loader contract (core)

1. Check the magic and version.
2. Check the whole-file SHA-256.
3. Validate the header strictly.
4. Map the tensors and verify per-tensor hashes.
5. Regenerate the substrate from its generator.
6. Run the probe self-test: re-measure $\hat\rho$ and the residual curve, and compare
   against the certificate within tolerance.
7. Check that the knowledge file's schema hash matches.

Any failure refuses the load with the reason.

### 8.3 Knowledge file `*.pt` — a normal `.pt`, handled as unified memory

It is a standard PyTorch zip archive:
`{schema, sub_keys_1, sub_keys_2, values (N×d_v fp16), usage, confidence, row_gen}`.

- **Portable read:** `torch.load(path, mmap=True, weights_only=True)` works anywhere.
- **Runtime read/write:** payload offsets are resolved from the zip directory and mapped
  `MAP_SHARED`, so rows are read from and written to SSD in place. This was measured on
  torch 2.11 (Windows): payloads are stored uncompressed, 64-byte aligned, and in-place
  writes are visible to `torch.load`.
- **Tensor location:** each payload is located by its unique byte size, then confirmed
  by comparing sampled rows against a `torch.load` view.
- **Integrity:**
  - a write-ahead journal `*.pt.wal` holds (row, generation, old, new, crc);
  - `seal()` recomputes the zip CRC-32 of modified entries, so the archive stays valid
    for any zip tool;
  - on open, an unsealed journal is replayed or rolled back.
- **Size:** fixed at creation ($N$ slots). Resizing is an offline migration.

### 8.4 One mind, two files

The same core can pair with different knowledge files (per domain, per user). The
knowledge file without its core is inert: vectors with no reader.

---

## 9. Hyperparameters (initial; sized for one 12 GB GPU)

| Symbol | Meaning | Default (est.) |
|---|---|---|
| $m$ | planes per token (width $2m$) | 256 (512 real) |
| heads | torus-attention heads | 8 |
| $l_P$ / core / $l_C$ | prelude / shared core / coda layers | 1 / 2 / 2 |
| $d_c$ | coda width | 256 |
| $k_\tau$ | thought tokens per segment | 16 |
| $n_L$, $T$, $N_{\text{sup}}$ | L-ticks per cycle, cycles per segment, max segments | 6, 3, 16 (TRM) |
| $\alpha,\beta$ | stream update rates | learned, init 0.5, clamp [0.05, 0.95] |
| $\varepsilon$ | projection smoothing | 0.05 |
| spectrum | geometric : random planes | 1 : 1 |
| $s$, $N$ | sub-keys per half, memory slots | 256, 65,536 |
| $d_k$, $d_v$, top-$k$ | key dim, value dim, rows read | 128, 512, 32 |
| $\sigma_0,\gamma$, restarts | restart kick, anneal, budget | 0.3 rad, 0.7, 4 |
| $\lambda_q,\lambda_\ell,\lambda_J,\lambda_d$ | loss weights | 0.5, 0.1, 1e-3, 0.05 |
| optimizer | AdamW lr, warm-up, EMA | 1e-4, 2k, 0.999 |
| parameters | prelude + core + Σ + coda | ≈ 12M dense + 33M memory (est.) |

---

## 10. Pseudocode

```python
def episode(X, core, memory, cfg, train):
    e = prelude(X)                                   # §3.4, T^{n x m}
    zL, zH = core.init_states(X)
    for s in range(cfg.N_sup):
        for t in range(cfg.T):
            grad = train and t == cfg.T - 1          # TRM: only the last cycle
            with torch.set_grad_enabled(grad):
                for _ in range(cfg.n_L):
                    zL = proj((1-a)*zL + a*F(bind(rhoL, proj(zL + zH + e))))
                zH = proj((1-b)*zH + b*(F(bind(rhoH, proj(zH + zL))) + g*memory(zH)))
        y_hat, q = head(zH), halt(zH)
        yield s, y_hat, q, residual(zL, zH)          # deep supervision per segment
        zL, zH = zL.detach(), zH.detach()
        if not train and accept(q, residual, loo_energy(X, core, memory)):
            break

def test_time_write(X, core, memory, cfg):
    overlay = memory.overlay()                       # RAM only
    if surprise(X, core, overlay) > cfg.tau_S:
        overlay.step(grad_demo_loss(X, core, overlay))
        if loo_energy(X, core, overlay) == 0:        # verified: all demos exact
            memory.commit(overlay)                   # journaled write to SSD
    overlay.discard()
```

---

## 11. Risk register

| ID | Risk | Detection | Mitigation |
|---|---|---|---|
| R1 | the inductive protocol scores far below HRM/TRM's transductive numbers | A8 on the same data | report both protocols; never compare across protocols |
| R2 | the torus state loses useful magnitude information | A1 vs an RMSNorm state | keep a magnitude channel (one real plane) if A1 says so |
| R3 | $O(n^2)$ attention over demonstrations (ARC: up to 8,100 tokens) | step time vs $n$ | start ≤ 10×10; then gated-delta linear attention (KDA-style) for L-ticks |
| R4 | memory collapse (few slots used) | slot-usage entropy | per-row importance, usage-balanced key reset (Memory Layers at Scale) |
| R5 | test-time writes corrupt knowledge | regression on held-out after writes | verified-commit rule, journal, rollback, importance shield |
| R6 | the coda states wrong answers fluently | text-vs-head agreement metric | authority rule §6.6; answers come only from the head |
| R7 | laptop-GPU training budget | wall-clock per stage | TRM-size model; curriculum starts on cheap generated tasks |
| R8 | stale zip CRCs after in-place edits | `zipfile` test in CI | `seal()`, journal |
| R9 | core/knowledge pairing drift | schema hash check | refuse mismatched pairs |
| R10 | "hierarchy" adds nothing | A2 | report it; keep the two-stream split TRM shows matters |
| R11 | an unfalsifiable efficiency claim | §6.5 table vs measurement | measure FLOPs and memory per tick in A6 |

---

## 12. Ablation ladder (each rung gates the next; every criterion is machine-decidable)

| Rung | Question | Pass criterion |
|---|---|---|
| A0 | do the substrate identities hold? | Props 1–2 numerically equal to 1e-5; `gradcheck` on $\Pi_\varepsilon$ |
| A1 | state manifold: torus vs RMSNorm vs none | held-out accuracy on C1 at matched params; residual curves |
| A2 | two streams + roles vs one stream vs two separate nets | Δ accuracy on C1/C2 at matched compute |
| A3 | gradient: 1-step (HRM) vs last-cycle (TRM) vs last-k | Δ accuracy; stability (loss spikes) |
| A4 | memory: none vs product-key vs + verified test-time writes | accuracy on cross-episode recall families; zero regression after writes |
| A5 | coda gradient $\lambda_\ell\in\{0, 0.1, 1\}$ | text/head agreement without loss of head accuracy |
| A6 | compute scaling | accuracy rises with inference segments beyond the training mean |
| A7 | learned HRM-Φ vs symbolic program search (benchmark v2 depth bands) | HRM-Φ solves depth-4/5 where `synthesis_depth=3` cannot |
| A8 | inductive vs transductive (puzzle-ID) on ARC-AGI-1 | both reported, same data, same compute |

---

## 13. Honest limitations

1. **Unvalidated composition.** A1, A2 and A7 are where the design most plausibly fails.
2. **Verification is only as strong as the task.** Demo-conditioned tasks have an exact
   leave-one-out energy. Free text does not, and its answers stay *unverified*.
3. **Transduction is still learned-in.** A neural refiner outputs answers, not programs.
   The program remains implicit (ARC Prize's concern), which is why the symbolic system
   stays as the inductive baseline and teacher.
4. **Scale.** A 12M-parameter core on one laptop GPU is TRM-sized. Language fluency from
   a 2-layer coda will be limited. The design scales the coda later without touching Φ.
5. **"Understanding".** Measured claims only: exact leave-one-out verification,
   augmentation agreement, residual and contraction certificates, abstention.

---

## 14. References

- Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734 (2025).
- ARC Prize Team, *The Hidden Drivers of HRM's Performance on ARC-AGI*, arcprize.org/blog/hrm-analysis (Aug 2025).
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, arXiv:2510.04871 (2025).
- *Dissecting Hierarchical Reasoning Models: A Mechanistic Study*, arXiv:2609.22197 (2026).
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, arXiv:2502.05171 / NeurIPS 2025.
- Berges et al., *Memory Layers at Scale*, ICML 2025 (arXiv:2412.09764); Lample et al., *Large Memory Layers with Product Keys*, NeurIPS 2019.
- Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663 (2025).
- Su et al., *RoFormer: Rotary Position Embedding* (2021); Dehghani et al., *Scaling ViT to 22B* (QK-norm, 2023).
- Kim, Papamakarios, Mnih, *The Lipschitz Constant of Self-Attention*, ICML 2021.
- Bai, Kolter, Koltun, *Deep Equilibrium Models* (2019); Bai et al., *Stabilizing Equilibrium Models by Jacobian Regularization* (2021).
- Anil et al., *Path Independent Equilibrium Models Can Better Exploit Test-Time Computation* (2022).
- Plate, *Holographic Reduced Representations*; Komer et al., *Spatial Semantic Pointers* (fractional power encoding).
- PEM v1.1 design document (local: `C:\Users\Admin\Desktop\Code\AI\PEM`).

*End of document.*
