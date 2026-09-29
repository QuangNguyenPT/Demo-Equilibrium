---
title: "HRM-X Specification v0.3 — Understanding by Construction"
type: specification
tags:
  - architecture
  - specification
  - mathematics
  - hrm-x
  - understanding
status: superseded
date: 2026-09-27
supersedes: "[[HRM-EQB Specification v0.2 - Theorems and Proofs]] (as the reasoning design; its algebra, token field and runtime rules are kept)"
superseded_by: "HRM-X Specification v0.4 - Operator Inference over Structured Situations (ADR-008)"
related:
  - "[[ADR-007 Grounded Algebra and Executable Concepts]]"
  - "[[ADR-006 HRM-EQB Training and Benchmark Results]]"
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
---

# HRM-X — Understanding by Construction
## Specification v0.3 — every section a formula, every theorem a capability

> [!WARNING] Superseded — never built
> v0.3 was a **specification only**; it was superseded by
> [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
> ([[ADR-008 Operator Inference over Structured Situations]]), which is the
> normative design and *was* built and measured in `hrm_x/`. The
> [[ADR-007 Grounded Algebra and Executable Concepts]] redesign it proposed
> became v0.4. The diagnosis in Part I below is still a valid record of the
> `hrm_eqb/` measurements that motivated the change.

> **One-line thesis.** Learn *conventions* by pattern and derive *necessities* by structure.
> The word "seven" and the phrase "times" are conventions, so a transformer learns them.
> $7 \times 8 = 56$ is a necessity, so it is computed in an algebra where multiplication *is*
> repeated addition, by theorem. No parameter stores arithmetic, so none can memorize it.

---

## Part I — Diagnosis of the trained HRM-EQB (measured, 2026-09-27)

`models/eqb-small`: 18,233 steps, 150 min, core sha256 `2bb582c5280bf980`. See
[[ADR-006 HRM-EQB Training and Benchmark Results]] for the benchmark table.

| # | Finding | Evidence |
|---|---|---|
| D1 | **Numbers are atomic symbols.** Every integer −64…64 has its own unrelated embedding; 65 does not exist. | `hrm_eqb/field.py:21-22` (`INT_MIN, INT_MAX = -64, 64`) |
| D2 | **It learned a fuzzy number line, not arithmetic.** Embedding similarity falls with distance, but $e(a{+}b)-e(a) \ne e(b)-e(0)$. | Probe: corr(similarity, −\|a−b\|) = **+0.44**; additive-consistency error **1.39** (random vectors: 1.41; an additive code: 0) |
| D3 | **Fixed, shallow computation.** Certified rate $r_n = 0.304$, so every solve converges in 4–6 cycles *whatever the task*. Routing is fixed within a solve, so data-dependent hops = 3 prelude layers + 3 segments = **6**. | `info`: Γ = 0.214; training log `cycles 5` on every line; `model.py:163-166` |
| D4 | **A contraction cannot search.** E2′ (start state irrelevant) means one fixed point, i.e. one hypothesis. Search needs alternatives and backtracking. | Sudoku blank-cell accuracy 0.125 vs chance 1/9 = 0.111 (`bench.py:135`); ARC 0/158 |
| D5 | **Procedures whose length grows with the input fail.** | `sequence/scale` 0.00, `cumsum` 0.00, `sort` 0.05, all 8 grid families 0.00 |
| D6 | **Most mechanisms are unmeasured or measured useless.** The memory (291k params) contributes exactly 0. The reasoning core is 131k of 1.87M params (7 %); prelude + coda are 70 %. | `empty_brain` = `generated` = 0.298; parameter census |
| D7 | **No channel for definitions.** Training is rule induction from demos over atomic tokens, so "× means repeated +" has no way in. | `hrm_eqb/tasks.py` families |

What it *did* learn — `text/*` ≈ 1.00, `language` 0.983 — is per-position symbol mapping.
An earlier session measured that, per training step, the full model only keeps pace with a
prelude-only baseline ([[Session - 2026-09-27 - HRM-EQB Full Build]]). The certified core
added nothing measurable.

### Why it feels like patching

Every theorem in v0.2 is a **safety** theorem: convergence (C1, E1–E6), boundedness (K2,
A1, H1), non-expansiveness (S1, S2, N1), atomic writes (F3). None is a **capability**
theorem of the form "the model computes $f$ correctly for all inputs of kind $K$". Each
defect found in the build received a local fix plus a local lemma (§14 gained 11 of them).
The mechanisms are individually proven safe and jointly unmeasured. Nothing in the design
says what the model must be able to do, so there is nothing to derive the architecture from.

v0.3 inverts the order:
1. Define understanding formally (Part II).
2. Derive the smallest architecture that satisfies the definition (Part III).
3. Make every theorem a capability claim, each with a test (Part IV).

**Why not "bigger model, longer training"?** D1 and D3 are representational. No amount of
training creates a token for 65, or makes a 5-cycle contraction take $a$ steps for $a \times b$.
More data only improves in-distribution accuracy, which is exactly the pattern regime.

---

## Part II — What "understanding" means

**Domain.** $\mathfrak D = (V, O)$: values with operations. Examples:
- $(\mathbb Z, 0, \mathrm{succ}, +, \times)$;
- weekdays $(\mathbb Z/7, \mathrm{next})$;
- strings (concat, reverse, upper);
- grids (translate, rotate, recolor).

**Definition 1 (grounding).** A model grounds $\mathfrak D$ if it has a decodable encoding
$\varphi: V \to \mathcal Z$ and, for every primitive $o \in O$, a latent operator $\hat o$ with
$$\varphi(o(v_1,\dots,v_k)) = \hat o(\varphi(v_1),\dots,\varphi(v_k)) \quad \text{for all } v \in V.$$
In other words, $\varphi$ is a homomorphism. "For all" is the whole point: it covers inputs
the model has never seen.

**Definition 2 (understanding a concept).** A model understands concept $c$ if:
1. its knowledge holds a *definition* $B_c$: a composition of grounded primitives and concepts
   already understood;
2. its answer to any query about $c$ is produced by *executing* $B_c$, or by a compiled form
   proven or verified equal to it;
3. it can emit that execution as an explanation.

**Observable criteria.** These are the benchmark in Part V.

| | Criterion | A pattern learner… | An understander… |
|---|---|---|---|
| U1 | **Extrapolation**: accuracy independent of operand size and length | collapses outside the training range | is flat up to the algebra's range |
| U2 | **Definitional transfer**: a concept given only by definition is used correctly with zero examples | cannot, since it needs examples | can: definition in, answers out |
| U3 | **Faithful explanation**: the explanation replays to the answer | rationalizes after the fact | serializes the actual computation |
| U4 | **Honest ignorance**: undefined or ungrounded input gives a question, not a guess | guesses fluently | asks |

**Principle P (division of labour).** Anything that could have been otherwise is learned by
pattern (the transformer): word forms, syntax, which operation a sentence asks for, how to
phrase the answer. Anything that could not have been otherwise is derived by structure (the
algebra and the executor): that $2 \times 3 = 3 + 3 = 6$.

---

## Part III — The model

```
 English text x ──► (§3) Reader R_θ ──► program P ─┐      (transformer: pattern IN)
                    (transformer)                  │
                                                   ▼
          (§2) Knowledge 𝒦 ◄──────────► (§4) Thinker ℰ  = H-level: slow, exact, depth = problem
          concept library      unfold   │    ▲       (executor over the algebra §1)
          (data, not weights)           ▼    │ call
                                (§5) Skills + cleanup  = L-level: fast, automatic
                                        │
                                        ▼
                                (§6) Judge 𝒱 ── accept / ask a question
                                        │
                     trace τ, value y   ▼
 English answer ◄── (§8) Writer W_θ ◄──┘                (transformer: pattern OUT)
 + explanation
                     (§9) Learning: wake (solve) · sleep (dream, abstract, compile)
```

HRM's two timescales remain, and each now has a job:
- **H** is the executor stepping through definitions: slow, explicit, planning.
- **L** is compiled skills and cleanup attractors: fast, automatic.

### §1 The grounded algebra $\mathcal A$ (0 learned parameters)

**Carrier.** $\mathcal Z = \mathbb C^D$, unit phasors $\mathcal U = \{z: |z_j| = 1\}$.

**Operations.**
- bind $z \circledast w = z \odot w$;
- unbind $z \circledast \bar w$;
- power $z^{\circ k}$ (elementwise);
- bundle $\sum_i z_i$;
- similarity $s(z, w) = \mathrm{Re}\langle z, w\rangle / D$.

This is the FHRR substrate `hrm/` already has. The difference is that computation now
*happens* in it. ADR-004 found it was decoration.

**Integers: residue phasors.** Pick pairwise-coprime moduli $p_1..p_r$ with
$M = \prod_c p_c$, $R$ replicas per modulus, and multipliers $q_{c,j} \in (\mathbb Z/p_c)^\times$:
$$\varphi(n)_{c,j} = \exp\!\big(2\pi i\, q_{c,j}\, n / p_c\big), \qquad D = rR.$$
Default: $p = (7,11,13,17,19,23)$ and $R = 16$, giving $D = 96$ complex (192 real) and
$M = 7{,}436{,}429$ (range $\pm 3.7\text{M}$).

**A1 — Addition is binding (exact homomorphism).** For all $a, b \in \mathbb Z/M$:
$\varphi(a+b) = \varphi(a) \circledast \varphi(b)$, $\varphi(-a) = \overline{\varphi(a)}$,
$\varphi(0) = \mathbf 1$.
*Proof.* $e^{i\theta(a+b)} = e^{i\theta a}e^{i\theta b}$ componentwise, and $n \mapsto e^{2\pi i qn/p}$
is $p$-periodic. ∎
*Measured:* max error $1.7\times10^{-8}$ over 1,000 pairs up to $10^6$.

**A2 — Multiplication is repeated addition, and its compiled form is equal by theorem.**
Let $R_b(z) = z \circledast \varphi(b)$ ("add $b$"). Then for $a \ge 0$
$$R_b^{\,a}(\varphi(0)) = \varphi(ab) = \varphi(b)^{\circ a}.$$
*Proof.* Induction on $a$ using A1:
$R_b^{a+1}(\varphi(0)) = \varphi(ab) \circledast \varphi(b) = \varphi(ab + b)$.
The right-hand equality is the exponent law. ∎

So "2 × 3 is 3 + 3" is an identity *in the representation*. The one-step compiled form
(power) agrees with the definition because the mathematics forces it, not because training
made it so.
*Measured:* $20000 \times 150$ by 20,000 repeated bindings gives 3,000,000. It equals the
compiled power and the truth, with float32 drift $6\times10^{-4}$.

**A3 — Exact decoding, with a noise bound.** Decode each residue by
$\hat r_c = \arg\max_{r} \sum_j \mathrm{Re}\big(\overline{e^{2\pi i q_{c,j} r/p_c}}\, z_{c,j}\big)$,
then recover $n \in [-M/2, M/2)$ by the Chinese Remainder Theorem. For
$z = \varphi(n) + \varepsilon$ with $\varepsilon_j \sim \mathcal{CN}(0, \sigma^2)$ i.i.d.:
$$\Pr[\text{decode error}] \;\le\; \sum_{c=1}^{r} \sum_{\Delta=1}^{p_c-1} Q\!\left(\frac{\sqrt{R\,(1-\kappa_{c,\Delta})}}{\sigma}\right),
\qquad \kappa_{c,\Delta} = \tfrac1R \sum_j \cos\!\big(2\pi q_{c,j}\Delta/p_c\big).$$
*Proof.*
1. Take the true residue $r^*$ and any $r = r^* - \Delta$.
2. The score gap is $s_{r^*} - s_r = R(1-\kappa_{c,\Delta}) + \xi$.
3. Here $\xi = \sum_j \mathrm{Re}(c_j \varepsilon_j)$ with $|c_j|^2 = 2 - 2\cos(2\pi q_{c,j}\Delta/p_c)$.
4. So $\xi$ is Gaussian with variance $\sigma^2 R (1-\kappa_{c,\Delta})$.
5. An error at $(c, \Delta)$ requires $\xi < -R(1-\kappa)$.
6. Apply the Gaussian tail bound, then a union bound over all $(c, \Delta)$. ∎

The constants $\kappa$ are computed once from the codebook, as Γ was in v0.2.
*Measured* (this codebook, $\kappa \in [-0.53, +0.28]$):

| σ | empirical error | bound |
|---|---|---|
| 0.8 | 0.0000 | 0.0001 |
| 1.2 | 0.0312 | 0.0378 |
| 1.6 | 0.2752 | 0.4806 |

**A4 — Cleanup makes long executions exact.** Let $\Pi = \varphi \circ \mathrm{dec}$ (decode,
then re-encode). Whenever A3 decodes correctly, $\Pi(z) = \varphi(n)$ *exactly*. A cheaper
per-component snap to the nearest $p_c$-th root of unity is exact whenever every
$|\varepsilon_j| < \sin(\pi/p_c)$:
- the angle error is at most $\arcsin|\varepsilon_j|$;
- valid angles are spaced $2\pi/p_c$ apart, because $q$ is invertible mod $p_c$.

For $p_{\max} = 23$ that tolerance is $0.136$, against float32 step error $\sim 10^{-7}$.
Hence an execution of *any* length, with cleanup after each primitive, is exact. ∎

**Other types in the same algebra.**

| Type | Encoding | Used for |
|---|---|---|
| `Cyc_k` | $\exp(2\pi i q n / k)$ | weekdays (7), months (12), letters (26), colors (10). "Saturday + 3 days = Tuesday" is A1 on $\mathbb Z/7$ |
| `Bool` | $\pm 1$ | truth values |
| `Sym` | random phasor codebook; distinct atoms have $s = O(1/\sqrt D)$ | words, names, color names |
| `Pos` | `Int × Int` | grid positions; rotate, flip, transpose, translate are integer arithmetic on the coordinates, so exact by A1 |
| `Seq[τ]`, `Grid` | explicit token lists (the existing token field) | no bundle crosstalk, since length is explicit |
| `Record` | role–filler bundle $\sum_i \rho_i \circledast v_i$ | read by unbinding, then cleanup |

**Magnitude.** Residues cannot compare sizes, so "less than" is computed by decode (A3) plus
an integer comparison in the executor. An approximate channel
$\mu(n) = e^{i\omega n}$ with $\omega M < \pi$ supports estimation ("about how many").
Finding D2 shows HRM-EQB learned exactly such a channel by pattern. That is useful, but it is
a *sense of size*, not arithmetic.

### §2 Concept language and knowledge $\mathcal K$

**Terms.**
$$P ::= x \mid \mathrm{lit}(v) \mid c(P_1..P_k) \mid \mathrm{iterate}(P_n, \lambda z.P, P_0) \mid \mathrm{if}(P_b, P, P) \mid \mathrm{let}\ x = P\ \mathrm{in}\ P \mid \mathrm{fold}(\lambda(z,x).P, P_0, P_s) \mid \mathrm{choose}(P_S)$$
**Types:** `Int`, `Cyc_k`, `Bool`, `Sym`, `Seq[τ]`, `Grid`, and functions.
**Primitives** (rank 0): `zero`, `succ` $(\circledast\varphi(1))$, `neg` (conjugate), `bind`,
`unbind`, `eq`, `less` (decode then compare), `iterate`, `fold`, `if`, `choose`, and the
position operations.

**Library.** Each entry is
$\mathcal K = \{\, c \mapsto (k_c \in \mathcal U,\ \tau_c,\ B_c,\ \rho(c),\ f_c^{?},\ \mathrm{stats}) \,\}$:
- $k_c$ is the name key, the target of the Reader's concept head (§3);
- $\tau_c$ is the type, $B_c$ the body, $\rho(c)$ the rank;
- $f_c^{?}$ is an optional compiled skill (§5).

The **rank** is $\rho(c) = 1 + \max\{\rho(c') : c' \in B_c\}$. A body may reference only
strictly lower ranks. Iteration happens only through `iterate` and `fold`, whose counters
decrease.

```
add(a, b)    := iterate(b, succ, a)                          ρ=1   compiled: a ⊛ b        (A1)
mul(a, b)    := iterate(a, λz. add(z, b), zero)              ρ=2   compiled: φ(b)^∘a      (A2)
pow(a, b)    := iterate(b, λz. mul(z, a), succ(zero))        ρ=3   compiled: square-and-multiply
div(a, b)    := count of subtracting b from a while ¬less(a, b)   ρ=3
next_day(d)  := succ on Cyc_7                                 ρ=1
number(ws)   := fold(λ(z, w). add(mul(z, base(w)), value(w)), zero, ws)   "thirty seven" = 3·10 + 7
```

Concepts enter the library in two ways:
- **Definitions in English**, parsed by the Reader. "To triple a number, add it to itself
  three times" becomes `triple(x) := iterate(3, λz. add(z, x), zero)`.
- **Induction from examples** (§7).

**T1 — Termination.** Every closed, well-typed term over a well-founded library terminates
within $W(P)$ steps, where $W$ is computable from the term and its decoded counters (each
counter is $< M$).
*Proof.* Lexicographic induction on (rank, counter). Unfolding strictly lowers the maximal
rank. `iterate` and `fold` strictly lower their counter or remaining length. Primitives take
one step. ∎

### §3 Reader $R_\theta$: the transformer, pattern **in**

Words $w_{1..n}$ (English, word-level, $|V_E| \approx 1{,}024$) or a task token field
(grids and sequences, with the kept T1 rotation binding):
$$H = \mathrm{Enc}_\theta(E w + \mathrm{pos}) \in \mathbb R^{n\times d}.$$
Programs are decoded autoregressively, each step taking a softmax over **three heads**:
$$q_\theta(P \mid x) = \prod_t \mathrm{softmax}\Big(\big[\ W_s g_t\ ;\ \langle g_t, K_{\mathcal K}\rangle/\sqrt d\ ;\ \langle g_t, H_i\rangle/\sqrt d\ \big]\Big),\qquad g_t = \mathrm{Dec}_\theta(P_{<t}, H).$$
- **Syntax head.** Structure tokens: `(`, `iterate`, `λ`, …
- **Concept head.** Scores library keys $K_{\mathcal K}$. The output vocabulary *is* the
  library, so a newly defined concept can be selected with no retraining of an output layer.
- **Pointer head.** Copies input spans such as numerals and names.

**Literals are compositional.** A numeral or number word is parsed into the `number`
concept: "thirty seven" becomes `add(mul(3,10),7)`, and digits are folded by Horner's rule.
So the Reader needs only about 30 base words (zero … nineteen, the tens, hundred, thousand).
Their meanings are library entries grounded by counting: $\ell(w_{k+1}) = \mathrm{succ}(\ell(w_k))$.
Unseen large numbers are compositions of seen words, which is what makes U1 reachable.

**Why a transformer here.** Surface form is conventional and many-to-one: "times",
"multiplied by", "lots of" and "groups of" all mean `mul`. Attention learns this from modest
data. This is where pattern learning belongs.

### §4 Thinker $\mathcal E$: H-level executor (slow, exact, depth = the problem's)

A configuration is $C = (t, \eta, \kappa)$: focus term, environment, continuation stack.
Small-step semantics $C_{k+1} = \mathrm{step}(C_k)$, call-by-value, left to right:

| redex | step |
|---|---|
| primitive $o(v)$ | $\Pi(\hat o(v))$ — algebra operation then cleanup (A4) |
| $c(v)$, compiled $f_c$ trusted (K1) | $f_c(v)$ — an **L-level call** |
| $c(v)$ otherwise | $B_c[v/x]$ — **unfold**: "think about what $c$ means" |
| $\mathrm{iterate}(n, f, z)$ | decode $n$ (A3); $n = 0 \Rightarrow z$, else $\mathrm{iterate}(n{-}1, f, f(z))$ |
| $\mathrm{choose}(S)$ | push a choice point ordered by $\pi_\theta(\cdot \mid C)$; backtrack on Judge failure (§7) |

The trace $\tau = (C_0, \dots, C_T)$ is recorded at concept granularity (unfold and return
events).

**Halting.** The executor halts when $t$ is a value. T1 guarantees this, so no learned halting
head is needed for deterministic programs. Search halts on Judge acceptance or on budget, and
then asks.

**E — Exactness.** If every step is a primitive with cleanup, or a compiled skill trusted
under K1, then $\mathcal E(P) = [\![P]\!]$ for *every* input in range.
*Proof.* Induction on the number of steps, using A1, A2, A4 and T1. ∎
Consequence: U1 holds exactly wherever the Reader produced the right $P$.

**Depth.** $T$ is the problem's real cost. `mul(a,b)` by definition takes $\approx a$ unfold
steps, and $O(1)$ once compiled. Contrast D3: HRM-EQB takes 5 cycles for everything.

### §5 L-level: compiled skills and cleanup (fast, automatic)

**Algebraic compiles** are exact by theorem:
- `add` becomes binding (A1);
- `mul` becomes a power (A2);
- `pow` becomes square-and-multiply over `mul` ($O(\log b)$ steps).

**Learned compiles** cover concepts with no closed form. A small MLP $f_c(\cdot; \psi_c)$ is
trained on dreamed inputs to match $\mathcal E(B_c)$.

**K1 — Compilation gate** (the kept G1 keep gate). $f_c$ replaces unfolding only after it
agreed with the definition on $n$ independent random inputs, with zero disagreements. Then
$$\Pr[\text{true disagreement rate} > \epsilon] \le (1-\epsilon)^n$$
(460 checks give $\epsilon = 1\%$ at $\delta \approx 0.01$). The Judge keeps spot-checking at
rate $\rho$ via the slow path, so drift is caught.

**Cleanup attractors.**
- `Int` and `Cyc`: $\Pi = \varphi \circ \mathrm{dec}$ (A4).
- `Sym`: one step of softmax retrieval
  $z \mapsto \sum_a \mathrm{softmax}_a(\beta\, s(z,a))\, a$ over the codebook. For
  well-separated atoms, retrieval error decays exponentially in $\beta \times$ separation
  (Ramsauer et al. 2020, modern Hopfield retrieval theorem).

This is where v0.2's contraction mathematics belongs: **local** contraction into each stored
pattern's basin, not **global** contraction into one fixed point (D4).

### §6 Judge $\mathcal V$ and metacognition

Accept $y$ iff:
- **J1:** the execution reached a value within budget;
- **J2:** the result is type-correct;
- **J3:** an independent check agrees. Examples: an inverse (verify `mul` by `div`), compiled
  versus unfolded path, or exact replay on every demo (the kept `hrm/verify`).

**Reading confidence.** Over the Reader's beam,
$$c(y \mid x) = \sum_{P:\ \mathcal E(P) = y} q_\theta(P \mid x).$$
If $\max_y c < \tau$, the Judge asks which reading was meant. The kept Q1 theorem picks the
*most divided* alternatives. Example: "did you mean 2×(3+4) or (2×3)+4?"

**Unknown concepts** produce a question ("What does *thrice* mean?"). The answer is a
definition, the concept enters $\mathcal K$, and later queries use it: this is U2 in action.

### §7 Induction and search (few-shot grids, sequences, ARC; Sudoku)

Given demos $D = \{(x_i, y_i)\}$, the posterior over programs is
$$p(P \mid D) \propto q_\theta(P \mid D) \prod_i \mathbf 1\big[\mathcal E(P, x_i) = y_i\big].$$
**Search.** Best-first on $\log q_\theta$, each candidate verified exactly. Stop at $K$
consistent programs or at budget $B$. The answer is the prior-weighted majority; if the
consistent programs disagree, the Judge asks via Q1.

**Cost.** Guided search costs about depth × beam, versus $|\mathcal K|^{\text{depth}}$ for
enumeration. This removes ADR-004's ceiling: enumerative synthesis was capped at depth 2.

**Grid concepts** are exact algebra on `Pos` and colors:
- `translate(dx,dy)`: $\mathrm{pos} + (dx,dy)$;
- `rotate`: $(r,c) \mapsto (c, H{-}1{-}r)$;
- `flip`, `transpose`: the analogous coordinate maps;
- `recolor`: a map on `Sym`;
- `color_cycle`: `succ` on `Cyc_10`;
- `gravity`: a `fold` with `less`.

These are the eight families where HRM-EQB scored 0.000. Here each is a rank-1 or rank-2
concept, so solving them reduces to the search finding them.

**Sudoku.** `valid(g) := ∀ unit u: alldiff(u)`. `solve := choose`-driven search with
candidate elimination by `valid`. Exhaustive `choose` plus T1 give correctness on every
solvable puzzle. The policy $\pi_\theta$ learns cell and value order, which saves steps but
never affects correctness.

**ARC.** Success depends on library coverage and $\pi_\theta$. It is measured, not promised.

### §8 Writer $W_\theta$: the transformer, pattern **out**

$$p_\theta(u_{1..m} \mid \tau, y) = \prod_t \mathrm{softmax}(W_o\, \mathrm{Dec}_\theta(u_{<t},\ \mathrm{emb}(\tau))),$$
The decoder is shared with the Reader (tokens tagged as program or English). Values are
rendered by running `number` in reverse: decode (A3), then words.

**F1 — Faithfulness.** Every arithmetic claim in an explanation is copied (pointer head) from a
trace event, so the explanation replays to $y$. This is checked by closing the loop:
$\mathcal E(R_\theta(W_\theta(\tau))) = y$. The measured rate is the U3 metric; rendering
errors are the only failure mode.

**Worked example.**
```
x  = "What is two times three?"
P  = mul(number[two], number[three])              ← Reader: "times" → mul; literals compositional
τ  = mul(2,3) ⇒ iterate(2, λz.add(z,3), 0)        ← Thinker unfolds the definition
     ⇒ add(0,3) = 3 ⇒ add(3,3) = 6                  (each add = one binding + cleanup)
𝒱  = div(6,3) = 2 ✓                                ← Judge: independent inverse check
y  = "Two times three is six, because two times three means adding three two times:
      three plus three is six."                    ← Writer: serializes τ
```
Ask the same model "What is 4,812 times 37?" and nothing changes except $T$. Or it takes one
compiled step once K1 has trusted `mul`. No example near 4,812 was ever needed.

### §9 Learning: one objective, and what each term trains

Learned parameters: $\theta$ (Reader, Writer, search policy $\pi_\theta$) and $\psi$
(learned compiles). The knowledge $\mathcal K$ is data.

**Wake** (question $x$, answer $y^*$; the program is latent):
$$\mathcal L = \underbrace{-\log \!\!\sum_{P \in \mathrm{Beam}(x)}\!\! q_\theta(P\mid x)\,\mathbf 1[\mathcal E(P) = y^*]}_{\text{read: programs that execute to the answer (MML)}}
\;\underbrace{-\,\log p_\theta(u^* \mid \tau, y)}_{\text{write}}
\;\underbrace{-\sum \log \pi_\theta(\text{choices on the successful path})}_{\text{search}}
\;+\;\lambda \underbrace{\sum_c \mathbb E_z \|f_c(z;\psi) - \mathcal E(B_c)(z)\|^2}_{\text{compile}}$$
When a gold program is known (dreamed data), the read term becomes $-\log q_\theta(P^* \mid x)$.

**No term trains the algebra.** By construction, arithmetic cannot be memorized as pattern.

**Sleep** (reuses the `hrm/learning/sleep.py` gate):
- **S1 Dream.** Sample programs from $\mathcal K$, execute them, and render them with a
  template grammar and, later, the Writer. This yields (text, program, answer) triples that
  are correct by construction. It is how fewer than 1M parameters learn language-to-program
  without a large corpus.
- **S2 Abstract.** Add the concept $c^*$ that minimizes
  $|\mathcal K| + \sum_{\text{tasks}} |P_{\text{task}} \mid \mathcal K|$ (DreamCoder-style
  compression). Accept iff $\Delta\mathrm{MDL} < 0$. This is how `double` or `square` get
  *discovered* as named compositions.
- **S3 Compile.** Apply K1 to frequently executed concepts.

**M1 — Sleep converges.** Each accepted abstraction strictly lowers total description length.
Description length is a non-negative integer. So for a fixed task set, only finitely many
abstractions are accepted. ∎

### §10 Parameter budget: under 1M, because facts are not weights

| block | size | params |
|---|---|---|
| shared word embedding, tied output | 1,024 × 128 | 131,072 |
| Reader encoder | 2 layers × (attn 4d² + ff 2·d·2d) | 262,144 |
| shared decoder (program + English) | 2 layers × (self + cross + ff) | 393,216 |
| heads: syntax, concept-key projection, search policy | | ≈ 40,000 |
| **total learned** | | **≈ 0.83M** |
| algebra $\mathcal A$ | fixed codebooks | **0** |
| knowledge $\mathcal K$ | ≈ 1 KB per concept; data, not parameters | — |

Compare: HRM-EQB has 1.87M parameters, only 7 % of them in the reasoning core, and the
correctness of its arithmetic depends on all of them. Here no parameter carries arithmetic.

---

## Part IV — Theorem → test map (all capability claims)

| ID | Claim | Test (planned file `tests/test_x_*.py`) |
|---|---|---|
| A1 | addition = binding, for all $a, b$ | random pairs up to $\pm M/2$, float64 and float32 |
| A2 | repeated addition = multiplication = compiled power | loop vs power vs truth, $a$ up to $10^4$ |
| A3 | decoding exact; noise bound holds | exact decode on random ints; empirical error ≤ bound at σ ∈ {0.8, 1.2, 1.6} (verified 2026-09-27) |
| A4 | cleanup exact within radius; long runs exact | 20,000-step executions, bit-exact after cleanup |
| T1 | termination; $W(P)$ bound respected | random well-typed programs; recursion at equal rank rejected at insertion |
| E | executor = denotation | executor vs a reference Python interpreter on random programs |
| K1 | compile gate false-trust rate ≤ δ | seeded wrong compiles are rejected at the stated rate |
| F1 | explanations replay | $\mathcal E(R(W(\tau))) = y$ rate (measured) |
| M1 | abstraction lowers MDL | before/after description lengths on a fixed corpus |

**Not proved (and not claimed).** Reader accuracy, search success within budget on ARC, and
English coverage are learned or empirical. They are measured on Part V.

---

## Part V — English Understanding Suite v1 (new frozen benchmark; v1/v2 untouched)

| Suite | What | Criterion |
|---|---|---|
| E1 | arithmetic in words; train operands ≤ 20, test ≤ 10,000 | U1: report in-range vs out-of-range gap |
| E2 | concept by one English definition, 0 examples ("a dozen is twelve"; "to triple…") | U2 |
| E3 | "why is seven times eight fifty-six?" | U3: replay rate |
| E4 | days and months arithmetic (same algebra, new type) | transfer |
| E5 | instructed text operations ("reverse the word planet") | conventions via Reader |
| E6 | comparison and word problems ("3 bags with 4 apples each") | composition |
| E7 | undefined words, division by zero | U4: question or "undefined", never a guess |

**Ablations** (required by CLAUDE.md), which make each part load-bearing *by measurement*:
- **(a) Pattern baseline.** A transformer-only seq2seq with the same parameter budget.
- **(b) Learned embeddings instead of the algebra** (HRM-EQB's atomic integers). Prediction:
  E1 out-of-range collapses and E2 fails.
- **(c) No compiled skills** (always unfold). Prediction: same accuracy, more steps; this
  measures K1's value.
- **(d) No Judge.** Prediction: confident wrong answers where there should be questions.

---

## Part VI — Reconciliation with what exists

| Keep | Why |
|---|---|
| T1 rotation binding, token field (`hrm_eqb/field.py`, `tasks.py`) | grid and sequence perception for §3/§7 |
| `hrm/verify` exact replay | Judge J3 |
| `hrm/learning/sleep.py`, G1 keep gate | sleep S1–S3, K1 |
| Q1 question selection, V1/F3 verified atomic writes | Judge, library writes |
| `hrm_eqb/device.py`, bench harness, file-hash integrity | infrastructure |
| `hrm/substrate` FHRR | becomes the carrier of §1: now load-bearing |

| Retire | Evidence |
|---|---|
| quaternion product-key memory (291k params) | contributes exactly 0 (D6) |
| hormones, AttnRes chain | no ablation, no capability claim |
| coda (653k params) | replaced by the shared decoder |
| global contraction as the *reasoning* guarantee | forbids search and variable depth (D3, D4); its mathematics moves to local cleanup (§5) |
| `hrm/` enumerative synthesis | replaced by $q_\theta$-guided search (§7); its DSL becomes the primitive layer |

`hrm_eqb/` stays runnable as the measured baseline, as `hrm/` did after ADR-005.

---

## Part VII — Phased build (each phase ends: suite green, graph updated, vault note)

| Phase | Deliverable | Exit test |
|---|---|---|
| X0 | algebra $\mathcal A$ with its types | A1–A4 tests (no learning; ~200 lines) |
| X1 | concept language, executor, traces, rank checker, starter library (arithmetic, cyclic, text) | T1, E; `ask` on programs gives exact answers and traces |
| X2 | freeze English Suite v1; train baselines (a) and (b) **first** | the numbers to beat, recorded in an ADR |
| X3 | Reader and Writer (< 1M params) trained by dreaming plus MML | U1 gap, U3 replay rate |
| X4 | definitions in English; Judge questions | U2, U4 |
| X5 | guided induction over grid, sequence and text families; then Sudoku; ARC measured | per-family accuracy vs HRM-EQB's 0.000 |
| X6 | sleep: compile (K1) and abstraction (M1) | fewer steps with accuracy unchanged; MDL decreases |

**Risks, stated plainly.**
- The Reader is the only learned link on the answer path. If parsing is wrong, the answer is
  wrong, but the Judge (J3) and the ask rule (§6) turn many such errors into questions rather
  than false answers.
- Grammar coverage of English is the real scope limit. That is why the suite is templated
  English first.
- ARC-level induction may stay low even with guided search. It will be reported as measured.

## References

- Plate (2003) *Holographic Reduced Representations*; Frady, Kleyko et al. (2021–22) fractional
  power encoding and residue hyperdimensional computing.
- Ellis et al. (2021) *DreamCoder*: wake–sleep library learning by compression.
- Liang et al. (2017) *Neural Symbolic Machines*; Guu et al. (2017) maximum marginal
  likelihood over latent programs.
- Ramsauer et al. (2020) *Hopfield Networks is All You Need* (retrieval as local attraction).
- Graves (2014) *Neural Turing Machines*; Kaiser & Sutskever (2016) *Neural GPUs*
  (algorithmic depth that grows with the input).
