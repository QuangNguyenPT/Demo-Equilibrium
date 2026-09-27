---
title: "HRM-EQB — Hierarchical Reasoning by Equilibrium: Specification v0.2 (Theorems and Proofs)"
type: design
tags:
  - design
  - hrm-eqb
  - equilibrium
  - proofs
  - quaternion
  - neuromodulation
  - curiosity
status: proposed
date: 2026-09-27
supersedes: "[[HRM-Phi Design Document v0.1]]"
related:
  - "[[ADR-005 Neural HRM-Phi Replaces Program Search]]"
  - "[[Concept - Attention Residuals (RealFormer & AttnRes)]]"
---

# HRM-EQB — Hierarchical Reasoning by Equilibrium
## Specification v0.2 — every claim a theorem, every theorem a test

v0.1 (HRM-Φ) conceded that convergence on the phasor torus could not be proved. v0.2
closes that gap by design: the reasoning dynamics are contractive **by construction**, so
the model's behaviour follows from four numbers stored in its file. The spec adds:
- 4-D quaternion knowledge;
- rotation-vector text;
- Attention Residuals over the thinking chain;
- a hormone system (Doya's neuromodulators);
- curiosity that asks the most informative question;
- an autoresearch-style self-improvement gate with a provable false-keep bound.

Each theorem has a proof here and a numerical check in `tests/test_eqb_theorems.py` (§10).

**What is proved** is the model's *dynamics*: existence, uniqueness, convergence, halting,
gradients, robustness, retrieval exactness, and decision rules. **What is not proved** is
learning or generalization (§11).

---

## 0. The model in one page

The **task field** is $n$ tokens (grid cells, sequence elements, bytes of text, all
demonstrations plus the query and blank answer slots). Each token carries a state in
$\mathbb{R}^{4k}$. One state space has two algebras on the same bytes:
- the **complex view**, $2k$ planes $\mathbb{C}^{2k}$: rotation vectors for text and
  positions;
- the **quaternion view**, $k$ quaternions $\mathbb{H}^{k}$: rotations in 4-D for
  knowledge and understanding.

This works because $\mathbb{H}=\mathbb{C}\oplus\mathbb{C}j$ (Cayley–Dickson), so each
quaternion *is* two complex planes.

**Levels (three timescales).**

| Level | Operator | Guarantee |
|---|---|---|
| **L** (fast, general computation) | $T_L(l;h)=(1-\alpha)l+\alpha\,\sigma\big(A_s(lW_l^{\top}+hW_h^{\top})+c_L\big)$ | contraction in $l$ (E1) |
| **H** (slow, solution; thinks in the language space) | $T_H(h;l)=(1-\beta)h+\beta\,\sigma\big(A'_s(hV_h^{\top}+lV_l^{\top})+c_H\big)$ | joint equilibrium unique and reached for any $n_L$ (E2, E3) |
| **Segment** (refinement) | context $\theta_s=(A_s,A'_s,c_L,c_H)$ from the prelude, the knowledge memory read, AttnRes over past equilibria, and hormones | each segment converges regardless of context (E5); deep supervision across segments |

**Ingredients.**
- $A_s, A'_s$: sub-stochastic token-mixing matrices (routing, computed from the input and
  the previous segment).
- $W, V$: certified linear maps (orthogonal × bounded diagonal × orthogonal).
- $\sigma$: the radial ReLU on each quaternion.
- $c$: input and memory injections, which are unconstrained.

**Outputs.**
- A structured head (cleanup against the output codebook) at answer tokens.
- A language coda: a small transformer reading AttnRes-pooled thought tokens.
- A **Question** when curiosity detects a missing pattern (§7).

**Files.**
- `model.hrmc`: the whole trained model (all weights, architecture, budgets,
  self-test), pickle-free.
- `knowledge.pt`: the 4-D product-key memory, SSD-resident, written in place.

---

## 1. Notation and standing conventions

- $\|\cdot\|$ is the Euclidean (Frobenius) norm on $\mathbb{R}^{n\times4k}$. States are
  $h,l\in\mathcal{X}=\mathbb{R}^{n\times 4k}$.
- A linear map $X\mapsto AXW^{\top}$ (tokens mixed on the left, features on the right) has
  operator norm $\|A\|_2\|W\|_2$, because it is the Kronecker product $W\otimes A$ and
  $\|W\otimes A\|_2=\|W\|_2\|A\|_2$.
- $\mathrm{Lip}_x(F)$ is the Lipschitz constant of $F$ in argument $x$, uniformly in the
  other arguments.
- For a map $C$ on $\mathcal{X}\times\mathcal{X}$ and weights $w=(w_h,w_l)>0$, the
  **weighted block norm** is $\|(\Delta h,\Delta l)\|_w=w_h\|\Delta h\|+w_l\|\Delta l\|$.

---

## 2. Algebra

### T1 — RoPE is rotation-vector binding

**Statement.** For $q\in\mathbb{C}^{m}$, position $p\in\mathbb{R}$ and frequencies
$\omega\in\mathbb{R}^m$, RoPE equals binding with the phasor $P^{p}$,
$P_j=e^{i\omega_j}$:

$$\mathrm{RoPE}(q,p)=q\circledast P^{p},\qquad
\mathrm{Re}\langle q\circledast P^{p},k\circledast P^{p'}\rangle=\mathrm{Re}\langle q\circledast P^{p-p'},k\rangle .$$

**Proof.** RoPE rotates the real pair $(q_{2j},q_{2j+1})$ by angle $p\omega_j$. Rotating
the pair is multiplying the complex number $q_j$ by $e^{ip\omega_j}=(P^{p})_j$. For the
inner product,
$\mathrm{Re}\sum_j q_je^{ip\omega_j}\overline{k_je^{ip'\omega_j}}=\mathrm{Re}\sum_j q_j\bar k_je^{i(p-p')\omega_j}$. ∎

For multi-axis coordinates, $\mathrm{RoPE}_a(q,p)=q\circledast X^{p_x}\circledast Y^{p_y}\circledast\cdots$
(the bindings commute). Text is stored as rotation vectors in exactly this sense:
- a token is a unit phasor $e_c\in U(1)^{2k}$;
- its position is a binding;
- relative order is unbinding.

### T2 — On the torus, dot-product attention is L2 attention

**Statement.** If $q,k_c\in U(1)^m$, then $\|q-k_c\|^2=2m-2\,\mathrm{Re}\langle q,k_c\rangle$.
Hence
$\mathrm{softmax}_c(\beta\,\mathrm{Re}\langle q,k_c\rangle)=\mathrm{softmax}_c(-\tfrac\beta2\|q-k_c\|^2)$,
the logits lie in $[-\beta m,\beta m]$, and the attention weights are
$\tfrac\beta2\sqrt{Cm}$-Lipschitz in $q$ for $C$ keys.

**Proof.**
1. $\|q\|^2=\|k_c\|^2=m$. Expanding
   $\|q-k_c\|^2=\|q\|^2+\|k_c\|^2-2\,\mathrm{Re}\langle q,k_c\rangle$ gives the identity.
2. Softmax is invariant to adding a constant ($-\beta m$) shared by all $c$.
3. $|\mathrm{Re}\langle q,k\rangle|\le\|q\|\|k\|=m$.
4. The weights are $\mathrm{softmax}(\beta Kq)$ with $K$ stacking the keys as rows. The
   Jacobian is $(\mathrm{diag}(p)-pp^{\top})\beta K$, and $\|K\|_2\le\|K\|_F=\sqrt{Cm}$.
5. $\|\mathrm{diag}(p)-pp^{\top}\|_2\le\tfrac12$. For a unit vector $u$,
   $u^{\top}(\mathrm{diag}(p)-pp^{\top})u=\mathrm{Var}_{I\sim p}(u_I)$. The values $u_i$
   span at most $\max_{a,b}|u_a-u_b|\le\sqrt2$, so Popoviciu's inequality gives
   $\mathrm{Var}\le(\sqrt2)^2/4=\tfrac12$.
6. The domain $\mathbb{R}^{2m}$ is convex, so the Jacobian bound is a global Lipschitz
   bound. ∎

The bound depends only on $\beta, m, C$, never on activation magnitude. By contrast,
unconstrained dot-product attention is not globally Lipschitz (Kim et al., 2021).

### T3 — Quaternion binding: invertible, isometric, non-commutative

**Statement.** On $\mathbb{H}$ with the Hamilton product:
1. $|pq|=|p||q|$;
2. for unit $r$, $x\mapsto x\otimes r$ is an isometry of $\mathbb{R}^4$ with inverse
   $x\mapsto x\otimes\bar r$;
3. $ij=k=-ji$, so binding is non-commutative;
4. $\mathbb{C}=\mathrm{span}\{1,i\}\subset\mathbb{H}$ is a commutative subalgebra, so
   complex phasor binding is the special case.

**Proof.**
1. $|pq|^2=pq\,\overline{pq}=p\,q\bar q\,\bar p=|q|^2p\bar p=|p|^2|q|^2$, using
   $\overline{pq}=\bar q\bar p$ and the realness of $|q|^2$.
2. By (1), $|x r-y r|=|x-y||r|=|x-y|$. Also $x r\bar r=x|r|^2=x$.
3. Direct from Hamilton's rules.
4. The span of $\{1,i\}$ is closed under multiplication, which reduces to complex
   multiplication. ∎

### T4 — Quaternions are the rotations of 3-D and 4-D space

**Statement.**
- **(a)** For unit $q$ and pure $v$, $v\mapsto qv\bar q$ is a rotation in $SO(3)$.
  Composing two such maps gives $q_2q_1$ (order matters), and $q$ and $-q$ give the same
  rotation.
- **(b)** For unit $p,q$, $x\mapsto px\bar q$ is in $SO(4)$, and every element of $SO(4)$
  has this form, with $(p,q)$ unique up to a joint sign.

**Proof.**

(a):
1. The map is linear and, by T3, norm-preserving.
2. It maps pure quaternions to pure ones: $\mathrm{Re}(qv\bar q)=\mathrm{Re}(v)=0$.
3. So it lies in $O(3)$. It depends continuously on $q\in S^3$ (which is connected) and
   is the identity at $q=1$, so its determinant is $+1$.
4. Composition: $q_2(q_1v\bar q_1)\bar q_2=(q_2q_1)v\overline{(q_2q_1)}$.
5. The sign: $(-q)v\overline{(-q)}=qv\bar q$.

(b):
1. The map $\Psi(p,q)(x)=px\bar q$ is linear and norm-preserving. It is continuous on the
   connected group $S^3\times S^3$ and equals the identity at $(1,1)$, so it lies in
   $SO(4)$.
2. $\Psi$ is a Lie-group homomorphism. Its kernel: if $px\bar q=x$ for all $x$, then
   $x=1$ gives $p=q$; then $px\bar p=x$ for all $x$ makes $p$ central, so $p=\pm1$. The
   kernel $\{\pm(1,1)\}$ is discrete.
3. Hence the image has dimension $\dim(S^3\times S^3)=6=\dim SO(4)$. A subgroup of full
   dimension contains an open neighbourhood of the identity. It is compact, hence closed,
   hence also open and closed in the connected group $SO(4)$, so it equals $SO(4)$. ∎

Consequences:
- The vector3d domain's rotations are native quaternion products.
- A general 4-D rotation costs two quaternions (6 degrees of freedom, as many as
  $SO(4)$ has).

### T5 — Exact spectral certificates

**Statement.** Let $W=Q_1\,\mathrm{diag}(s)\,Q_2$ with $Q_1,Q_2$ orthogonal. Then
$\|W\|_2=\max_i|s_i|$. The Cayley transform $Q=(I+S)^{-1}(I-S)$ of any skew-symmetric
$S$ is orthogonal. With $s_i=\varsigma\cdot\mathrm{sigmoid}(\theta_i)$,
$\|W\|_2<\varsigma$ **for every value of the parameters**.

**Proof.**
1. The spectral norm is invariant under orthogonal factors:
   $\|Q_1DQ_2\|_2=\|D\|_2=\max|s_i|$.
2. $S$ has purely imaginary eigenvalues, so $I+S$ is invertible.
3. $(I-S)$ and $(I+S)^{-1}$ commute, being polynomials and inverses of polynomials in
   $S$. Therefore
   $Q^{\top}Q=(I+S)(I-S)^{-1}(I+S)^{-1}(I-S)=(I+S)(I+S)^{-1}(I-S)^{-1}(I-S)=I$,
   using $S^{\top}=-S$.
4. The sigmoid lies in $(0,1)$. ∎

The Cayley map reaches every rotation in $SO(d)$ except the measure-zero set with
eigenvalue $-1$, so the factorization loses no expressivity in practice. In floating
point, the sigmoid saturates to $1$ for $\theta\gtrsim37$, giving $\|W\|_2\le\varsigma$
rather than $<$. The certificate only needs $\le$, because the budgets themselves satisfy
$\Gamma<1$ strictly.

### S1 — Sub-stochastic routing is non-expansive

**Statement.** If $A\ge0$ entrywise with every row sum $\le1$ and every column sum
$\le1$, then $\|A\|_2\le1$. Any nonnegative matrix is brought into this set by dividing by
$\max(1,\max\text{row sum},\max\text{column sum})$. No convergence of Sinkhorn is needed.

**Proof.**
$\|A\|_2^2=\lambda_{\max}(A^{\top}A)\le\|A^{\top}A\|_\infty\le\|A^{\top}\|_\infty\|A\|_\infty=\|A\|_1\|A\|_\infty\le1$.
This uses that the spectral radius is bounded by any induced norm. ∎

### N1 — The radial ReLU is 1-Lipschitz and rotation-equivariant

**Statement.** On each quaternion block $x\in\mathbb{R}^4$, let
$\sigma(x)=g(|x|)\,x/|x|$ with $g(t)=\max(0,t-\mu)$, where the shrink is $\mu\ge0$.
Then $\sigma$ is 1-Lipschitz,
and $\sigma(Rx)=R\sigma(x)$ for every orthogonal $R$.

**Proof.**
1. $\sigma$ is continuous (it is $0$ on $|x|\le\mu$) and piecewise $C^1$.
2. Where differentiable, its Jacobian is
   $g'(r)\,\hat x\hat x^{\top}+\tfrac{g(r)}{r}(I-\hat x\hat x^{\top})$, whose operator
   norm is $\max(g'(r),g(r)/r)\le1$, because $g'\in\{0,1\}$ and $g(r)\le r$.
3. A continuous, piecewise-$C^1$ map on a convex set is Lipschitz with constant equal
   to the supremum of its Jacobian norm: integrate along segments, which meet the
   non-smooth set $\{|x|=\mu\}$ in finitely many points (a line meets a sphere at
   most twice).
4. Blocks act independently, and squared norms add, so the whole map is 1-Lipschitz.
5. Equivariance: $|Rx|=|x|$ and $Rx/|Rx|=R(x/|x|)$. ∎

---

## 3. Equilibrium: the hierarchical reasoning theorem

**Assumptions (A).** For the current segment context, uniformly in it:

| | constant |
|---|---|
| $\mathrm{Lip}_l(T_L)$ | $\rho_L<1$ |
| $\mathrm{Lip}_h(T_L)$ | $\kappa_L$ |
| $\mathrm{Lip}_h(T_H)$ | $a$ |
| $\mathrm{Lip}_l(T_H)$ | $b$ |

### E1 — The fast level has a unique equilibrium that moves smoothly with the plan

**Statement.** For each $h$, $T_L(\cdot;h)$ has a unique fixed point $l^*(h)$, reached
at rate $\rho_L$, and $\mathrm{Lip}(l^*)\le\kappa_L/(1-\rho_L)$.

**Proof.**
1. Existence, uniqueness and rate follow from the Banach fixed-point theorem on the
   complete space $\mathcal{X}$.
2. For the Lipschitz bound:
   $\|l^*(h)-l^*(h')\|=\|T_L(l^*(h);h)-T_L(l^*(h');h')\|\le\rho_L\|l^*(h)-l^*(h')\|+\kappa_L\|h-h'\|$.
   Rearrange. ∎

### E2 — Nested equilibrium: existence and uniqueness

**Statement.** Let $\Gamma_A:=a+b\kappa_L/(1-\rho_L)$. If $\Gamma_A<1$, then
$G(h)=T_H(h,l^*(h))$ is a contraction with rate $\Gamma_A$. The pair
$(h^*,l^*(h^*))$ is the unique joint equilibrium: $h^*=T_H(h^*,l^*)$ and
$l^*=T_L(l^*;h^*)$.

**Proof.**
1. $\|G(h)-G(h')\|\le a\|h-h'\|+b\|l^*(h)-l^*(h')\|\le\Gamma_A\|h-h'\|$ by E1.
2. Banach gives existence and uniqueness of $h^*$.
3. Any joint equilibrium has $l=l^*(h)$ by E1's uniqueness, so $h$ is a fixed point of
   $G$. ∎

### E3 — The HRM schedule converges for every number of L-steps

**Statement.** One HRM cycle with $n\ge1$ L-steps is

$$C_n(h,l)=\big(T_H(h,\,l_n),\;l_n\big),\qquad l_n=T_L^{\,n}(l;h).$$

Let $K_n=\kappa_L(1-\rho_L^{\,n})/(1-\rho_L)$ and

$$M_n=\begin{pmatrix}a+bK_n & b\rho_L^{\,n}\\ K_n & \rho_L^{\,n}\end{pmatrix}.$$

If $\Gamma_A<1$, then:
- the Perron root satisfies
  $r_n=\tfrac12\big(\mathrm{tr}M_n+\sqrt{\mathrm{tr}^2M_n-4a\rho_L^n}\big)<1$;
- $C_n$ is an $r_n$-contraction in $\|\cdot\|_w$, with $w$ the left Perron vector of
  $M_n$;
- the fixed point of $C_n$ is the nested equilibrium of E2, for every $n$;
- $r_n\to\Gamma_A$ as $n\to\infty$.

**Proof.**

*Step 1 (bound matrix).*
- By induction, $\mathrm{Lip}_h(T_L^{n})\le\kappa_L\sum_{i<n}\rho_L^i=K_n$ and
  $\mathrm{Lip}_l(T_L^n)\le\rho_L^n$, so
  $\|\Delta l_n\|\le K_n\|\Delta h\|+\rho_L^n\|\Delta l\|$.
- Then
  $\|\Delta h'\|\le a\|\Delta h\|+b\|\Delta l_n\|\le(a+bK_n)\|\Delta h\|+b\rho_L^n\|\Delta l\|$.
- Componentwise, $(\|\Delta h'\|,\|\Delta l'\|)\le M_n(\|\Delta h\|,\|\Delta l\|)$.

*Step 2 (Perron root).*
- The characteristic polynomial is $\lambda^2-\mathrm{tr}M_n\,\lambda+\det M_n$ with
  $\det M_n=a\rho_L^n$.
- Evaluating at 1:
  $p(1)=(1-a)(1-\rho_L^n)-bK_n=(1-\rho_L^n)\big(1-a-\tfrac{b\kappa_L}{1-\rho_L}\big)=(1-\rho_L^n)(1-\Gamma_A)>0$.
- A nonnegative $2\times2$ matrix has real eigenvalues (the discriminant equals
  $(M_{11}-M_{22})^2+4M_{12}M_{21}\ge0$). Its larger root is $<1$ iff $p(1)>0$ and
  $\mathrm{tr}<2$.
- The trace condition holds: $\mathrm{tr}M_n<a+(1-a)(1-\rho_L^n)+\rho_L^n=1+a\rho_L^n<2$.

*Step 3 (weighted norm).*
- Assume $b,\kappa_L,\rho_L>0$ (the levels are coupled). The construction C1 guarantees
  this whenever $s_l,s_h,v_l>0$, which T5's strictly positive scales always give. Then
  every entry of $M_n$ is positive.
- By Perron–Frobenius there is $w>0$ with $w^{\top}M_n=r_nw^{\top}$. Normalize
  $w_h=1$. The first column of the eigen-equation gives
  $w_l=(r_n-a-bK_n)/K_n$, which is positive because the Perron root of a positive
  matrix exceeds each diagonal entry.
- Then
  $\|C_n(z)-C_n(z')\|_w\le w^{\top}M_n(\|\Delta h\|,\|\Delta l\|)^{\top}=r_n\|z-z'\|_w$.
- Since $w_h=1$, $\|h-h'\|\le\|z-z'\|_w$: any bound in this norm bounds the solution
  state $h$ directly.

*Step 4 (same fixed point).*
- A fixed point has $l=T_L^n(l;h)$. The map $T_L^n(\cdot;h)$ is a contraction whose
  unique fixed point is $l^*(h)$, so $l=l^*(h)$ and $h=G(h)$.
- As $n\to\infty$, $M_n\to\begin{pmatrix}\Gamma_A&0\\ \kappa_L/(1-\rho_L)&0\end{pmatrix}$,
  whose Perron root is $\Gamma_A$. ∎

**Meaning.** HRM's L/H schedule ("run L a few steps, then update H") is not a heuristic
under (A): it converges to one well-defined equilibrium no matter how many L-steps are
taken. $n_L$ changes the speed, never the answer. This is the precise statement of "the
hierarchy".

### C1 — The construction meets (A), with a certificate independent of the step sizes

**Statement.** Take $T_L$, $T_H$ as in §0, with:
- $\|A_s\|_2,\|A'_s\|_2\le1$ (S1);
- $\sigma$ 1-Lipschitz (N1);
- $\|W_l\|\le s_l<1$, $\|W_h\|\le s_h$, $\|V_h\|\le v_h$, $\|V_l\|\le v_l$ (T5).

Then:
- $\rho_L=1-\alpha+\alpha s_l$, $\kappa_L=\alpha s_h$, $a=1-\beta+\beta v_h$, $b=\beta v_l$;
- $\Gamma_A<1\iff\boxed{\;\Gamma:=v_h+\dfrac{v_l\,s_h}{1-s_l}<1\;}$, independent of
  $\alpha,\beta\in(0,1]$;
- $\Gamma_A=1-\beta(1-\Gamma)$.

**Proof.**
1. $\|T_L(l;h)-T_L(l';h)\|\le(1-\alpha)\|\Delta l\|+\alpha\|A_s(\Delta l)W_l^{\top}\|\le(1-\alpha+\alpha s_l)\|\Delta l\|$,
   by N1 and §1. The other three constants follow in the same way.
2. Substituting:
   $\Gamma_A=1-\beta+\beta v_h+\beta v_l\frac{\alpha s_h}{\alpha(1-s_l)}=1-\beta(1-\Gamma)$. ∎

**Remark (the constants are sharp).** Take every certified map equal to $sI$ (Cayley
factors at the identity, constant scales), identity routing, and no shrink ($\mu=0$ makes
$\sigma$ the identity). Then:
- $T_L(l;h)-T_L(l';h)=\rho_L(l-l')$ exactly;
- the cycle is linear with Jacobian $M_n\otimes I$, so its spectral radius is exactly
  $r_n$.

No smaller constants can be certified for this construction. The test
`test_c1_e3_constants_are_attained_by_the_aligned_linear_core` checks both equalities to
$10^{-10}$.

So **four budgets $(s_l,s_h,v_h,v_l)$ decide provable convergence**:
- $\Gamma$ is the price of provability (smaller means faster and safer, but less
  expressive);
- $\beta$ sets the speed.

### E4 — Exact gradients, and what HRM's 1-step gradient costs

**Statement.** Suppose $C_n$ is differentiable at the equilibrium $z^*$ with Jacobian
$J$. This holds almost everywhere: $\sigma$ has a measure-zero kink set. Then:
- $z^*(\theta)$ is differentiable with $\partial_\theta z^*=(I-J)^{-1}\partial_\theta C_n$;
- $(I-J)^{-1}=\sum_{i\ge0}J^i$;
- for every $v$,
  $\big\|(I-J)^{-1}v-\sum_{i<k}J^iv\big\|_w\le\frac{r_n^k}{1-r_n}\|v\|_w$.

HRM's 1-step gradient is $k=1$, with relative error at most $r_n/(1-r_n)$, which is
informative only when $r_n<\tfrac12$. Backpropagating through the last $k$ cycles (TRM,
Geiping) has error $r_n^k/(1-r_n)$.

**Proof.**
1. $\|Ju\|_w\le r_n\|u\|_w$, because $J$ is the derivative of an $r_n$-Lipschitz map.
2. The Neumann series therefore converges; its tail is
   $\|\sum_{i\ge k}J^iv\|_w\le\sum_{i\ge k}r_n^i\|v\|_w$.
3. The implicit function theorem applies to $z-C_n(z,\theta)=0$, because $I-J$ is
   invertible. ∎

Backpropagation uses vector–Jacobian products $v^{\top}(I-J)^{-1}$. The same bound holds
for them in the dual norm
$\|u\|_{w*}=\max(\|u_h\|/w_h,\|u_l\|/w_l)$, because $\|J^{\top}\|_{w*}=\|J\|_w\le r_n$.

### E5 — Every segment converges, whatever the input, memory or hormones

**Statement.**
- The constants of C1 do not depend on the context $\theta_s$ (routing, injections,
  memory read, hormonal state). So E2–E4 hold for **every** input and every segment.
- For injections $c=(c_L,c_H)$, measured as $\|\Delta c\|=\|\Delta c_L\|+\|\Delta c_H\|$:

$$\|z^*(c)-z^*(c')\|_w\le\frac{L_c}{1-r_n}\|c-c'\|,\qquad
L_c=\max\!\Big(w_h\beta,\;\alpha\tfrac{1-\rho_L^{\,n}}{1-\rho_L}\,(w_hb+w_l)\Big).$$

**Proof.**
1. The C1 constants are expressed only through $\alpha,\beta,s,v$.
2. $L_c$ bounds one cycle in $c$.
   - $c_L$ enters each $T_L$ step inside the 1-Lipschitz $\sigma$ with coefficient
     $\alpha$. Over $n$ steps the effects compound with factor $\rho_L$:
     $\|\Delta l_n\|\le\alpha\frac{1-\rho_L^n}{1-\rho_L}\|\Delta c_L\|$.
   - Then $\|\Delta h'\|\le\beta\|\Delta c_H\|+b\|\Delta l_n\|$ and
     $\|\Delta l'\|=\|\Delta l_n\|$.
   - Weighting by $(w_h,w_l)$ and taking the worse coefficient gives $L_c$.
3. For the sensitivity:
   $\|z^*(c)-z^*(c')\|_w\le\|C(z^*(c),c)-C(z^*(c),c')\|_w+\|C(z^*(c),c')-C(z^*(c'),c')\|_w\le L_c\|c-c'\|+r_n\|z^*(c)-z^*(c')\|_w$.
   Rearrange. ∎

A noisy memory read or input perturbation of size $\delta$ moves the thought by at most
$L_c\delta/(1-r_n)$. Knowledge cannot derail thinking out of proportion.
`ContractionBounds.injection_gain` computes $L_c$.

### E6 — Halting is a theorem

**Statement.** For iterates $z_{t+1}=C_n(z_t)$:
$\|z_{t+1}-z^*\|_w\le\frac{r_n}{1-r_n}\|z_{t+1}-z_t\|_w$. Hence:
- **Halting rule:** stopping at the first $t$ with
  $\|z_{t+1}-z_t\|_w\le\varepsilon(1-r_n)/r_n$ guarantees $\|z_{t+1}-z^*\|_w\le\varepsilon$.
- **Answer stability:** suppose the answer logits are $L_{\rm dec}$-Lipschitz from
  $\|\cdot\|_w$ into the max-norm, and the equilibrium's decision margin (top logit minus
  runner-up) is $\gamma$. Then $\varepsilon<\gamma/(2L_{\rm dec})$ implies that the halted
  answer **equals** the equilibrium answer.

**Proof.**
1. $\|z_{t+1}-z^*\|_w=\|C(z_t)-C(z^*)\|_w\le r_n\|z_t-z^*\|_w\le r_n(\|z_t-z_{t+1}\|_w+\|z_{t+1}-z^*\|_w)$.
   Rearrange.
2. For stability: each logit moves by less than $\gamma/2$, so the arg-max cannot
   change. ∎

PEM's kinetic halt $R<\varepsilon$ was a heuristic; here the residual *certifies*
distance to the answer.

---

## 4. Knowledge in 4-D space

### K1 — Relations as 4-D rotations

**Statement.** Knowledge triples $(e_1,R,e_2)$ are encoded as $e_2\approx e_1\otimes R$
per quaternion component, with $R$ unit. The score is $-\|e_1\otimes R-e_2\|$. Then:
1. **Inverse:** the inverse relation is exactly $\bar R$.
2. **Composition:** multi-hop is $e_1\otimes R_1\otimes R_2$, and unit quaternions are
   closed under products.
3. **Order:** there are $R_1,R_2$ with $R_1R_2\ne R_2R_1$, so "father's mother" differs
   from "mother's father".
4. **Commutative case:** complex-phasor relations (RotatE) are the commutative special
   case $R\in\mathbb{C}$.
5. **Invariance:** scores are unchanged when all entities are right-rotated by a unit $u$
   and relations are conjugated $R\mapsto\bar uRu$:
   $\|e_1R-e_2\|=\|(e_1u)(\bar uRu)-(e_2u)\|$. The frame of the knowledge space is
   arbitrary; only relative rotations carry meaning.

**Proof.**
- (1)–(4) follow from T3, T4 and group closure of $S^3$.
- (5): $(e_1u)(\bar uRu)=e_1Ru$, and
  $\|e_1Ru-e_2u\|=\|(e_1R-e_2)u\|=\|e_1R-e_2\|$. ∎

This is QuatE (Zhang et al., 2019) inside the reasoning state. It explains why knowledge
lives in 4-D: 2-D (complex) relations cannot express order-sensitive composition.

### M1 — Product-key retrieval is exact, in $O(\sqrt N)$

**Statement.** Scores are $s(a,b)=s_1(a)+s_2(b)$ over $N=S_1S_2$ slots. Let $I_1,I_2$
be top-$k$ index sets of $s_1,s_2$ ($k\le\min(S_1,S_2)$). Then the $k$ largest values of
$s$ over all $N$ pairs equal the $k$ largest values over $I_1\times I_2$.

**Proof.**
1. *Domination.* Take any pair $(a,b)$ and let
   $D=\{(a',b')\in I_1\times I_2: s_1(a')\ge s_1(a),\,s_2(b')\ge s_2(b)\}$. Every
   element of $D$ scores $\ge s(a,b)$.
   - If $a\in I_1$, then $a'=a$ qualifies; otherwise all $k$ elements of $I_1$ do.
     Likewise for $b$.
   - So $D\neq\emptyset$, and if $(a,b)\notin I_1\times I_2$ then $|D|\ge k$.
2. Let $v_1\ge\dots\ge v_k$ be the global top values and $u_1\ge\dots\ge u_k$ those over
   $I_1\times I_2$. Since $I_1\times I_2$ is a subset, $u_i\le v_i$.
3. Fix $i\le k$ and look at the pairs holding $v_1..v_i$.
   - If all lie in $I_1\times I_2$, they are $i$ inside pairs with score $\ge v_i$.
   - Otherwise one lies outside, and by step 1 it is dominated by $\ge k\ge i$ inside
     pairs with score $\ge v_i$.
   - Either way $u_i\ge v_i$. ∎

Memory read cost is $O(S d_k+k^2+k d_v)$. Only $k$ value rows are touched, so the values
can live on SSD.

---

## 5. Language: rotation-vector text, thinking in the language space, Attention Residuals

### L1 — A random rotation-vector codebook separates its symbols

**Statement.** For $|\mathcal{V}|$ tokens with i.i.d. uniform phasors in $U(1)^{m}$:

$$\Pr\Big[\max_{c\ne c'}\tfrac1m\mathrm{Re}\langle e_c,e_{c'}\rangle\ge t\Big]\le|\mathcal{V}|^2e^{-t^2m/2}.$$

For $m=256$, $|\mathcal{V}|=256$, $t=0.4$ the bound is below $10^{-4}$.

**Proof.**
1. $\mathrm{Re}\langle e_c,e_{c'}\rangle=\sum_j\cos(\phi_j)$, where
   $\phi_j=\theta_j-\theta'_j$ is uniform, the terms are independent, lie in $[-1,1]$,
   and have mean $0$.
2. Hoeffding's inequality gives
   $\Pr[\sum_j\cos\phi_j\ge tm]\le\exp(-2t^2m^2/(4m))$.
3. Take a union bound over fewer than $|\mathcal{V}|^2$ pairs. ∎

### L2 — The high level thinks in the language space

**Statement.** $H$'s answer tokens are decoded by cleanup against the text codebook.
- Suppose the codebook's pairwise similarity is $<t$ (L1). Then the decision margin at a
  state that equals a code vector is at least $\beta_o m(1-t)$.
- By E6, once the halting rule fires with $\varepsilon<\gamma/(2L_{\rm dec})$, every $H$
  token decodes to the same symbol as at the equilibrium.

So the slow level's thought is a sequence of rotation vectors *in the text space*, and
it reads out as words deterministically.

**Proof.** The margin claim follows from T2's identity: the correct code scores $m$ and
every other code scores below $tm$. The rest is E6. ∎

### A1 — Attention Residuals keep the thinking chain bounded

**Statement.** Let $v_0,\dots,v_{s-1}$ be past segment equilibria and
$u=\sum_i\alpha_iv_i$ with
$\alpha=\mathrm{softmax}(w^{\top}\mathrm{RMSNorm}(v_i))$ (arXiv:2603.15031). Then
$u\in\mathrm{conv}\{v_i\}$ and $\|u\|\le\max_i\|v_i\|$. With $w=0$, $u$ is the uniform
mean. The plain residual sum $\sum_iv_i$ only guarantees $\|\cdot\|\le s\max_i\|v_i\|$.

**Proof.**
1. The softmax weights are nonnegative and sum to 1.
2. The norm is convex.
3. $w=0$ makes every logit equal. ∎

**Use.** AttnRes over $(h^*_1,\dots,h^*_s)$:
- initializes segment $s+1$;
- with $k_\tau$ learned pseudo-queries, is the **semi-embedder** that hands the
  thinking chain to the coda.

A1 guarantees the chain never inflates, which is the paper's "no PreNorm dilution"
property, now a lemma.

---

## 6. The hormone system (neuromodulation)

Following Doya (2002), four global signals set the meta-parameters of learning and
action. Each lives in a box $[\ell,u]\subset(0,1)$:

| Hormone | Computational role (Doya) | Acts on (always outside the certified maps) |
|---|---|---|
| **DA** dopamine | reward-prediction error $\delta=r-q$ | consolidation priority of a memory row; sign of the halting-head update |
| **5-HT** serotonin | discount / patience $\gamma$ | segment budget: max segments $\lceil1/(1-\gamma)\rceil$; halting threshold |
| **NE** noradrenaline | inverse temperature (explore ↔ exploit) | routing temperature in $A_s$; restart noise $\sigma_0$ |
| **ACh** acetylcholine | learning rate (storage vs retrieval) | test-time memory write rate $\eta$; injection gain of memory vs input |

**Dynamics.** Per segment,
$m_{s+1}=\Pi_{\rm box}\big((1-\eta_m)m_s+\eta_m\,g(\text{signals}_s)\big)$, where the
signals are:
- $\delta_s$ (reward-prediction error);
- $\hat r_s$ (contraction speed);
- $E^{\rm loo}_s$ (leave-one-out error, §7);
- the vote entropy $U_s$ (§7).

### H1 — Hormones are stable and cannot break convergence

**Statement.**
- **(a)** The box is invariant, and for constant signals $m_s$ converges geometrically
  at rate $1-\eta_m$.
- **(b)** The equilibrium certificate is independent of $m$, so every segment converges
  (E5) in every hormonal state.

**Proof.**
- (a): A convex combination of points in a convex box stays in the box. The map
  $m\mapsto\Pi((1-\eta_m)m+\eta_m g)$ is a $(1-\eta_m)$-contraction, because projection
  onto a convex set is 1-Lipschitz.
- (b): The hormones enter only $A_s$ (through its temperature), the injection gains,
  the write rates, the restart noise and the halting thresholds. None of these appears
  in C1's constants. ∎

Hormones *steer* (how bold, how patient, how eager to learn), but they can never make
thought diverge.

---

## 7. Cognition: seeing a missing pattern and asking

A **committee** is the set of $K$ equilibria obtained under augmentation views
$g_1..g_K$ (dihedral and color actions, de-augmented) and annealed restarts. Each member
$h$ is a deterministic function $x\mapsto\hat y_h(x)$. Two signals detect a missing
pattern:

1. **Inconsistency.** $E^{\rm loo}>0$: some demonstration output is mispredicted when
   hidden. The demos are not jointly explained.
2. **Underdetermination.** $U(x)=H(\text{vote}_x)>0$: explanations that agree on the
   demos disagree on $x$.

### Q1 — The best question is the committee's most divided input

**Statement.** Take a uniform prior over the committee and deterministic members. The
expected information gain about which member is right, from observing the true output
at $x$, equals the entropy of the vote distribution at $x$:
$\mathrm{EIG}(x)=H\big(\hat y(x)\big)$. The optimal question is
$x^*=\arg\max_{x\in\mathcal{P}}H(\mathrm{vote}_x)$ over a probe set $\mathcal{P}$ (the
query input, its augmentations, and demonstration inputs with small perturbations).

**Proof.** $\mathrm{EIG}(x)=I(Y_x;h)=H(Y_x)-H(Y_x\mid h)$. Given $h$, $Y_x=\hat y_h(x)$
is determined, so $H(Y_x\mid h)=0$. Under the uniform prior, $\Pr[Y_x=y]$ is the fraction
of members voting $y$. ∎

**Protocol.** When loaded (`hrm_eqb.load(...)`), calling the model on a task returns
either an **Answer** (committee unanimous, $E^{\rm loo}=0$, halting certified) or a
**Question**:

> "What is the output for $x^*$?" (expected information gain $H$ bits)

Before asking a human, the model asks its own knowledge memory. If a retrieved row's
prediction splits the committee to a single survivor, no question is needed.

This is the "curiosity" of the design: detecting missing information, and choosing the
single most informative question. Across tasks, the same signal ranks regions of the
competence map for self-study (learning progress, Oudeyer & Kaplan).

---

## 8. Self-improvement, influenced by autoresearch

autoresearch runs an agent loop: a fixed budget per experiment, one frozen metric,
keep-or-discard, a ledger, and never edit the evaluator. HRM-EQB applies it to itself:
1. curiosity proposes a study region;
2. practice runs at a fixed compute budget;
3. sleep consolidates memory into the core;
4. the frozen, hash-locked harness judges;
5. the change is kept or discarded;
6. `results.tsv` records it.

The missing piece was *how much* improvement justifies a keep.

### G1 — The keep gate has a provable false-keep rate

**Statement.** Let $d_i\in[-1,1]$ be paired per-task differences (candidate − baseline)
on $n$ held-out tasks, i.i.d. Keep iff
$\bar d\ge\varepsilon_n=\sqrt{2\ln(1/\delta)/n}$. If the true improvement is
$\mathbb{E}d\le0$, then $\Pr[\text{keep}]\le\delta$. Over $T$ experiments, using
$\delta/T$ keeps the total false-keep rate $\le\delta$.

**Proof.** Hoeffding for variables with range 2 gives
$\Pr[\bar d-\mathbb{E}d\ge\varepsilon]\le\exp(-2n\varepsilon^2/4)=\exp(-n\varepsilon^2/2)=\delta$.
Bonferroni gives the multi-experiment statement. ∎

**Consequence (honest).**
- On benchmark v2 ($n=108$), $\delta=0.05$ needs $\bar d\ge0.235$: that benchmark
  cannot certify small gains.
- The gate must use large generated held-out sets. For example, $n=4000$ gives
  $\varepsilon=0.039$.

---

## 9. The model files

### F1 — `model.hrmc`: the whole model, provable from its bytes

**Contents.**
- Architecture.
- **All trained weights**: prelude, the Cayley parameters and scale logits of every
  certified map, routing heads, injections, structured head, halting head, semi-embedder
  pseudo-queries, coda, codebooks.
- **The budgets $(s_l,s_h,v_h,v_l)$, $\alpha,\beta$.**
- The hormone boxes.
- Probe inputs with SHA-256 hashes of their outputs.
- Lineage.
- A whole-file SHA-256.

Pickle-free, 4096-aligned tensors.

**Theorem (certificate soundness).** The loader materializes every certified matrix from
its stored Cayley parameters and scale logits and computes its spectral norm (float64
SVD). It cross-checks each norm against $\max_i\varsigma\,\mathrm{sigmoid}(\theta_i)$
(T5), then evaluates $\Gamma$ (C1) and $r_n$ (E3) **from those norms, not from any
claim in the header**. If $\Gamma<1$, every theorem of §3 holds for the loaded model on
every input.

**Proof.** T5 turns the stored parameters into exact norm bounds, and the SVD also covers
floating-point rounding in the materialized factors. S1 and N1 are invariants of the
code path, pinned by tests. C1 turns the bounds into (A). E2–E6 follow. ∎

`EquilibriumCore.bounds()` is this computation.

**Other load checks.** The probe self-test (bit-exact output hashes) verifies that the
bytes still compute what they computed when saved.

### F2 — `knowledge.pt`: 4-D memory as SSD-unified memory

**Contents.**
- A standard PyTorch archive: sub-keys $K^{(1)},K^{(2)}$, values
  $V\in\mathbb{H}^{N\times k_v}$ (fp16, stored as $N\times4k_v$), usage, confidence,
  per-row generation, and a schema hash matching the `.hrmc`.
- `torch.load(weights_only=True, mmap=True)` reads it anywhere.
- At runtime, the value payload is mapped `MAP_SHARED`, so rows are read and written in
  place on SSD. Measured feasible on torch 2.11: entries are stored uncompressed and
  64-byte aligned.

**Write safety.**
- A test-time write reaches disk only if $E^{\rm loo}=0$ after it (verified-write rule).
- Writes go through a WAL journal, and `seal()` restores the zip CRCs.

Without this file, the memory read is $0$. By E5 the model still converges and reasons
in context: an empty brain.

---

## 10. Verification map (theorem → executable check)

Every row is a test in `tests/test_eqb_theorems.py` (23 tests, float64, seeded), run
against the code in the `hrm_eqb` package. **Status 2026-09-27: all pass**, and the full
suite passes (243 passed, 1 skipped).

| Theorem | Numerical check | Code |
|---|---|---|
| T1 | RoPE == explicit per-pair rotation; relative-position identity | `algebra.rope` |
| T2 | softmax(dot) == softmax(−L2/2); logit bound; finite-difference Lipschitz ratio $\le\frac\beta2\sqrt{Cm}$ | `algebra.phasor_attention` |
| T3 | $\lvert pq\rvert=\lvert p\rvert\lvert q\rvert$; inverse by conjugate; isometry; $ij=-ji$; complex subalgebra | `algebra.qmul` |
| T4 | $qv\bar q$ is in $SO(3)$ and composes as $q_2q_1$, with $\pm q$ equal; $px\bar q$ is orthogonal with det $+1$ (surjectivity by proof only) | `algebra.rotate3/4` |
| T5 | SVD norm $=\max s$; Cayley orthogonality; $<\varsigma$ at $\theta=\pm30$; $\le\varsigma$ when saturated | `certified.CertifiedLinear` |
| S1 | sub-stochastic routing has spectral norm $\le1$; idempotent; rejects negatives | `certified.substochastic` |
| N1 | Lipschitz ratio $\le1$ for shrink $\in\{0,0.3,1\}$; per-block rotation equivariance | `certified.radial_relu` |
| C1 | measured ratios $\le(\rho_L,\kappa_L,a,b)$; measured norms $\le$ budgets; $\Gamma_A=1-\beta(1-\Gamma)$; $\Gamma\ge1$ budgets rejected | `equilibrium.EquilibriumCore` |
| C1/E3 sharpness | aligned linear core: ratio $=\rho_L$ and spectral radius of cycle Jacobian $=r_n$, to $10^{-10}$ | same |
| E3 | closed-form $r_n$ == eigenvalue; left Perron vector; cycle contraction $\le r_n$ in $\lVert\cdot\rVert_w$; $r_{200}=\Gamma_A$ | `ContractionBounds.perron` |
| E2/E3 | same equilibrium for $n_L\in\{1,2,6\}$ and different starts; joint fixed-point residuals $<10^{-9}$ | `EquilibriumCore.solve` |
| E4 | spectral radius of $J\le r$; $\lVert Ju\rVert_w\le r\lVert u\rVert_w$; Neumann tail $\le r^k/(1-r)$ for $k\le8$, primal and dual (VJP) | autograd Jacobian |
| E5 | 5 random contexts converge; equilibrium shift $\le L_c/(1-r)\,\lVert\Delta c\rVert$ | `ContractionBounds.injection_gain` |
| E6 | a-posteriori bound on 40 iterates; halting at $10^{-4}$ gives $\lVert h-h^*\rVert\le10^{-4}$ | `EquilibriumCore.solve` |
| K1 | inverse; unit composition; non-commutative two-hop; frame invariance | `algebra.relation_distance` |
| M1 | product top-$k$ == brute-force top-$k$, bit-exact, 3 shapes | `product_keys.product_topk` |
| L1/L2 | seeded codebook max similarity $<0.4$ at $m=\lvert\mathcal V\rvert=256$; decode unchanged under noise $<\gamma/(2L)$ | `algebra.phasor_codebook`, `cleanup` |
| A1 | zero query == mean; weights sum to 1; $\lVert u\rVert\le\max\lVert v_i\rVert$ per token | `attnres.DepthAttention` |
| H1 | box invariance under wild drives; geometric convergence to $\Pi_{\rm box}(g)$ at rate $1-\eta_m$ | `neuromodulation.Neuromodulators` |
| Q1 | EIG by explicit Bayesian posterior == vote entropy; argmax probe | `curiosity` |
| G1 | margins at $n=108$ and $4000$; Monte-Carlo false-keep $\le\delta$ (20 000 trials); Bonferroni | `gate` |

**Mutation checks (not part of the suite).** Removing the sub-stochastic
normalization gives a measured ratio of $1.52$ against $\rho_L=0.48$, and a gain-$1.5$
ReLU gives a ratio of $1.21$ against $1$. Both are caught by the C1 and N1 checks. On
random pairs the measured ratios reach 25–45% of the certified constants; the sharpness
test shows the constants are nonetheless attained.

---

## 11. What is not proved (and cannot be, by these means)

1. **Learning.** Nothing guarantees training finds weights that solve tasks. The proofs
   hold for *any* weights, including useless ones.
2. **Generalization.** Demonstration consistency ($E^{\rm loo}=0$) is not designer
   intent (PEM R7). The committee (§7) exposes ambiguity but cannot remove it without a
   question.
3. **Expressivity cost.** Contraction by construction restricts the hypothesis class.
   Whether $\Gamma<1$ models match unconstrained TRM accuracy is an experiment (ablation
   A1′: budgets vs unconstrained), not a theorem.
4. **Across segments.** The segment loop (memory reads, routing updates, AttnRes) is a
   learned refinement. Each segment is certified, but the sequence of segments is not
   proved to converge.
5. **Language faithfulness.** L2 proves the H-state decodes stably. It does not prove the
   coda's sentences are true; answers still come only from the structured head.
6. **Floating point.** The theorems hold in real arithmetic. The code keeps them sound in
   float64 by certifying from SVD norms of the materialized matrices (F1). The numerical
   checks use tolerances of $10^{-12}$ to $10^{-10}$.
7. **What is built.** Built and verified: the certified operators, the equilibrium core
   with its certificate and halting rule, AttnRes, product-key search, hormones,
   curiosity and the gate. **Not yet built:** the prelude, the memory module and its
   `knowledge.pt` file, the structured head, the coda, training, the `.hrmc` writer and
   loader, and the ARC, Sudoku and language benchmarks. These are phases N1–N9 of
   [[ADR-005 Neural HRM-Phi Replaces Program Search]].

---

## 12. Default budgets

| Budget | Value | Consequence |
|---|---|---|
| $s_l, s_h, v_h, v_l$ | 0.5, 0.25, 0.4, 0.4 | $\Gamma=0.4+0.4\cdot0.25/0.5=0.6$ |
| $\alpha,\beta$ | 0.9, 0.9 | $\Gamma_A=1-0.9\cdot0.4=0.64$ |
| $n_L$ | 6 | $r_6$ from E3 (computed by the loader) |

Smaller $\Gamma$ buys faster convergence and a better 1-step gradient (E4 needs
$r<\tfrac12$). Larger $\Gamma$ buys expressivity. The budgets are the one dial that trades
provability against capacity.

---

## 13. References

- Wang et al., HRM (arXiv:2506.21734); ARC Prize, *The Hidden Drivers of HRM's Performance on ARC-AGI* (2025); Jolicoeur-Martineau, TRM (arXiv:2510.04871).
- Kimi Team, *Attention Residuals* (arXiv:2603.15031, 2026).
- Geiping et al., recurrent-depth latent reasoning (arXiv:2502.05171).
- Berges et al., *Memory Layers at Scale* (ICML 2025); Lample et al., product keys (2019).
- Sun et al., RotatE (ICLR 2019); Zhang et al., QuatE (NeurIPS 2019).
- Su et al., RoFormer (2021); Kim et al., *The Lipschitz Constant of Self-Attention* (ICML 2021).
- Bai et al., DEQ (2019), Jacobian regularization (2021); Winston & Kolter, monotone operator equilibrium networks (2020).
- Doya, *Metalearning and neuromodulation*, Neural Networks 15 (2002); Yu & Dayan, uncertainty, neuromodulation and attention (2005).
- Oudeyer & Kaplan, intrinsic motivation / learning progress (2007); Houlsby et al., BALD (2011).
- Hoeffding (1963); Horn & Johnson, *Matrix Analysis* (Perron–Frobenius); Conway & Smith, *On Quaternions and Octonions* (2003).
- Karpathy, autoresearch (2026); PEM v1.1 design document.
