"""Numerical checks of every theorem in 'HRM-EQB Specification v0.2 - Theorems and Proofs' (float64)."""

import math

import pytest

torch = pytest.importorskip("torch")

from hrm_eqb import (  # noqa: E402
    CertifiedLinear, ContractionBounds, Context, DepthAttention, EquilibriumCore, Neuromodulators,
    as_complex, as_real, cayley, cleanup, keep_margin, most_informative, phasor_attention,
    phasor_codebook, product_topk, qconj, qmul, radial_relu, relation_distance, rope, rotate3,
    rotate4, should_keep, substochastic, vote_entropy,
)
from hrm_eqb.equilibrium import weighted_distance  # noqa: E402

F64 = torch.float64
TOL = 1e-12
TOKENS, DIM = 6, 16
BUDGETS = ContractionBounds(s_l=0.5, s_h=0.25, v_h=0.4, v_l=0.4, alpha=0.9, beta=0.9)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def unit_quaternions(*shape):
    q = torch.randn(*shape, 4, dtype=F64)
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def random_core(shrink: float = 0.1) -> EquilibriumCore:
    core = EquilibriumCore(DIM, BUDGETS, shrink=shrink).double()
    with torch.no_grad():
        for p in core.parameters():
            p.normal_(0.0, 1.0)
    return core


def random_context() -> Context:
    return Context(torch.rand(TOKENS, TOKENS, dtype=F64), torch.rand(TOKENS, TOKENS, dtype=F64),
                   torch.randn(TOKENS, DIM, dtype=F64), torch.randn(TOKENS, DIM, dtype=F64))


def state():
    return torch.randn(TOKENS, DIM, dtype=F64)


def ratio(num, den) -> float:
    return float(torch.linalg.vector_norm(num) / torch.linalg.vector_norm(den))


# ---------- Â§2 algebra ----------

def test_t1_rope_is_phasor_binding():
    m = 8
    x, k = torch.randn(3, 2 * m, dtype=F64), torch.randn(3, 2 * m, dtype=F64)
    freqs = 10000.0 ** (-torch.arange(m, dtype=F64) / m)
    pos, pos2 = torch.tensor([0.0, 3.0, 17.5], dtype=F64), torch.tensor([2.0, -1.0, 4.0], dtype=F64)
    ang = pos[:, None] * freqs
    even, odd = x[:, 0::2], x[:, 1::2]
    explicit = torch.stack((even * ang.cos() - odd * ang.sin(), even * ang.sin() + odd * ang.cos()), -1).flatten(-2)
    assert torch.allclose(rope(x, pos, freqs), explicit, atol=TOL)
    lhs = (rope(x, pos, freqs) * rope(k, pos2, freqs)).sum(-1)
    rhs = (rope(x, pos - pos2, freqs) * k).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-10)
    assert torch.allclose(as_real(as_complex(x)), x)


def test_t2_torus_attention_is_l2_attention_and_lipschitz():
    m, c, beta = 12, 9, 3.0
    keys = phasor_codebook(c, m, seed=1)
    q = phasor_codebook(1, m, seed=2)[0]
    dist2 = (torch.abs(q - keys) ** 2).sum(-1)
    assert torch.allclose(phasor_attention(q, keys, beta), (-beta / 2 * dist2).softmax(-1), atol=TOL)
    assert float((q @ keys.conj().T).real.abs().max()) <= m + TOL
    bound = beta / 2 * math.sqrt(c * m)
    for _ in range(50):
        q1, q2 = torch.randn(2 * m, dtype=F64) * 3, torch.randn(2 * m, dtype=F64) * 3
        p1 = phasor_attention(as_complex(q1), keys, beta)
        p2 = phasor_attention(as_complex(q2), keys, beta)
        assert ratio(p1 - p2, q1 - q2) <= bound + TOL


def test_t3_quaternion_binding():
    p, q, x = unit_quaternions(50) * 2.0, torch.randn(50, 4, dtype=F64), torch.randn(50, 4, dtype=F64)
    r = unit_quaternions(50)
    norm = lambda t: torch.linalg.vector_norm(t, dim=-1)  # noqa: E731
    assert torch.allclose(norm(qmul(p, q)), norm(p) * norm(q), atol=TOL)
    assert torch.allclose(qmul(qmul(x, r), qconj(r)), x, atol=TOL)
    assert torch.allclose(norm(qmul(x, r) - qmul(q, r)), norm(x - q), atol=TOL)
    i, j, k = torch.eye(4, dtype=F64)[1:]
    assert torch.equal(qmul(i, j), k) and torch.equal(qmul(j, i), -k)
    a, b = torch.randn(2, dtype=F64), torch.randn(2, dtype=F64)
    embed = lambda z: torch.cat((z, torch.zeros(2, dtype=F64)))  # noqa: E731
    product = torch.view_as_complex(a.contiguous()) * torch.view_as_complex(b.contiguous())
    assert torch.allclose(qmul(embed(a), embed(b))[:2], torch.view_as_real(product), atol=TOL)


def test_t4_quaternions_rotate_3d_and_4d():
    q1, q2 = unit_quaternions(), unit_quaternions()
    basis3 = torch.eye(3, dtype=F64)
    rot = torch.stack([rotate3(q1, e) for e in basis3], dim=1)
    assert torch.allclose(rot.T @ rot, torch.eye(3, dtype=F64), atol=TOL)
    assert abs(float(torch.linalg.det(rot)) - 1.0) < TOL
    v = torch.randn(3, dtype=F64)
    assert torch.allclose(rotate3(q2, rotate3(q1, v)), rotate3(qmul(q2, q1), v), atol=TOL)
    assert torch.allclose(rotate3(-q1, v), rotate3(q1, v), atol=TOL)
    rot4 = torch.stack([rotate4(q1, q2, e) for e in torch.eye(4, dtype=F64)], dim=1)
    assert torch.allclose(rot4.T @ rot4, torch.eye(4, dtype=F64), atol=TOL)
    assert abs(float(torch.linalg.det(rot4)) - 1.0) < TOL


def test_t5_certified_linear_has_exact_spectral_norm():
    lin = CertifiedLinear(DIM, budget=0.7).double()
    with torch.no_grad():
        lin.left.normal_(0, 2)
        lin.logit.copy_(torch.linspace(-30, 30, DIM, dtype=F64))
    q = cayley(lin.left)
    assert torch.allclose(q.T @ q, torch.eye(DIM, dtype=F64), atol=1e-10)
    assert abs(lin.spectral_norm() - float(lin.scales().detach().max())) < 1e-10
    assert float(lin.scales().detach().max()) < 0.7
    with torch.no_grad():
        lin.logit.fill_(1e4)
    assert lin.spectral_norm() <= 0.7 * (1 + 1e-12)


def test_s1_substochastic_routing_is_nonexpansive():
    for n in (1, 3, 8, 32):
        a = substochastic(torch.rand(n, n, dtype=F64) * 5)
        assert float(torch.linalg.matrix_norm(a, ord=2)) <= 1.0 + TOL
        assert torch.equal(substochastic(a), a)
    with pytest.raises(ValueError):
        substochastic(-torch.ones(2, 2))


def test_n1_radial_relu_is_1_lipschitz_and_equivariant():
    for shrink in (0.0, 0.3, 1.0):
        for _ in range(200):
            x, y = torch.randn(DIM, dtype=F64), torch.randn(DIM, dtype=F64) * 0.5
            assert ratio(radial_relu(x, shrink) - radial_relu(y, shrink), x - y) <= 1.0 + TOL
        rot = torch.linalg.qr(torch.randn(DIM // 4, 4, 4, dtype=F64)).Q
        x = torch.randn(DIM // 4, 4, dtype=F64)
        turned = (rot @ x[..., None]).squeeze(-1)
        lhs = radial_relu(turned.flatten(), shrink)
        rhs = (rot @ radial_relu(x.flatten(), shrink).reshape(-1, 4)[..., None]).squeeze(-1).flatten()
        assert torch.allclose(lhs, rhs, atol=TOL)


# ---------- Â§3 equilibrium ----------

def test_c1_measured_lipschitz_within_certificate():
    core, ctx = random_core(), random_context()
    cert = core.bounds()
    for measured, budget in zip((cert.s_l, cert.s_h, cert.v_h, cert.v_l),
                                (BUDGETS.s_l, BUDGETS.s_h, BUDGETS.v_h, BUDGETS.v_l)):
        assert 0.0 < measured <= budget
    rho, kappa, a, b = cert.lipschitz()
    with torch.no_grad():
        for _ in range(30):
            l1, l2, h1, h2 = state(), state(), state(), state()
            assert ratio(core.low_step(l1, h1, ctx) - core.low_step(l2, h1, ctx), l1 - l2) <= rho + TOL
            assert ratio(core.low_step(l1, h1, ctx) - core.low_step(l1, h2, ctx), h1 - h2) <= kappa + TOL
            assert ratio(core.high_step(h1, l1, ctx) - core.high_step(h2, l1, ctx), h1 - h2) <= a + TOL
            assert ratio(core.high_step(h1, l1, ctx) - core.high_step(h1, l2, ctx), l1 - l2) <= b + TOL
    assert cert.gamma() < 1.0
    assert abs(cert.gamma_a() - (1 - cert.beta * (1 - cert.gamma()))) < TOL
    with pytest.raises(ValueError):
        EquilibriumCore(DIM, ContractionBounds(0.5, 0.9, 0.5, 0.9, 0.9, 0.9))


@pytest.mark.parametrize("n_low", [1, 2, 6])
def test_e3_perron_rate_and_cycle_contraction(n_low):
    core, ctx = random_core(), random_context()
    cert = core.bounds()
    rate, w = cert.perron(n_low)
    m = cert.cycle_matrix(n_low)
    assert abs(rate - float(torch.linalg.eigvals(m).real.max())) < TOL
    assert torch.allclose(torch.tensor(w, dtype=F64) @ m, rate * torch.tensor(w, dtype=F64), atol=TOL)
    assert w[1] > 0 and rate < 1.0
    with torch.no_grad():
        for _ in range(30):
            h1, l1, h2, l2 = state(), state(), state(), state()
            c1, c2 = core.cycle(h1, l1, ctx, n_low), core.cycle(h2, l2, ctx, n_low)
            out = weighted_distance(c1[0] - c2[0], c1[1] - c2[1], w)
            assert float(out) <= rate * float(weighted_distance(h1 - h2, l1 - l2, w)) + TOL
    assert abs(cert.perron(200)[0] - cert.gamma_a()) < 1e-10


def test_e2_e3_same_equilibrium_for_every_schedule_and_start():
    core, ctx = random_core(), random_context()
    highs = []
    with torch.no_grad():
        for n_low in (1, 2, 6):
            for _ in range(2):
                sol = core.solve(state() * 3, state() * 3, ctx, n_low, tol=1e-11, max_cycles=2000)
                assert sol.converged and sol.error_bound <= 1e-11
                assert float(torch.linalg.vector_norm(core.low_step(sol.low, sol.high, ctx) - sol.low)) < 1e-9
                assert float(torch.linalg.vector_norm(core.high_step(sol.high, sol.low, ctx) - sol.high)) < 1e-9
                highs.append(sol.high)
    for h in highs[1:]:
        assert float(torch.linalg.vector_norm(h - highs[0])) <= 2e-11 + 1e-13


def _cycle_jacobian(core, ctx, high, low, n_low):
    def flat_cycle(z):
        h, l = z[: TOKENS * DIM].view(TOKENS, DIM), z[TOKENS * DIM:].view(TOKENS, DIM)
        nh, nl = core.cycle(h, l, ctx, n_low)
        return torch.cat((nh.flatten(), nl.flatten()))
    z = torch.cat((high.flatten(), low.flatten())).detach()
    return torch.autograd.functional.jacobian(flat_cycle, z)


def test_c1_e3_constants_are_attained_by_the_aligned_linear_core():
    """Sharpness: with W = s I, identity routing and no shrink, every bound holds with equality."""
    core = EquilibriumCore(DIM, BUDGETS, shrink=0.0).double()
    with torch.no_grad():
        for lin in (core.low_self, core.low_from_high, core.high_self, core.high_from_low):
            lin.left.zero_()
            lin.right.zero_()
            lin.logit.fill_(2.0)
    eye = torch.eye(TOKENS, dtype=F64)
    ctx = Context(eye, eye, torch.randn(TOKENS, DIM, dtype=F64), torch.randn(TOKENS, DIM, dtype=F64))
    cert = core.bounds()
    rho = cert.lipschitz()[0]
    with torch.no_grad():
        l1, l2, h = state(), state(), state()
        assert abs(ratio(core.low_step(l1, h, ctx) - core.low_step(l2, h, ctx), l1 - l2) - rho) < TOL
    for n_low in (1, 3):
        jac = _cycle_jacobian(core, ctx, state(), state(), n_low)
        radius = float(torch.linalg.eigvals(jac).abs().max())
        assert abs(radius - cert.perron(n_low)[0]) < 1e-10


def test_e4_neumann_gradient_error_bound():
    core, ctx, n_low = random_core(), random_context(), 2
    with torch.no_grad():
        sol = core.solve(state(), state(), ctx, n_low, tol=1e-12, max_cycles=2000)
    jac = _cycle_jacobian(core, ctx, sol.high, sol.low, n_low)
    rate, (w_h, w_l) = core.bounds().perron(n_low)
    assert float(torch.linalg.eigvals(jac).abs().max()) <= rate + TOL
    half = TOKENS * DIM
    primal = lambda v: w_h * float(v[:half].norm()) + w_l * float(v[half:].norm())  # noqa: E731
    dual = lambda v: max(float(v[:half].norm()) / w_h, float(v[half:].norm()) / w_l)  # noqa: E731
    eye = torch.eye(2 * half, dtype=F64)
    for op, norm in ((jac, primal), (jac.T, dual)):
        v = torch.randn(2 * half, dtype=F64)
        assert norm(op @ v) <= rate * norm(v) + TOL
        exact = torch.linalg.solve(eye - op, v)
        partial, term = torch.zeros_like(v), v.clone()
        for k in range(1, 9):
            partial, term = partial + term, op @ term
            assert norm(exact - partial) <= rate ** k / (1 - rate) * norm(v) + 1e-10


def test_e5_every_context_converges_and_sensitivity_is_bounded():
    core, n_low = random_core(), 3
    cert = core.bounds()
    rate, w = cert.perron(n_low)
    gain = cert.injection_gain(n_low)
    with torch.no_grad():
        for _ in range(5):
            ctx = random_context()
            base = core.solve(state(), state(), ctx, n_low, tol=1e-12, max_cycles=2000)
            assert base.converged
            d_low, d_high = torch.randn(TOKENS, DIM, dtype=F64) * 0.1, torch.randn(TOKENS, DIM, dtype=F64) * 0.1
            moved = Context(ctx.route_low, ctx.route_high, ctx.inject_low + d_low, ctx.inject_high + d_high)
            shifted = core.solve(base.high, base.low, moved, n_low, tol=1e-12, max_cycles=2000)
            shift = float(weighted_distance(shifted.high - base.high, shifted.low - base.low, w))
            budget = gain / (1 - rate) * (float(d_low.norm()) + float(d_high.norm()))
            assert shift <= budget + 1e-10


def test_e6_residual_certifies_distance_to_equilibrium():
    core, ctx, n_low = random_core(), random_context(), 2
    rate, w = core.bounds().perron(n_low)
    with torch.no_grad():
        star = core.solve(state(), state(), ctx, n_low, tol=1e-13, max_cycles=4000)
        h, l = state() * 5, state() * 5
        for _ in range(40):
            nh, nl = core.cycle(h, l, ctx, n_low)
            err = float(weighted_distance(nh - star.high, nl - star.low, w))
            assert err <= rate / (1 - rate) * float(weighted_distance(nh - h, nl - l, w)) + 1e-11
            h, l = nh, nl
        halted = core.solve(state() * 5, state() * 5, ctx, n_low, tol=1e-4, max_cycles=2000)
    assert halted.converged
    assert float(torch.linalg.vector_norm(halted.high - star.high)) <= 1e-4 + 1e-12


# ---------- Â§4 knowledge ----------

def test_k1_relations_are_4d_rotations():
    k = 5
    e1, e2 = torch.randn(k, 4, dtype=F64), torch.randn(k, 4, dtype=F64)
    r1, r2, u = unit_quaternions(k), unit_quaternions(k), unit_quaternions(k)
    assert torch.allclose(qmul(qmul(e1, r1), qconj(r1)), e1, atol=TOL)
    both = qmul(r1, r2)
    assert torch.allclose(torch.linalg.vector_norm(both, dim=-1), torch.ones(k, dtype=F64), atol=TOL)
    assert float(torch.linalg.vector_norm(qmul(qmul(e1, r1), r2) - qmul(qmul(e1, r2), r1))) > 1e-3
    turned = relation_distance(qmul(e1, u), qmul(qmul(qconj(u), r1), u), qmul(e2, u))
    assert abs(float(turned) - float(relation_distance(e1, r1, e2))) < TOL


def test_m1_product_topk_equals_brute_force():
    for size_a, size_b, k in ((32, 32, 8), (64, 16, 16), (10, 10, 1)):
        s1, s2 = torch.randn(3, size_a, dtype=F64), torch.randn(3, size_b, dtype=F64)
        values, slots = product_topk(s1, s2, k)
        full = (s1[:, :, None] + s2[:, None, :]).flatten(-2)
        assert torch.equal(values, full.topk(k, dim=-1).values)
        assert torch.equal(full.gather(-1, slots), values)


# ---------- Â§5 language ----------

def test_l1_l2_codebook_separates_and_decodes_stably():
    m = size = 256
    book = phasor_codebook(size, m, seed=0)
    sim = (book @ book.conj().T).real / m
    sim.fill_diagonal_(-1.0)
    t = float(sim.max())
    assert t < 0.4
    margin = m * (1 - t)
    lipschitz = math.sqrt(m)
    for c in (0, 17, 255):
        noise = torch.randn(m, dtype=torch.complex128)
        noise *= 0.99 * margin / (2 * lipschitz) / torch.linalg.vector_norm(noise)
        assert int(cleanup(book[c] + noise, book)) == c


def test_a1_attention_residual_stays_in_convex_hull():
    chain = torch.randn(7, TOKENS, DIM, dtype=F64) * torch.linspace(0.1, 10, 7, dtype=F64)[:, None, None]
    attn = DepthAttention(DIM, queries=3).double()
    assert torch.allclose(attn(chain)[0], chain.mean(0), atol=TOL)
    with torch.no_grad():
        attn.query.normal_(0, 3)
    weights = attn.weights(chain)
    assert torch.allclose(weights.sum(1), torch.ones(3, TOKENS, dtype=F64), atol=TOL)
    out = attn(chain)
    per_token_max = torch.linalg.vector_norm(chain, dim=-1).amax(0)
    assert bool((torch.linalg.vector_norm(out, dim=-1) <= per_token_max + TOL).all())


# ---------- Â§6 hormones ----------

def test_h1_hormones_stay_in_box_and_converge():
    mods = Neuromodulators.create(lower=(0.1, 0.2, 0.1, 0.05), upper=(0.9, 0.95, 0.8, 0.5), rate=0.2)
    for _ in range(100):
        mods.step(torch.randn(4, dtype=F64) * 3)
        assert bool(((mods.lower <= mods.level) & (mods.level <= mods.upper)).all())
    drive = torch.tensor([2.0, 0.5, -1.0, 0.3], dtype=F64)
    target = torch.minimum(torch.maximum(drive, mods.lower), mods.upper)
    start = float((mods.level - target).abs().max())
    for t in range(1, 30):
        mods.step(drive)
        assert float((mods.level - target).abs().max()) <= (1 - mods.rate) ** t * start + TOL
    assert mods.segment_budget() == math.ceil(1 / (1 - mods.get("serotonin")))


# ---------- Â§7 curiosity ----------

def test_q1_expected_information_gain_is_vote_entropy():
    probes = torch.randint(0, 3, (10, 8, 2, 2))
    probes[4] = 1
    for outputs in probes:
        _, counts = torch.unique(outputs.reshape(8, -1), dim=0, return_counts=True)
        posterior_entropy = sum(float(c) / 8 * math.log2(float(c)) for c in counts)
        assert abs(vote_entropy(outputs) - (math.log2(8) - posterior_entropy)) < TOL
    best, bits = most_informative(probes)
    assert bits == max(vote_entropy(o) for o in probes)
    assert vote_entropy(probes[best]) == bits and vote_entropy(probes[4]) == 0.0


# ---------- Â§8 gate ----------

def test_g1_keep_gate_false_keep_rate():
    assert abs(keep_margin(108, 0.05) - 0.2355) < 1e-3
    assert abs(keep_margin(4000, 0.05) - 0.0387) < 1e-3
    gen = torch.Generator().manual_seed(1)
    trials, n, delta = 20000, 50, 0.1
    null = torch.randint(-1, 2, (trials, n), generator=gen).double()
    keeps = (null.mean(1) >= keep_margin(n, delta)).double().mean()
    assert float(keeps) <= delta
    assert should_keep(torch.ones(200, dtype=F64) * 0.5, 0.05, experiments=10)
    assert not should_keep(null[0], 0.05)
    with pytest.raises(ValueError):
        should_keep(torch.tensor([2.0]), 0.05)
