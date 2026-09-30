"""HRM-EQB model build (spec §14): tasks, task field, routing, memory, training-gradient lemma, learning POC."""

import copy
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hrm_eqb.certified import substochastic  # noqa: E402
from hrm_eqb.config import PRESETS, ModelConfig  # noqa: E402
from hrm_eqb.equilibrium import ContractionBounds, Context, EquilibriumCore, mix  # noqa: E402
from hrm_eqb.field import (IDENTITY, MASK, VOID, answers_equal, budget_chunks, canvas_of, collate,  # noqa: E402
                           decode_answer, encode_task, random_augmentation)
from hrm_eqb.model import HRMEQB  # noqa: E402
from hrm_eqb.tasks import FAMILIES, Task, count_solutions, generate, load_arc, load_sudoku_csv  # noqa: E402
from hrm_eqb.train import TrainConfig, answer_loss, coda_loss, exact, task_batches  # noqa: E402

F64 = torch.float64
TINY = PRESETS["tiny"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def all_families(seed: int = 0, max_side: int = 6):
    rng = np.random.default_rng(seed)
    return [generate(d, f, rng, max_side) for d, fams in FAMILIES.items() for f in fams]


# ---------- config ----------

def test_config_round_trip_is_strict():
    cfg = PRESETS["small"]
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**cfg.to_dict(), "extra": 1})
    missing = cfg.to_dict()
    missing.pop("dim")
    with pytest.raises(ValueError):
        ModelConfig.from_dict(missing)
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, s_h=0.9, v_l=0.9)       # Γ >= 1 is refused


# ---------- tasks ----------

def test_every_family_generates_consistent_tasks():
    for task in all_families():
        assert task.answer is not None and task.description
        if task.domain == "sudoku":
            count, solution = count_solutions(task.query)
            assert count == 1 and np.array_equal(solution, task.answer)
            for axis in (0, 1):
                assert (np.sort(task.answer, axis=axis) == np.arange(1, 10).reshape((-1, 1) if axis == 0 else (1, -1))).all()


def test_loaders_read_arc_and_sudoku_formats(tmp_path):
    (tmp_path / "abc.json").write_text(
        '{"train": [{"input": [[1, 0]], "output": [[0, 1]]}], "test": [{"input": [[2, 0]], "output": [[0, 2]]}]}')
    arc = load_arc(tmp_path)
    assert len(arc) == 1 and arc[0].family == "abc" and arc[0].answer.tolist() == [[0, 2]]
    rng = np.random.default_rng(0)
    task = generate("sudoku", "sudoku", rng)
    q = "".join("." if v == 0 else str(v) for v in task.query.ravel())
    a = "".join(str(v) for v in task.answer.ravel())
    (tmp_path / "s.csv").write_text(f"source,question,answer,rating\nx,{q},{a},3\n")
    loaded = load_sudoku_csv(tmp_path / "s.csv", limit=5)
    assert len(loaded) == 1 and np.array_equal(loaded[0].query, task.query)


# ---------- task field ----------

def test_field_encodes_and_decodes_every_family():
    for task in all_families(seed=3, max_side=10):
        enc = encode_task(task)
        assert enc.target is not None, (task.domain, task.family)
        assert answers_equal(decode_answer(enc, enc.target), task.answer)
        assert (enc.content[enc.answer_pos] == MASK).all()
        assert enc.content.size == (2 * len(task.demos) + 2) * enc.canvas[0] * enc.canvas[1]


def test_canvas_covers_scaled_and_transposed_answers():
    rng = np.random.default_rng(11)
    for domain, family in (("text", "double"), ("grid", "rotate"), ("grid", "transpose")):
        for _ in range(100):
            assert encode_task(generate(domain, family, rng, 10)).target is not None


def test_augmentation_is_an_exact_group_action():
    """Lemma G1a: g⁻¹(g(x)) = x, and the answer of g·T is g(answer of T)."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        g = random_augmentation(rng)
        x = rng.integers(0, 10, size=(int(rng.integers(1, 7)), int(rng.integers(1, 7))))
        assert np.array_equal(g.inverse(g.forward(x)), x)
        assert (g.forward(x) == 0).sum() == (x == 0).sum()          # background fixed
    task = generate("grid", "rotate", rng, 6)
    g = random_augmentation(rng)
    moved = g.apply(task)
    assert np.array_equal(g.inverse(moved.answer), task.answer)
    assert np.array_equal(moved.answer, g.forward(task.answer))
    assert IDENTITY.apply(task).query.tolist() == task.query.tolist()


def test_budget_chunks_bound_padded_attention_pairs():
    encoded = [encode_task(t) for t in all_families(seed=2, max_side=10)]
    groups = budget_chunks(encoded, pairs=300_000)
    assert sorted(i for g in groups for i in g) == list(range(len(encoded)))
    for g in groups:
        n = max(encoded[i].content.size for i in g)
        assert len(g) == 1 or len(g) * n * n <= 300_000


def test_training_batches_respect_budgets():
    cfg = TrainConfig(batch_tokens=4000, batch_pairs=500_000)
    stream = task_batches(np.random.default_rng(0), cfg, 10)
    for _ in range(20):
        tasks = next(stream)
        n = max(encode_task(t).content.size for t in tasks)
        assert len(tasks) == 1 or (len(tasks) * n <= 4000 and len(tasks) * n * n <= 500_000)


# ---------- routing and memory inside the model ----------

def test_model_routing_is_substochastic_per_head():
    """S1 + S2 on the model's own routings: every head has row/col sums <= 1 and norm <= 1."""
    model = HRMEQB(TINY)
    batch = collate([encode_task(t) for t in all_families()])
    with torch.no_grad():
        x = model.prelude(batch)
        route = model.routing(model.route_low, x, batch.valid, gain=1.5)
    assert route.shape[1] == TINY.heads
    assert float(route.sum(-1).max()) <= 1 + 1e-5 and float(route.sum(-2).max()) <= 1 + 1e-5
    norms = torch.linalg.matrix_norm(route.double(), ord=2)
    assert float(norms.max()) <= 1 + 1e-5
    pad = ~batch.valid
    assert float(route.masked_select(pad[:, None, :, None].expand_as(route)).abs().max()) == 0.0


def test_headwise_mixing_is_nonexpansive():
    """S2: ||[A_1 X_1, ..., A_H X_H]|| <= max_h ||A_h|| ||X||."""
    for _ in range(20):
        route = substochastic(torch.rand(3, 4, 7, 7, dtype=F64))
        x = torch.randn(3, 7, 16, dtype=F64)
        out = mix(route, x)
        bound = torch.linalg.matrix_norm(route, ord=2).amax(dim=-1)
        per_item = torch.linalg.vector_norm(out, dim=(-2, -1)) / torch.linalg.vector_norm(x, dim=(-2, -1))
        assert bool((per_item <= bound + 1e-12).all())


def test_memory_read_is_bounded_by_the_rows_it_reads():
    """K2: ||(Σ α_i v_i) ⊗ r|| <= max_i ||v_i|| for every reader."""
    model = HRMEQB(TINY)
    mem = model.memory
    with torch.no_grad():
        mem.values.normal_(0, 3)
        u = torch.randn(5, 11, TINY.dim)
        rotated, slots = mem.mixed(u)
        norms = torch.linalg.vector_norm(rotated.flatten(-2), dim=-1)
        read_max = torch.linalg.vector_norm(mem.values[slots], dim=-1).amax(dim=-1)
    assert bool((norms <= read_max + 1e-5).all())
    mem.silent = True
    with torch.no_grad():
        assert float(mem.mixed(u)[0].abs().max()) == 0.0


def test_padding_does_not_change_a_task():
    """A task's answer logits do not depend on what it is batched with (masking is exact)."""
    cfg = dataclasses.replace(TINY, solve_tol=1e-7, max_cycles=400)
    model = HRMEQB(cfg).eval()
    small, large = generate("sequence", "reverse", np.random.default_rng(1)), generate("grid", "flip", np.random.default_rng(2), 6)
    with torch.no_grad():
        alone = model(collate([encode_task(small)]))[-1].logits[0]
        together = model(collate([encode_task(small), encode_task(large)]))[-1].logits[0, :alone.shape[0]]
    assert float((alone - together).abs().max()) < 1e-3
    assert torch.equal(alone.argmax(-1), together.argmax(-1))


def test_schedule_changes_speed_not_answers():
    """E3 at model level: n_low = 1 and n_low = 6 reach the same equilibria (within the tolerances)."""
    tasks = all_families()
    batch = collate([encode_task(t) for t in tasks])
    logits = []
    base = HRMEQB(TINY).eval()
    for n_low in (1, 6):
        model = copy.deepcopy(base)
        model.cfg = dataclasses.replace(TINY, n_low=n_low, solve_tol=1e-6, max_cycles=500)
        with torch.no_grad():
            logits.append(model(batch)[-1].logits)
    assert float((logits[0] - logits[1]).abs().max()) < 1e-3


def test_model_certificate_and_finite_gradients():
    model = HRMEQB(TINY)
    tasks = all_families()
    batch = collate([encode_task(t) for t in tasks])
    out = model(batch)
    assert all(o.error_bound <= TINY.solve_tol for o in out)
    loss = answer_loss(out, batch, 0.2) + coda_loss(model, out, batch, tasks)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert model.core.bounds().gamma() < 1.0


def test_coda_cannot_change_answers():
    """L3: the coda loss has zero gradient on every reasoning weight (it reads a detached chain)."""
    model = HRMEQB(TINY)
    tasks = all_families()[:6]
    batch = collate([encode_task(t) for t in tasks])
    out = model(batch)
    coda_loss(model, out, batch, tasks).backward()
    for name, p in model.named_parameters():
        if not name.startswith("coda."):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, name


# ---------- E7: training gradient = truncated Neumann series ----------

def test_e7_k_cycle_backprop_is_the_truncated_neumann_series():
    tokens, dim, n_low = 4, 8, 2
    core = EquilibriumCore(dim, ContractionBounds(0.5, 0.25, 0.4, 0.4, 0.9, 0.9), shrink=0.1).double()
    with torch.no_grad():
        for p in core.parameters():
            p.normal_(0, 1)
    route = torch.rand(tokens, tokens, dtype=F64)
    inject = torch.randn(tokens, dim, dtype=F64, requires_grad=True)
    fixed_high = torch.randn(tokens, dim, dtype=F64)

    def ctx_of(c):
        return Context(route, route.T, c, fixed_high)

    with torch.no_grad():
        star = core.solve(torch.zeros(tokens, dim, dtype=F64), torch.zeros(tokens, dim, dtype=F64), ctx_of(inject),
                          n_low, tol=1e-14, max_cycles=5000)
    z = torch.cat((star.high.flatten(), star.low.flatten()))
    half = tokens * dim

    def cyc(zz, c):
        h, l = core.cycle(zz[:half].view(tokens, dim), zz[half:].view(tokens, dim), ctx_of(c), n_low)
        return torch.cat((h.flatten(), l.flatten()))

    jz = torch.autograd.functional.jacobian(lambda zz: cyc(zz, inject.detach()), z)
    jc = torch.autograd.functional.jacobian(lambda c: cyc(z, c), inject.detach()).reshape(2 * half, half)
    v = torch.randn(2 * half, dtype=F64)
    exact_grad = jc.T @ torch.linalg.solve(torch.eye(2 * half, dtype=F64) - jz.T, v)
    errors = []
    for k in range(1, 7):
        h, l = star.high, star.low
        for _ in range(k):
            h, l = core.cycle(h, l, ctx_of(inject), n_low)
        (g,) = torch.autograd.grad((torch.cat((h.flatten(), l.flatten())) * v).sum(), inject)
        series, term = torch.zeros_like(v), v.clone()
        for _ in range(k):
            series, term = series + term, jz.T @ term
        assert torch.allclose(g.flatten(), jc.T @ series, atol=1e-9)
        errors.append(float((g.flatten() - exact_grad).norm()))
    assert all(b < a for a, b in zip(errors, errors[1:]))


# ---------- learning POC through the certified core ----------

def test_certified_model_overfits_a_fixed_task_set():
    """ADR-005 N3(a): a tiny certified model fits 8 fixed tasks exactly (learning works through the core)."""
    torch.set_num_threads(4)
    rng = np.random.default_rng(1)
    tasks = [generate("sequence", "reverse", rng) for _ in range(8)]
    batch = collate([encode_task(t) for t in tasks])
    model = HRMEQB(TINY)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    solved = 0.0
    for step in range(1, 601):
        loss = answer_loss(model(batch, tol=1e-2), batch, 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                solved = float(exact(model(batch)[-1], batch).float().mean())
            model.train()
            if solved == 1.0:
                break
    assert solved == 1.0
    assert model.core.bounds().gamma() < 1.0     # trained weights are still certified


def test_encoded_padding_uses_void():
    task = Task("sequence", "x", [(np.array([[1, 2]]), np.array([[3]]))], np.array([[4, 5]]), np.array([[6]]))
    enc = encode_task(task)
    assert enc.canvas == (1, 2)
    assert enc.target.tolist()[1] == VOID
    assert canvas_of(task) == (1, 2)


def test_training_resumes_from_a_rolling_checkpoint(tmp_path):
    """A crash loses at most checkpoint_every steps: training restarts after the saved step."""
    from hrm_eqb.train import train
    ck = str(tmp_path / "checkpoint-train-latest.pt")
    cfg = dict(batch_tokens=512, batch_pairs=50_000, checkpoint=ck, checkpoint_every=2, log_every=1, coda_weight=0.0)
    train(HRMEQB(TINY), TrainConfig(steps=4, **cfg), torch.device("cpu"), log=lambda s: None)
    assert torch.load(ck, weights_only=True)["step"] == 4
    logs = []
    train(HRMEQB(TINY), TrainConfig(steps=6, **cfg), torch.device("cpu"), log=logs.append)
    assert "resumed" in logs[0] and "step 5" in logs[0]
    assert torch.load(ck, weights_only=True)["step"] == 6
