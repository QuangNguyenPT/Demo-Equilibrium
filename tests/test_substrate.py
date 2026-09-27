"""Substrate properties. Nothing downstream is meaningful if these fail."""
import numpy as np
import pytest

from hrm.config import SubstrateConfig
from hrm.substrate import (BackendRejected, Codebook, KernelBackend, PhasorVector, bundle,
                           get_backend, set_backend)
from hrm.substrate.pool import ByteLRUCache
from hrm.substrate.quantize import decode_angles, encode_angles, quantization_error

CFG = SubstrateConfig(dim=1024)
RNG = np.random.default_rng(7)


def rand(dim: int = CFG.dim) -> PhasorVector:
    return PhasorVector.random(dim, RNG)


class TestBindUnbind:
    def test_round_trip_exact(self):
        a, b = rand(), rand()
        assert np.allclose(((a * b) / b).data, a.data, atol=1e-5)

    @pytest.mark.parametrize("dim", [64, 256, 1024])
    def test_exact_at_any_dimension(self, dim):
        a, b = rand(dim), rand(dim)
        assert ((a * b) / b) @ a > 0.9999

    def test_commutative_and_associative(self):
        a, b, c = rand(), rand(), rand()
        assert np.allclose((a * b).data, (b * a).data, atol=1e-6)
        assert np.allclose(((a * b) * c).data, (a * (b * c)).data, atol=1e-5)

    def test_random_vectors_near_orthogonal(self):
        assert max(abs(rand() @ rand()) for _ in range(20)) < 5.0 / np.sqrt(CFG.dim)

    def test_similarity_bounds(self):
        a = rand()
        assert a @ a == pytest.approx(1.0, abs=1e-5)
        assert a @ ~a <= 1.0
        assert -1.0 <= a @ rand() <= 1.0


class TestFractionalPower:
    def test_integer_composition(self):
        x = rand()
        assert np.allclose((x ** 2).data, (x * x).data, atol=1e-5)
        assert np.allclose((x ** 0).data, np.ones(CFG.dim), atol=1e-6)

    def test_fractional_addition(self):
        x = rand()
        assert ((x ** 1.5) * (x ** 0.5)) @ (x ** 2.0) > 0.999

    def test_kernel_decays_smoothly(self):
        x = rand()
        sims = [(x ** 0.0) @ (x ** dx) for dx in (0.0, 0.5, 1.0, 2.0, 4.0)]
        assert sims[0] == pytest.approx(1.0, abs=1e-5)
        assert all(sims[i] >= sims[i + 1] - 0.05 for i in range(len(sims) - 1))


class TestSuperposition:
    @pytest.mark.parametrize("n_items", [5, 20, 50])
    def test_snr_matches_sqrt_prediction(self, n_items):
        items = [rand() for _ in range(n_items)]
        s = bundle(items)
        hits = [s @ it for it in items]
        misses = [s @ rand() for _ in range(10)]
        assert min(hits) > 3.0 * np.std(misses)
        assert np.mean(hits) == pytest.approx(1.0 / np.sqrt(n_items), rel=0.35)

    def test_retrieval_from_bound_superposition(self):
        pairs = [(rand(), rand()) for _ in range(8)]
        s = bundle([k * v for k, v in pairs])
        k0, v0 = pairs[0]
        cands = [("target", v0)] + [(f"d{i}", rand()) for i in range(9)]
        assert Codebook.cleanup(s / k0, cands, top_k=1)[0][0] == "target"

    def test_empty_bundle_rejected(self):
        with pytest.raises(ValueError):
            bundle([])


class TestCodebook:
    def test_same_seed_same_symbols_regardless_of_order(self):
        c1, c2 = Codebook(CFG), Codebook(CFG)
        c2.symbol("unrelated")
        assert np.array_equal(c1.symbol("X").data, c2.symbol("X").data)

    def test_different_seed_different_symbols(self):
        other = Codebook(SubstrateConfig(dim=CFG.dim, seed=CFG.seed + 1))
        assert abs(Codebook(CFG).symbol("X") @ other.symbol("X")) < 0.15

    def test_distinct_names_near_orthogonal(self):
        cb = Codebook(CFG)
        for a, b in (("Z_OBJ", "Z_TIME"), ("Z_OBJ", "Z_EX"), ("parent", "child")):
            assert abs(cb.symbol(a) @ cb.symbol(b)) < 0.15

    def test_circular_symbol_is_periodic(self):
        c = Codebook(CFG).circular("C", 10)
        assert (c ** 10) @ (c ** 0) > 0.999
        assert (c ** 1) @ (c ** 0) < 0.9

    def test_circular_and_plain_symbol_independent(self):
        cb = Codebook(CFG)
        assert abs(cb.circular("C", 10) @ cb.symbol("C")) < 0.15

    def test_eviction_regenerates_identically(self):
        tiny = Codebook(SubstrateConfig(dim=CFG.dim, codebook_cache_mb=1))
        first = tiny.symbol("A").data.copy()
        for i in range(300):                     # 300 x 8 KB > 1 MB: evicts "A"
            tiny.symbol(f"filler{i}")
        assert np.array_equal(tiny.symbol("A").data, first)

    def test_cached_symbols_are_read_only(self):
        vec = Codebook(CFG).symbol("X")
        with pytest.raises(ValueError):
            vec.data[0] = 0

    def test_power_matches_direct_power(self):
        cb = Codebook(CFG)
        assert np.allclose(cb.power("X", 2.5).data, (cb.symbol("X") ** 2.5).data)


class TestSinkhorn:
    def test_doubly_stochastic(self):
        m = KernelBackend().sinkhorn(RNG.standard_normal((4, 4)) * 3.0, 20)
        assert np.allclose(m.sum(axis=0), 1.0, atol=1e-5)
        assert np.allclose(m.sum(axis=1), 1.0, atol=1e-5)
        assert (m >= 0).all()

    def test_extreme_logits_stay_finite_and_non_expansive(self):
        m = KernelBackend().sinkhorn(np.array([[1e3, -1e3], [-1e3, 1e3]]), 20)
        assert np.isfinite(m).all()
        assert m.sum(axis=0).max() <= 1.0 + 1e-9 and m.sum(axis=1).max() <= 1.0 + 1e-9


class TestQuantization:
    def test_round_trip_error_bound(self):
        assert quantization_error(rand().data) < 1e-4

    def test_codes_are_uint16(self):
        assert encode_angles(rand().data).dtype == np.uint16

    def test_decoded_similarity_preserved(self):
        v = rand()
        assert PhasorVector(decode_angles(encode_angles(v.data))) @ v > 0.9999


class TestBackendGate:
    def test_reference_backend_accepted(self):
        assert set_backend(KernelBackend()).name == "numpy"

    def test_wrong_kernel_rejected_and_not_installed(self):
        class Skewed(KernelBackend):
            name = "skewed"

            def bind(self, a, b):
                return a * b * np.complex64(1.01)

        before = get_backend()
        with pytest.raises(BackendRejected, match="bind"):
            set_backend(Skewed())
        assert get_backend() is before

    def test_crashing_kernel_rejected(self):
        class Broken(KernelBackend):
            name = "broken"

            def similarity(self, a, b):
                raise OSError("device lost")

        with pytest.raises(BackendRejected, match="device lost"):
            set_backend(Broken())


def test_byte_lru_bounds_bytes():
    cache = ByteLRUCache(max_bytes=3 * 800)
    for i in range(5):
        cache.put(i, np.zeros(100, dtype=np.float64))    # 800 bytes each
    assert len(cache) == 3 and cache.nbytes == 2400
    assert cache.get(0) is None and cache.get(4) is not None
