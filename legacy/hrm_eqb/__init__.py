"""HRM-EQB: hierarchical reasoning by certified equilibrium (spec v0.2, torch)."""

from hrm_eqb.algebra import (
    as_complex, as_real, cleanup, phasor_attention, phasor_codebook, qconj, qmul,
    relation_distance, rope, rotate3, rotate4,
)
from hrm_eqb.attnres import DepthAttention, rms_normalize
from hrm_eqb.certified import CertifiedLinear, cayley, radial_relu, substochastic
from hrm_eqb.config import PRESETS, ModelConfig
from hrm_eqb.curiosity import most_informative, vote_entropy
from hrm_eqb.equilibrium import ContractionBounds, Context, EquilibriumCore, Solution
from hrm_eqb.gate import keep_margin, should_keep
from hrm_eqb.mind import Answer, Mind, Question
from hrm_eqb.model import HRMEQB
from hrm_eqb.neuromodulation import HORMONES, Neuromodulators
from hrm_eqb.product_keys import product_topk
from hrm_eqb.tasks import Task

__all__ = [
    "HRMEQB", "PRESETS", "Answer", "Mind", "ModelConfig", "Question", "Task",
    "HORMONES", "CertifiedLinear", "ContractionBounds", "Context", "DepthAttention",
    "EquilibriumCore", "Neuromodulators", "Solution", "as_complex", "as_real", "cayley",
    "cleanup", "keep_margin", "most_informative", "phasor_attention", "phasor_codebook",
    "product_topk", "qconj", "qmul", "radial_relu", "relation_distance", "rms_normalize",
    "rope", "rotate3", "rotate4", "should_keep", "substochastic", "vote_entropy",
]
