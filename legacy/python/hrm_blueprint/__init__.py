"""
HRM+LLM Hybrid Architecture: Python Blueprint Layer

Python acts as the high-level declarative blueprint and architecture layout,
while computational kernels and execution run inside the native C++ engine.
"""

from .configs import ModelConfig, HRMConfig, TransformerConfig, EngineConfig
from .blueprint import LayerLayout, ModelBlueprint
from .transformer import TransformerBackboneBlueprint
from .hrm_controller import HRMControllerBlueprint
from .hybrid_model import HybridHRMLLMBlueprint
from .bridge import EngineBridge

__all__ = [
    "ModelConfig",
    "HRMConfig",
    "TransformerConfig",
    "EngineConfig",
    "LayerLayout",
    "ModelBlueprint",
    "TransformerBackboneBlueprint",
    "HRMControllerBlueprint",
    "HybridHRMLLMBlueprint",
    "EngineBridge",
]
