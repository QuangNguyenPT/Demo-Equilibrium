from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class TransformerConfig:
    """Blueprint layout for Transformer LLM backbone."""
    num_layers: int = 6
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_kv_heads: Optional[int] = None  # None = MHA, < num_heads = GQA
    intermediate_dim: int = 1536        # SwiGLU hidden dim (approx 8/3 * hidden_dim)
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    # RealFormer-style attention residuals (cross-layer score passthrough)
    use_attn_residuals: bool = False     # pass pre-softmax scores between layers
    attn_residual_block_size: int = 4    # Block AttnRes: layers per block (1 = full AttnRes)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_attention_heads


@dataclass
class HRMConfig:
    """Blueprint layout for Hierarchical Reasoning Module (HRM)."""
    hidden_dim: int = 512
    max_reasoning_steps: int = 8
    low_level_substeps: int = 2
    act_threshold: float = 0.5
    use_act_halting: bool = True
    inject_mode: str = "residual_sum"   # 'residual_sum' or 'linear_projection'
    # PEM-inspired hard-rail conjunctive halt
    use_hard_rails: bool = True          # conjunctive: kinetic+semantic+verify all pass
    eps_kinetic: float = 1e-3            # residual threshold for kinetic convergence
    eps_energy: float = 0.15             # energy threshold for semantic convergence
    tau_verify: float = 0.65             # halt_confidence threshold for verification
    # Annealed restart parameters
    restart_budget: int = 3              # max restarts before ABSTAIN
    sigma0_restart: float = 0.05         # initial perturbation std
    gamma_restart: float = 0.6           # anneal factor per restart
    plateau_cycles: int = 4              # steps without improvement before restart


@dataclass
class EngineConfig:
    """C++ Native execution engine runtime settings."""
    arena_capacity_mb: int = 128
    num_threads: int = 4
    enable_avx2: bool = True
    c_lib_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Unified configuration encapsulating both LLM backbone and HRM Reasoner."""
    vocab_size: int = 32000
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    hrm: HRMConfig = field(default_factory=HRMConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    model_name: str = "HRM-LLM-Hybrid-Base"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        tf_data = data.get("transformer", {})
        hrm_data = data.get("hrm", {})
        eng_data = data.get("engine", {})

        return cls(
            vocab_size=data.get("vocab_size", 32000),
            transformer=TransformerConfig(**tf_data) if tf_data else TransformerConfig(),
            hrm=HRMConfig(**hrm_data) if hrm_data else HRMConfig(),
            engine=EngineConfig(**eng_data) if eng_data else EngineConfig(),
            model_name=data.get("model_name", "HRM-LLM-Hybrid-Base")
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        if not HAS_YAML:
            raise ImportError("PyYAML is not installed. Please run `pip install pyyaml` to export YAML configs.")
        return yaml.dump(self.to_dict(), sort_keys=False)
