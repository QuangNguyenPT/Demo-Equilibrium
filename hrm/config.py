"""Model configuration: the single source of truth for every tunable.

Each model section owns one dataclass; `ModelConfig` nests them. Loading is
strict: a saved config with a missing, unknown or mistyped field is rejected
instead of silently defaulted, because a model running under a
half-understood config is architecturally wrong, not merely slower.
"""
from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


class ConfigError(ValueError):
    """Config is malformed, incomplete, or out of range."""


@dataclass
class SubstrateConfig:
    """FHRR phasor substrate."""
    dim: int = 4096                 # complex dimensions per symbol
    seed: int = 1337                # same seed -> same symbol for every name
    dtype: str = "complex64"
    codebook_cache_mb: int = 256    # byte bound on cached symbols and powers

    def validate(self) -> None:
        if self.dim < 64 or self.dim & (self.dim - 1):
            raise ConfigError("substrate.dim must be a power of two >= 64")
        if self.seed < 0:
            raise ConfigError("substrate.seed must be >= 0")
        if self.dtype not in ("complex64", "complex128"):
            raise ConfigError("substrate.dtype must be complex64 or complex128")
        if self.codebook_cache_mb < 1:
            raise ConfigError("substrate.codebook_cache_mb must be >= 1")


@dataclass
class KnowledgeConfig:
    """SSD-resident knowledge store (keys on disk, streamed in chunks)."""
    max_disk_gb: float = 20.0       # hard cap on key storage; eviction beyond it
    initial_rows: int = 1024        # key-file capacity on creation (doubles on demand)
    retrieval_k: int = 8
    retrieval_chunk: int = 2048     # rows per streamed block: RAM temp = chunk * dim * 8 bytes
    beta_conf: float = 0.3          # retrieval score bonus per unit of confidence
    beta_cx: float = 0.1            # retrieval score penalty per 10 units of complexity
    dup_sim: float = 0.95           # same-signature items above this key similarity merge
    evict_fraction: float = 0.05    # coldest share evicted when the disk cap is reached

    def validate(self) -> None:
        if self.max_disk_gb <= 0:
            raise ConfigError("knowledge.max_disk_gb must be > 0")
        if self.initial_rows < 1 or self.retrieval_k < 1 or self.retrieval_chunk < 1:
            raise ConfigError("knowledge row/retrieval counts must be >= 1")
        if not 0.0 < self.dup_sim <= 1.0:
            raise ConfigError("knowledge.dup_sim must be in (0, 1]")
        if not 0.0 < self.evict_fraction < 1.0:
            raise ConfigError("knowledge.evict_fraction must be in (0, 1)")


@dataclass
class PerceptionConfig:
    """Domain vocabularies (bounds on the enumerated base hypotheses)."""
    grid_max_shift: int = 8          # translations enumerated in [-s, s]
    grid_max_objects: int = 16       # segmentation cap per scene
    seq_max_scale: int = 3           # affine a in [-m, m] \ {0}
    seq_max_offset: int = 9          # affine b in [-m, m]
    vec_max_scale: int = 8           # lattice step used when encoding a 3-vector
    vec_rotations: Tuple[str, ...] = ("x", "y", "z", "d")   # axes enumerated

    def validate(self) -> None:
        if min(self.grid_max_shift, self.grid_max_objects,
               self.seq_max_scale, self.seq_max_offset, self.vec_max_scale) < 1:
            raise ConfigError("perception bounds must be >= 1")
        unknown = sorted(set(self.vec_rotations) - {"x", "y", "z", "d"})
        if unknown:
            raise ConfigError(f"perception.vec_rotations has unknown axis/axes {unknown}")


@dataclass
class VerifyConfig:
    """The independent judge: exact replay + anti-memorization + MDL."""
    max_program_complexity: int = 12  # hard cap; relative cap is input size / 4
    alpha_mdl: float = 0.1            # MDL = accuracy - alpha * complexity

    def validate(self) -> None:
        if self.max_program_complexity < 1:
            raise ConfigError("verify.max_program_complexity must be >= 1")
        if self.alpha_mdl < 0:
            raise ConfigError("verify.alpha_mdl must be >= 0")


@dataclass
class ReasoningConfig:
    """Two-level reasoning: fast L-level resonator ticks inside deep H-level cycles."""
    n_streams: int = 4
    lam_min: float = 0.2              # KM averaging bounds: convergence needs lam
    lam_max: float = 0.8              # strictly inside (0, 1)
    ticks_per_cycle: int = 8          # L-ticks per H-cycle
    max_ticks: int = 2000             # hard engineering cap, not the budget
    sinkhorn_iters: int = 20
    fast_top_k: int = 8               # fast path: candidates verified without iterating
    refresh_top_k: int = 6            # H-refresh: candidates whose energy is measured
    eps_kinetic: float = 0.05         # per-element RMS state change counted as "stopped"
    eps_energy: float = 0.08          # semantic energy counted as "explained"
    tau_verify: float = 0.98          # worst-demo similarity required to accept
    plateau_cycles: int = 5
    restart_sigma0: float = 0.3       # restart kick (radians), annealed by gamma per restart
    restart_gamma: float = 0.7
    restart_budget: int = 8
    memory_prior: float = 0.3         # prior per unit confidence for retrieved knowledge
    synthesis_beam: int = 8
    synthesis_depth: int = 3

    def validate(self) -> None:
        if not 0.0 < self.lam_min < self.lam_max < 1.0:
            raise ConfigError("KM convergence requires 0 < lam_min < lam_max < 1")
        if min(self.n_streams, self.ticks_per_cycle, self.max_ticks, self.sinkhorn_iters,
               self.fast_top_k, self.refresh_top_k, self.plateau_cycles,
               self.synthesis_beam, self.synthesis_depth) < 1:
            raise ConfigError("reasoning counts must be >= 1")
        if self.restart_budget < 0:
            raise ConfigError("reasoning.restart_budget must be >= 0")
        if self.eps_kinetic <= 0 or self.eps_energy <= 0 or not 0 < self.tau_verify <= 1:
            raise ConfigError("reasoning thresholds out of range")
        if self.restart_sigma0 <= 0 or not 0 < self.restart_gamma <= 1:
            raise ConfigError("restart sigma0 must be > 0 and gamma in (0, 1]")


@dataclass
class LearningConfig:
    """Sleep cycle: consolidation behind the frozen benchmark gate."""
    min_successes: int = 2            # verified uses before knowledge is consolidated
    consolidate_amount: float = 0.34  # prior raised per consolidation
    regression_tol: float = 0.0       # per-family solve-rate drop the gate tolerates
    promotion: str = "auto"           # auto: a passing core goes live; staged: waits for a human

    def validate(self) -> None:
        if self.min_successes < 1:
            raise ConfigError("learning.min_successes must be >= 1")
        if not 0.0 < self.consolidate_amount <= 1.0:
            raise ConfigError("learning.consolidate_amount must be in (0, 1]")
        if not 0.0 <= self.regression_tol < 1.0:
            raise ConfigError("learning.regression_tol must be in [0, 1)")
        if self.promotion not in ("auto", "staged"):
            raise ConfigError("learning.promotion must be 'auto' or 'staged'")


@dataclass
class MetaConfig:
    """Metacognition: intent inference, the competence map, curiosity."""
    intent_threshold: float = 0.6     # below this intent probability the model asks
    intent_lr: float = 0.1            # regression step per human-confirmed intent
    min_attempts: int = 3             # attempts before a region is judged
    mastery: float = 0.9              # competence counted as mastered
    patience: int = 8                 # attempts without progress before "stuck"
    explore: float = 0.1              # chance of a random study choice

    def validate(self) -> None:
        if not 0.0 < self.intent_threshold < 1.0 or not 0.0 < self.mastery <= 1.0:
            raise ConfigError("meta thresholds must be in (0, 1)")
        if self.intent_lr <= 0 or self.min_attempts < 1 or self.patience < 1:
            raise ConfigError("meta rates and counts must be positive")
        if not 0.0 <= self.explore < 1.0:
            raise ConfigError("meta.explore must be in [0, 1)")


@dataclass
class LanguageConfig:
    """The native language section."""
    backend: str = "none"             # none | hashing | transformers | ollama
    model: str = ""                   # transformers: HF repo id; ollama: model tag
    endpoint: str = "http://localhost:11434"
    embed_dim: int = 256              # hashing backend dimension
    max_proposals: int = 3            # inner-speech hypotheses per task
    language_prior: float = 0.5       # prior for programs proposed in language
    max_new_tokens: int = 64

    def validate(self) -> None:
        if self.backend not in ("none", "hashing", "transformers", "ollama"):
            raise ConfigError("language.backend must be none, hashing, transformers or ollama")
        if self.backend in ("transformers", "ollama") and not self.model:
            raise ConfigError(f"language.model is required for the {self.backend} backend")
        if min(self.embed_dim, self.max_proposals, self.max_new_tokens) < 1:
            raise ConfigError("language sizes must be >= 1")
        if self.language_prior < 0:
            raise ConfigError("language.language_prior must be >= 0")


@dataclass
class ModelConfig:
    substrate: SubstrateConfig = field(default_factory=SubstrateConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)

    def validate(self) -> "ModelConfig":
        for f in fields(self):
            getattr(self, f.name).validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return _build(cls, data, "config").validate()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
        return cls.from_dict(data)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"{path}: unknown field(s) {unknown}")
    kwargs = {}
    for name in names:
        if name not in data:
            raise ConfigError(f"{path}.{name}: missing")
        kwargs[name] = _coerce(hints[name], data[name], f"{path}.{name}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    if is_dataclass(tp):
        return _build(tp, value, path)
    origin = typing.get_origin(tp)
    if origin is tuple:                       # homogeneous tuple of scalars, e.g. axis names
        args = typing.get_args(tp)
        variadic = len(args) == 2 and args[1] is Ellipsis
        if (not variadic and len(args) != 1) or not isinstance(value, list):
            raise ConfigError(f"{path}: expected a JSON array of {tp}")
        item = args[0]
        return tuple(_coerce(item, v, f"{path}[{i}]") for i, v in enumerate(value))
    # bool is a subclass of int: reject it wherever a number is expected
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    else:
        raise ConfigError(f"{path}: unsupported field type {tp!r}")
    raise ConfigError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
