from typing import Optional

from hrm.config import LanguageConfig
from hrm.language.backends import HashingLanguage, OllamaLanguage, TransformersLanguage
from hrm.language.base import LanguageError, LanguageSection
from hrm.language.speech import InnerSpeech, literal_explanation
from hrm.language.text import parse_task, parse_value, rewrite


def make_language(cfg: LanguageConfig) -> Optional[LanguageSection]:
    """Language section named by the config, or None when the model runs without one."""
    if cfg.backend == "none":
        return None
    if cfg.backend == "hashing":
        return HashingLanguage(cfg.embed_dim)
    if cfg.backend == "transformers":
        return TransformersLanguage(cfg.model)
    return OllamaLanguage(cfg.model, cfg.endpoint)


__all__ = ["HashingLanguage", "InnerSpeech", "LanguageError", "LanguageSection",
           "OllamaLanguage", "TransformersLanguage", "literal_explanation", "make_language",
           "parse_task", "parse_value", "rewrite"]
