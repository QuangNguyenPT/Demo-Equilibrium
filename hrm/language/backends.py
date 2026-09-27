"""Language backends.

HashingLanguage       dependency-free: hashed bag-of-words embeddings; generation is
                      delegated to an optional responder (scripted in tests/offline).
TransformersLanguage  local Hugging Face model; embeddings are mean-pooled final
                      hidden states (needs `pip install torch transformers`).
OllamaLanguage        local Ollama server over HTTP (stdlib only).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Callable, Optional

import numpy as np

from hrm.language.base import LanguageError, LanguageSection, unit

_TOKEN = re.compile(r"[a-z0-9']+")


class HashingLanguage(LanguageSection):
    name = "hashing"

    def __init__(self, dim: int = 256, responder: Optional[Callable[[str], str]] = None):
        self.dim = dim
        self.responder = responder

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return unit(v)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return self.responder(prompt) if self.responder else ""


class TransformersLanguage(LanguageSection):
    name = "transformers"
    has_hidden_states = True

    def __init__(self, model_name: str, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LanguageError("the transformers backend needs "
                                "`pip install torch transformers`") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except OSError as exc:
            raise LanguageError(f"cannot load language model '{model_name}': {exc}") from exc
        self.lm.to(self.device).eval()

    def embed(self, text: str) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(self.device)
        with torch.no_grad():
            hidden = self.lm(**batch, output_hidden_states=True).hidden_states[-1][0]
        return unit(hidden.float().mean(dim=0).cpu().numpy())

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        torch = self._torch
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        batch = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.lm.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, batch["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


class OllamaLanguage(LanguageSection):
    name = "ollama"

    def __init__(self, model: str, endpoint: str = "http://localhost:11434",
                 timeout: float = 120.0):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.endpoint}{path}",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LanguageError(f"ollama at {self.endpoint} failed: {exc}") from exc

    def embed(self, text: str) -> np.ndarray:
        data = self._post("/api/embed", {"model": self.model, "input": text})
        try:
            return unit(np.asarray(data["embeddings"][0], dtype=np.float32))
        except (KeyError, IndexError, TypeError) as exc:
            raise LanguageError(f"unexpected ollama embed reply: {str(data)[:200]}") from exc

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        data = self._post("/api/generate", {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": max_new_tokens}})
        if not isinstance(data.get("response"), str):
            raise LanguageError(f"unexpected ollama generate reply: {str(data)[:200]}")
        return data["response"]
