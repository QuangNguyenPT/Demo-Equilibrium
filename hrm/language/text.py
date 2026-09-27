"""Task notation for text: how a task written in words becomes structure.

    1 2 3 -> 3 2 1 ; 4 5 -> 5 4 ; 7 8 9 -> ?
    0 1 / 1 0 -> 1 0 / 0 1 ; 2 0 / 0 2 -> ?          (grid rows separated by '/')

Segments are separated by ';' or new lines; `?` on the right marks the
question. Parsing is deterministic and strict. Free language goes through the
language section first, which is asked to rewrite it into this notation; its
reply is untrusted and must parse here, or it is ignored.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import numpy as np

from hrm.language.base import LanguageError, LanguageSection

_NUMBERS = re.compile(r"^-?\d+(\s+-?\d+)*$")

ParsedTask = Tuple[List[Tuple[Any, Any]], Optional[Any]]


def parse_task(text: str) -> Optional[ParsedTask]:
    """(demos, query) from the notation, or None if the text is not in it."""
    demos: List[Tuple[Any, Any]] = []
    query = None
    for segment in re.split(r"[;\n]", text):
        if not segment.strip():
            continue
        lhs, arrow, rhs = segment.partition("->")
        if not arrow:
            return None
        inp = _value(lhs)
        if inp is None:
            return None
        if rhs.strip() == "?":
            if query is not None:
                return None
            query = inp
            continue
        out = _value(rhs)
        if out is None:
            return None
        demos.append((inp, out))
    return (demos, query) if demos else None


def rewrite(language: LanguageSection, text: str, max_new_tokens: int = 128) -> Optional[ParsedTask]:
    """Ask the language section to restate free text in the notation, then parse strictly."""
    prompt = ("Rewrite the task below as examples in this exact format and output nothing else:\n"
              "1 2 3 -> 3 2 1 ; 4 5 -> 5 4 ; 7 8 9 -> ?\n"
              "Use numbers only, '->' between input and output, ';' between examples, "
              "and '?' as the output of the question.\n\nTask: " + text)
    try:
        said = language.generate(prompt, max_new_tokens)
    except LanguageError:
        return None
    for line in said.splitlines():
        parsed = parse_task(line.strip())
        if parsed is not None:
            return parsed
    return None


def parse_value(text: str) -> Optional[Any]:
    """One input/output in the notation: '1 2 3' or grid rows '1 0 / 0 1'."""
    return _value(text)


def _value(text: str) -> Optional[Any]:
    rows = [r.strip() for r in text.strip().split("/")]
    if not rows or not all(_NUMBERS.match(r) for r in rows):
        return None
    values = [[int(v) for v in r.split()] for r in rows]
    if len(values) == 1:
        return values[0]
    if len({len(r) for r in values}) != 1:
        return None
    return np.asarray(values, dtype=np.int64)
