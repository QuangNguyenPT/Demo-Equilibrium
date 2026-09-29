"""Tasks the model learns from and is judged on.

Generated families carry an exact answer and a plain-language description of the rule
(the coda's teaching signal). ARC-AGI JSON and Sudoku-Extreme CSV loaders turn the public
benchmarks into the same Task shape. Every array is a 2-D int64 canvas; sequences and
text are single rows.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Grid = np.ndarray
Rule = Callable[[Grid], Grid]
Sampler = Callable[[np.random.Generator], Grid]

KIND = {"grid": "grid", "arc": "grid", "sudoku": "grid", "sequence": "int", "text": "byte"}
COLOR_NAMES = ("black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "maroon")
WORDS = (
    "apple", "river", "stone", "cloud", "green", "table", "music", "light", "house", "bread",
    "chair", "water", "plant", "night", "sound", "paper", "train", "smile", "dream", "field",
    "horse", "ocean", "piano", "storm", "sugar", "tiger", "voice", "wheel", "zebra", "brick",
    "candy", "dance", "eagle", "frost", "grape", "honey", "judge", "knife", "lemon", "mango",
    "nurse", "olive", "pearl", "queen", "robin", "shelf", "tulip", "unity", "vivid", "world",
    "cat", "dog", "sun", "map", "pen", "cup", "box", "key", "hat", "bus", "fox", "owl",
    "bird", "fish", "tree", "rain", "snow", "fire", "wind", "moon", "star", "road", "door",
    "book", "milk", "salt", "gold", "iron", "rock", "sand", "leaf", "seed", "root", "wing",
)
TEXT_FAMILIES = ("reverse", "upper", "sort", "caesar", "double", "drop_vowels")
GRID_FAMILIES = ("translate", "color_cycle", "recolor", "rotate", "flip", "transpose", "gravity",
                 "translate+color_cycle")
SEQUENCE_FAMILIES = ("offset", "scale", "affine", "reverse", "sort", "rotate", "cumsum", "diff",
                     "scale+reverse")
FAMILIES: Dict[str, Tuple[str, ...]] = {
    "grid": GRID_FAMILIES, "sequence": SEQUENCE_FAMILIES, "text": TEXT_FAMILIES, "sudoku": ("sudoku",),
}
SUDOKU_RULE = "fill every empty cell so each row, column and 3x3 box holds the digits 1 to 9"


@dataclass
class Task:
    """Demonstration pairs, a query and (when known) its answer."""

    domain: str
    family: str
    demos: List[Tuple[Grid, Grid]]
    query: Grid
    answer: Optional[Grid]
    description: str = ""
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return KIND[self.domain]


# ---------- grid ----------

def _grid_sampler(max_side: int, force_color: Optional[int] = None) -> Sampler:
    def sample(rng: np.random.Generator) -> Grid:
        h, w = (int(v) for v in rng.integers(3, max_side + 1, size=2))
        g = np.zeros((h, w), dtype=np.int64)
        for i in range(int(rng.integers(1, 4))):
            bh, bw = int(rng.integers(1, max(1, h // 2) + 1)), int(rng.integers(1, max(1, w // 2) + 1))
            y, x = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
            color = force_color if (force_color is not None and i == 0) else int(rng.integers(1, 10))
            g[y:y + bh, x:x + bw] = color
        return g
    return sample


def _translate(dx: int, dy: int) -> Rule:
    def rule(g: Grid) -> Grid:
        out = np.zeros_like(g)
        h, w = g.shape
        ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
        yd, xd = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
        out[ys, xs] = g[yd, xd]
        return out
    return rule


def _color_cycle(k: int) -> Rule:
    return lambda g: np.where(g > 0, (g - 1 + k) % 9 + 1, 0)


def _gravity(g: Grid) -> Grid:
    out = np.zeros_like(g)
    for x in range(g.shape[1]):
        col = g[:, x][g[:, x] != 0]
        if len(col):
            out[g.shape[0] - len(col):, x] = col
    return out


def _move_words(dx: int, dy: int) -> str:
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'right' if dx > 0 else 'left'}")
    if dy:
        parts.append(f"{abs(dy)} {'down' if dy > 0 else 'up'}")
    return "move every shape " + " and ".join(parts)


def _grid_rule(family: str, rng: np.random.Generator, max_side: int) -> Tuple[Rule, str, Sampler]:
    sample = _grid_sampler(max_side)
    if family in ("translate", "translate+color_cycle"):
        dx, dy = (int(v) for v in rng.choice([v for v in range(-2, 3) if v], 2))
        move, text = _translate(dx, dy), _move_words(dx, dy)
        if family == "translate":
            return move, text, sample
        k = int(rng.integers(1, 9))
        cycle = _color_cycle(k)
        return (lambda g: cycle(move(g))), f"{text}, then shift every color up by {k}", sample
    if family == "color_cycle":
        k = int(rng.integers(1, 9))
        return _color_cycle(k), f"shift every color up by {k}", sample
    if family == "recolor":
        a, b = (int(c) for c in rng.choice(np.arange(1, 10), 2, replace=False))
        return (lambda g: np.where(g == a, b, g)), f"paint every {COLOR_NAMES[a]} cell {COLOR_NAMES[b]}", \
            _grid_sampler(max_side, force_color=a)
    if family == "rotate":
        k = int(rng.integers(1, 4))
        return (lambda g: np.rot90(g, -k).copy()), f"rotate the grid {90 * k} degrees clockwise", sample
    if family == "flip":
        if rng.integers(0, 2):
            return (lambda g: g[:, ::-1].copy()), "mirror the grid left to right", sample
        return (lambda g: g[::-1, :].copy()), "mirror the grid top to bottom", sample
    if family == "transpose":
        return (lambda g: g.T.copy()), "swap rows and columns", sample
    if family == "gravity":
        return _gravity, "let every colored cell fall to the bottom of its column", sample
    raise ValueError(f"unknown grid family '{family}'")


# ---------- sequence ----------

def _sequence_sampler(rng: np.random.Generator) -> Grid:
    n = int(rng.integers(4, 7))
    return rng.choice(np.arange(0, 10), n, replace=False).astype(np.int64)[None, :]


def _sequence_rule(family: str, rng: np.random.Generator) -> Tuple[Rule, str, Sampler]:
    a = int(rng.choice([-3, -2, 2, 3]))
    b = int(rng.choice([v for v in range(-9, 10) if v]))
    k = int(rng.integers(1, 3))
    desc = int(rng.integers(0, 2))
    rules: Dict[str, Tuple[Rule, str]] = {
        "offset": (lambda s: s + b, f"add {b} to every number"),
        "scale": (lambda s: s * a, f"multiply every number by {a}"),
        "affine": (lambda s: s * a + b, f"multiply every number by {a}, then add {b}"),
        "reverse": (lambda s: s[:, ::-1].copy(), "reverse the order of the numbers"),
        "sort": ((lambda s: -np.sort(-s, axis=1)) if desc else (lambda s: np.sort(s, axis=1)),
                 f"sort the numbers in {'descending' if desc else 'ascending'} order"),
        "rotate": (lambda s: np.roll(s, -k, axis=1), f"rotate the numbers left by {k}"),
        "cumsum": (lambda s: np.cumsum(s, axis=1), "replace every number by the running total"),
        "diff": (lambda s: s[:, 1:] - s[:, :-1], "replace each number by how much it grew from the previous one"),
        "scale+reverse": (lambda s: (s * a)[:, ::-1].copy(), f"multiply every number by {a}, then reverse the order"),
    }
    if family not in rules:
        raise ValueError(f"unknown sequence family '{family}'")
    rule, text = rules[family]
    return rule, text, _sequence_sampler


# ---------- text (language tasks over bytes) ----------

def _as_bytes(word: str) -> Grid:
    return np.frombuffer(word.encode("ascii"), dtype=np.uint8).astype(np.int64)[None, :]


def _as_word(row: Grid) -> str:
    return bytes(int(v) for v in row.ravel()).decode("ascii")


TEXT_SOURCES = ("words", "letters")


def random_letters(r: np.random.Generator, lengths: Tuple[int, int] = (3, 5)) -> str:
    """A lowercase string outside WORDS: the input no finite table of training words can cover."""
    vocab = set(WORDS)
    while True:
        w = "".join(chr(97 + int(c)) for c in r.integers(0, 26, int(r.integers(lengths[0], lengths[1] + 1))))
        if w not in vocab:
            return w


def _text_rule(family: str, rng: np.random.Generator, source: str = "words") -> Tuple[Rule, str, Sampler]:
    """`source="words"` draws inputs from the 90-word list (as trained in ADR-006); "letters" draws
    random strings, so a text family can only be solved by its rule, never by recalling a word."""
    if source not in TEXT_SOURCES:
        raise ValueError(f"unknown text source '{source}'")
    k = int(rng.integers(1, 4))
    short = family == "double"
    pool = [w for w in WORDS if len(w) <= 4] if short else list(WORDS)

    def sample(r: np.random.Generator) -> Grid:
        if source == "letters":
            return _as_bytes(random_letters(r, (3, 4) if short else (3, 5)))
        return _as_bytes(pool[int(r.integers(0, len(pool)))])

    words: Dict[str, Tuple[Callable[[str], str], str]] = {
        "reverse": (lambda w: w[::-1], "write the word backwards"),
        "upper": (lambda w: w.upper(), "write the word in capital letters"),
        "sort": (lambda w: "".join(sorted(w)), "sort the letters alphabetically"),
        "caesar": (lambda w: "".join(chr((ord(c) - 97 + k) % 26 + 97) for c in w),
                   f"shift every letter {k} places forward in the alphabet"),
        "double": (lambda w: "".join(c + c for c in w), "write every letter twice"),
        "drop_vowels": (lambda w: "".join(c for c in w if c not in "aeiou"), "remove the vowels"),
    }
    if family not in words:
        raise ValueError(f"unknown text family '{family}'")
    fn, text = words[family]
    return (lambda g: _as_bytes(fn(_as_word(g)))), text, sample


# ---------- sudoku ----------

def count_solutions(puzzle: Grid, limit: int = 2) -> Tuple[int, Optional[Grid]]:
    """Number of completions of a 9x9 puzzle (0 = blank), capped at limit, and the first one."""
    cells = [int(v) for v in puzzle.ravel()]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(cells):
        if v:
            r, c = divmod(i, 9)
            bit, b = 1 << v, (r // 3) * 3 + c // 3
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return 0, None
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    empties = [i for i, v in enumerate(cells) if v == 0]
    found: List[List[int]] = []

    def search() -> None:
        best, best_mask, best_count = -1, 0, 10
        for i in empties:
            if cells[i] == 0:
                r, c = divmod(i, 9)
                mask = ~(rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]) & 0x3FE
                count = bin(mask).count("1")
                if count < best_count:
                    best, best_mask, best_count = i, mask, count
                    if count == 0:
                        return
        if best < 0:
            found.append(list(cells))
            return
        r, c = divmod(best, 9)
        b = (r // 3) * 3 + c // 3
        for v in range(1, 10):
            bit = 1 << v
            if best_mask & bit:
                cells[best] = v
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
                search()
                cells[best] = 0
                rows[r] &= ~bit
                cols[c] &= ~bit
                boxes[b] &= ~bit
                if len(found) >= limit:
                    return

    search()
    first = np.array(found[0], dtype=np.int64).reshape(9, 9) if found else None
    return len(found), first


def _sudoku_solution(rng: np.random.Generator) -> Grid:
    order = [3 * band + r for band in rng.permutation(3) for r in rng.permutation(3)]
    stacks = [3 * s + c for s in rng.permutation(3) for c in rng.permutation(3)]
    digits = rng.permutation(9) + 1
    return np.array([[digits[(3 * (r % 3) + r // 3 + c) % 9] for c in stacks] for r in order], dtype=np.int64)


def sudoku_task(rng: np.random.Generator, clues: Tuple[int, int] = (34, 45)) -> Task:
    """A uniquely solvable puzzle with clues in [lo, hi] (no demos: the rule is fixed)."""
    while True:
        solution = _sudoku_solution(rng)
        puzzle = solution.copy().ravel()
        blanks = 81 - int(rng.integers(clues[0], clues[1] + 1))
        puzzle[rng.permutation(81)[:blanks]] = 0
        puzzle = puzzle.reshape(9, 9)
        if count_solutions(puzzle, limit=2)[0] == 1:
            return Task("sudoku", "sudoku", [], puzzle, solution, SUDOKU_RULE)


# ---------- generation ----------

def generate(domain: str, family: str, rng: np.random.Generator, max_side: int = 10, n_demos: int = 3,
             text_source: str = "words") -> Task:
    """One task of a family with a freshly drawn concrete rule."""
    if domain == "sudoku":
        return sudoku_task(rng)
    if domain == "grid":
        rule, text, sample = _grid_rule(family, rng, max_side)
    elif domain == "sequence":
        rule, text, sample = _sequence_rule(family, rng)
    elif domain == "text":
        rule, text, sample = _text_rule(family, rng, text_source)
    else:
        raise ValueError(f"unknown domain '{domain}'")
    inputs = [sample(rng) for _ in range(n_demos + 1)]
    pairs = [(x, rule(x)) for x in inputs]
    return Task(domain, family, pairs[:-1], pairs[-1][0], pairs[-1][1], text)


def sample_task(rng: np.random.Generator, domains: Dict[str, float], max_side: int = 10) -> Task:
    """Draw a domain by weight, then a family uniformly, then a task."""
    names = sorted(domains)
    weights = np.array([domains[n] for n in names], dtype=np.float64)
    domain = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    families = FAMILIES[domain]
    return generate(domain, families[int(rng.integers(0, len(families)))], rng, max_side)


# ---------- public benchmarks ----------

def load_arc(directory: Path) -> List[Task]:
    """ARC-AGI JSON files -> one Task per test pair (family = puzzle id)."""
    tasks = []
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        demos = [(np.array(p["input"], dtype=np.int64), np.array(p["output"], dtype=np.int64)) for p in data["train"]]
        for i, test in enumerate(data["test"]):
            answer = np.array(test["output"], dtype=np.int64) if "output" in test else None
            tasks.append(Task("arc", path.stem, demos, np.array(test["input"], dtype=np.int64), answer,
                              meta={"test_index": str(i)}))
    return tasks


def load_sudoku_csv(path: Path, limit: int, skip: int = 0) -> List[Task]:
    """Sudoku-Extreme rows (source, question, answer, rating) -> Tasks; '.' or '0' is blank."""
    tasks = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for i, row in enumerate(reader):
            if i < skip:
                continue
            if len(tasks) >= limit:
                break
            question = [0 if ch in ".0" else int(ch) for ch in row[1]]
            answer = [int(ch) for ch in row[2]]
            tasks.append(Task("sudoku", "sudoku_extreme", [], np.array(question, dtype=np.int64).reshape(9, 9),
                              np.array(answer, dtype=np.int64).reshape(9, 9), SUDOKU_RULE, meta={"rating": row[3]}))
    return tasks


def fits(task: Task, max_tokens: int) -> bool:
    """Whether the task's token field stays within max_tokens (see field.encode_task)."""
    grids: Sequence[Grid] = [g for pair in task.demos for g in pair] + [task.query]
    h = max(g.shape[0] for g in grids)
    w = max(g.shape[1] for g in grids)
    return (2 * len(task.demos) + 2) * h * w <= max_tokens
