"""Stories: sentences -> a situation that changes over time -> answers derived, not guessed.

Perception (learned): a small transformer tags each token with a role (owner, source,
destination, amount, item) and the sentence with an event schema. Names and numbers are never
in its vocabulary; they are copied from the tokens by role, so unseen names and magnitudes
cost nothing. Surface form ("gives B n", "gives n to B", "receives n from A") is convention,
which is why this part is learned.

Situation (exact): quantities are workspace terms qty(t, entity, item). A state sentence sets
one at time t; an event is a transition t -> t+1 whose effects are equalities (sub/add) and
whose frame axiom carries every unaffected quantity forward. Every equality is tagged with the
sentence it came from, so a negative quantity is a contradiction that names its sentences.
Verification replays the same reading with a plain dictionary simulation and checks
conservation of items under transfers: a different code path and a different representation.
"""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from hrm_x.mind import Hyp, Mind, Result
from hrm_x.rules import Rule, saturate
from hrm_x.workspace import EGraphAPI as Workspace, Why, make_workspace

ROLES = ("O", "OWNER", "SRC", "DST", "AMT", "ITEM")
SCHEMAS = ("STATE", "TRANSFER", "ACQUIRE", "CONSUME", "Q_ONE", "Q_SUM")
TRAIN_NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy",
               "Mallory", "Niaj", "Olivia", "Peggy", "Rupert", "Sybil", "Trent", "Victor", "Walter", "Yolanda"]
TRAIN_ITEMS = ["apples", "pears", "marbles", "books", "coins", "stickers"]
# (schema, template). {A} is the subject; roles are read from the template's slot names.
TEMPLATES: List[Tuple[str, str]] = [
    ("STATE", "{A:OWNER} has {n:AMT} {i:ITEM} ."),
    ("STATE", "{A:OWNER} owns {n:AMT} {i:ITEM} ."),
    ("TRANSFER", "{A:SRC} gives {B:DST} {n:AMT} {i:ITEM} ."),
    ("TRANSFER", "{A:SRC} gives {n:AMT} {i:ITEM} to {B:DST} ."),
    ("TRANSFER", "{B:DST} receives {n:AMT} {i:ITEM} from {A:SRC} ."),
    ("TRANSFER", "{B:DST} takes {n:AMT} {i:ITEM} from {A:SRC} ."),
    ("ACQUIRE", "{A:OWNER} buys {n:AMT} {i:ITEM} ."),
    ("ACQUIRE", "{A:OWNER} finds {n:AMT} {i:ITEM} ."),
    ("CONSUME", "{A:OWNER} eats {n:AMT} {i:ITEM} ."),
    ("CONSUME", "{A:OWNER} loses {n:AMT} {i:ITEM} ."),
    ("Q_ONE", "how many {i:ITEM} does {A:OWNER} have ?"),
    ("Q_SUM", "how many {i:ITEM} do {A:OWNER} and {B:OWNER} have together ?"),
]
SLOT = re.compile(r"\{(\w):(\w+)\}")


def fill(template: str, names: Dict[str, str], n: int, item: str) -> Tuple[List[str], List[str]]:
    """Template -> (tokens, role tags)."""
    tokens, tags = [], []
    for word in template.split():
        m = SLOT.fullmatch(word)
        if m is None:
            tokens.append(word)
            tags.append("O")
            continue
        slot, role = m.groups()
        tokens.append(str(n) if slot == "n" else item if slot == "i" else names[slot])
        tags.append(role)
    return tokens, tags


def tokenize(sentence: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+|[?.]", sentence)


class Vocab:
    """Function words and verbs only; names -> <cap>, numbers -> <num>, other words -> <unk>."""

    def __init__(self) -> None:
        words = sorted({w for _, t in TEMPLATES for w in t.split() if not SLOT.fullmatch(w)} | set(TRAIN_ITEMS))
        self.itos = ["<pad>", "<cap>", "<num>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, token: str, first: bool) -> int:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return self.stoi["<num>"]
        low = token.lower()
        if low in self.stoi and not (token[0].isupper() and not first):
            return self.stoi[low]
        return self.stoi["<cap>"] if token[0].isupper() else self.stoi["<unk>"]


class Tagger(nn.Module):
    """2-layer transformer encoder: per-token role logits and a sentence schema logit (pooled)."""

    def __init__(self, vocab: int, dim: int = 48, heads: int = 4, layers: int = 2, max_len: int = 20) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.roles = nn.Linear(dim, len(ROLES))
        self.schema = nn.Linear(dim, len(SCHEMAS))

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = ids == 0
        h = self.encoder(self.embed(ids) + self.pos(torch.arange(ids.shape[1])), src_key_padding_mask=pad)
        pooled = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True)
        return self.roles(h), self.schema(pooled)


def training_sentences(count: int, rng: random.Random, unk_items: float = 0.3) -> List[Tuple[List[str], List[str], str]]:
    out = []
    for _ in range(count):
        schema, template = rng.choice(TEMPLATES)
        a, b = rng.sample(TRAIN_NAMES, 2)
        item = rng.choice(TRAIN_ITEMS)
        tokens, tags = fill(template, {"A": a, "B": b}, rng.randint(1, 20), item)
        if rng.random() < unk_items:
            tokens = ["zzitem" if t == item else t for t in tokens]   # forces ITEM to be read from position
        tokens[0] = tokens[0][0].upper() + tokens[0][1:]
        out.append((tokens, tags, schema))
    return out


@dataclass
class Parser:
    """The learned perception: trains in seconds on CPU (all CUDA work would go through select_device)."""

    vocab: Vocab = field(default_factory=Vocab)
    model: Optional[Tagger] = None
    params: int = 0

    def train(self, sentences: int = 4000, epochs: int = 12, seed: int = 0) -> float:
        rng = random.Random(seed)
        torch.manual_seed(seed)
        data = training_sentences(sentences, rng)
        self.model = Tagger(len(self.vocab.itos))
        self.params = sum(p.numel() for p in self.model.parameters())
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        ids, roles, schemas = self._batch(data)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        for _ in range(epochs):
            perm = torch.randperm(len(data))
            for start in range(0, len(data), 64):
                idx = perm[start:start + 64]
                role_logits, schema_logits = self.model(ids[idx])
                loss = loss_fn(role_logits.reshape(-1, len(ROLES)), roles[idx].reshape(-1)) + \
                    loss_fn(schema_logits, schemas[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return loss.item()

    def save(self, path: str) -> None:
        """Learned perception weights (.pt): the only gradient-trained parameters in the mind."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "Parser":
        self.model = Tagger(len(self.vocab.itos))
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.params = sum(p.numel() for p in self.model.parameters())
        return self

    def _batch(self, data):
        width = max(len(t) for t, _, _ in data)
        ids = torch.zeros(len(data), width, dtype=torch.long)
        roles = torch.full((len(data), width), -1, dtype=torch.long)
        schemas = torch.tensor([SCHEMAS.index(s) for _, _, s in data])
        for i, (tokens, tags, _) in enumerate(data):
            for j, (tok, tag) in enumerate(zip(tokens, tags)):
                ids[i, j] = self.vocab.encode(tok, j == 0)
                roles[i, j] = ROLES.index(tag)
        return ids, roles, schemas

    @torch.no_grad()
    def read(self, sentence: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        tokens = tokenize(sentence)
        ids = torch.tensor([[self.vocab.encode(t, j == 0) for j, t in enumerate(tokens)]])
        role_logits, schema_logits = self.model(ids)
        return tokens, role_logits[0].softmax(-1), schema_logits[0].softmax(-1)


# ---------- readings (the hypotheses) ----------

@dataclass(frozen=True)
class Frame:
    schema: str
    owner: Tuple[str, ...] = ()
    src: Optional[str] = None
    dst: Optional[str] = None
    amount: Optional[Fraction] = None
    item: Optional[str] = None

    def show(self) -> str:
        if self.schema == "TRANSFER":
            return f"transfer({self.src}->{self.dst}, {self.amount} {self.item})"
        if self.schema in ("Q_ONE", "Q_SUM"):
            return f"ask({'+'.join(self.owner)}, {self.item})"
        return f"{self.schema.lower()}({','.join(self.owner)}, {self.amount} {self.item})"


def singular(word: str) -> str:
    return word.lower()[:-1] if word.lower().endswith("s") else word.lower()


def frames_for(parser: Parser, sentence: str, keep: float = 0.02) -> List[Tuple[Frame, float]]:
    """Readings of one sentence with their probabilities (schema alternatives x role swaps)."""
    tokens, roles, schema = parser.read(sentence)
    tag = roles.argmax(-1).tolist()

    def pick(role: str) -> List[str]:
        return [tokens[j] for j, r in enumerate(tag) if ROLES[r] == role]

    def conf(role: str) -> float:
        js = [j for j, r in enumerate(tag) if ROLES[r] == role]
        return min((float(roles[j, ROLES.index(role)]) for j in js), default=1.0)

    amounts = [Fraction(t) for t in pick("AMT") if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    items = [singular(t) for t in pick("ITEM")]
    amount = amounts[0] if amounts else None
    item = items[0] if items else None
    out = []
    for s in schema.argsort(descending=True)[:2].tolist():
        p = float(schema[s])
        name = SCHEMAS[s]
        if p < keep:
            continue
        if name == "TRANSFER":
            src = [t.lower() for t in pick("SRC")] or [None]
            dst = [t.lower() for t in pick("DST")] or [None]
            c = min(conf("SRC"), conf("DST"))
            out.append((Frame(name, (), src[0], dst[0], amount, item), p * c))
            if c < 0.99:
                out.append((Frame(name, (), dst[0], src[0], amount, item), p * (1 - c)))
        else:
            owners = tuple(t.lower() for t in pick("OWNER"))
            out.append((Frame(name, owners, None, None, amount, item), p * conf("OWNER")))
    return out


@dataclass
class StoryTask:
    sentences: List[str]
    question: str


@dataclass
class Reading:
    frames: Tuple[Frame, ...]
    question: Frame


def _fold(op: str) -> Rule:
    def build(ws: Workspace, s: Dict[str, int]) -> int:
        a, b = ws.value(s["?a"]), ws.value(s["?b"])
        return ws.const("Q", a + b if op == "add" else a - b)
    return Rule(f"exact:{op}", (op, "?a", "?b"), build,
                lambda ws, s: ws.value(s["?a"]) is not None and ws.value(s["?b"]) is not None)


FOLD = [_fold("add"), _fold("sub")]


def simulate(reading: Reading) -> Tuple[Optional[Fraction], List[str]]:
    """Independent check: plain-dictionary replay plus invariants. Returns (answer, violations)."""
    state: Dict[Tuple[str, str], Fraction] = {}
    problems: List[str] = []
    external = Fraction(0)
    for k, f in enumerate(reading.frames):
        if f.schema == "STATE":
            state[(f.owner[0], f.item)] = f.amount
            continue
        before = sum(v for (e, i), v in state.items() if i == f.item)
        if f.schema == "TRANSFER":
            for who, sign in ((f.src, -1), (f.dst, 1)):
                if (who, f.item) not in state:
                    return None, problems + [f"s{k}: {who}'s {f.item} unknown"]
                state[(who, f.item)] += sign * f.amount
        else:
            who, sign = f.owner[0], 1 if f.schema == "ACQUIRE" else -1
            if (who, f.item) not in state:
                return None, problems + [f"s{k}: {who}'s {f.item} unknown"]
            state[(who, f.item)] += sign * f.amount
            external += sign * f.amount
        after = sum(v for (e, i), v in state.items() if i == f.item)
        expected = before + (0 if f.schema == "TRANSFER" else (f.amount if f.schema == "ACQUIRE" else -f.amount))
        if after != expected:
            problems.append(f"s{k}: conservation violated")
        problems += [f"s{k}: {e}'s {i} negative" for (e, i), v in state.items() if v < 0]
    q = reading.question
    try:
        total = sum(state[(o, q.item)] for o in q.owner)
    except KeyError:
        return None, problems + ["question asks about an unknown quantity"]
    return total, problems


class Story:
    name = "story"

    def __init__(self, parser: Parser, max_readings: int = 6) -> None:
        self.parser = parser
        self.max_readings = max_readings

    def propose_cost(self, task: StoryTask, level: int) -> float:
        return 20.0 * (len(task.sentences) + 1)

    def propose(self, task: StoryTask, level: int, mind: Mind) -> Optional[Tuple[List[Hyp], int]]:
        if level > 0:
            return None
        per_sentence = [frames_for(self.parser, s) for s in task.sentences]
        questions = frames_for(self.parser, task.question)
        readings = []
        for combo in itertools.product(*per_sentence, questions):
            p = 1.0
            for _, w in combo:
                p *= w
            frames = tuple(f for f, _ in combo[:-1])
            q = combo[-1][0]
            if q.schema not in ("Q_ONE", "Q_SUM") or any(f.schema in ("Q_ONE", "Q_SUM") for f in frames):
                continue
            readings.append((p, Reading(frames, q)))
        readings.sort(key=lambda r: -r[0])
        hyps = [Hyp("; ".join(f.show() for f in r.frames) + " | " + r.question.show(), p, r)
                for p, r in readings[: self.max_readings]]
        return hyps, len(task.sentences) + 1

    def derive_cost(self, task: StoryTask, hyp: Hyp, mind: Mind) -> float:
        return 10.0 * len(hyp.payload.frames) ** 2

    def derive(self, task: StoryTask, hyp: Hyp, mind: Mind) -> int:
        reading: Reading = hyp.payload
        ws = make_workspace()
        pairs = sorted({(o, f.item) for f in reading.frames + (reading.question,)
                        for o in (f.owner + tuple(x for x in (f.src, f.dst) if x))})
        ent = {e: ws.const("E", e) for e, _ in pairs}
        itm = {i: ws.const("I", i) for _, i in pairs}

        def qty(t: int, e: str, i: str) -> int:
            return ws.add("qty", [ws.const("T", t), ent[e], itm[i]])

        t = 0
        for k, f in enumerate(reading.frames):
            label = frozenset({f"s{k}"})
            if f.schema == "STATE":
                ws.union(qty(t, f.owner[0], f.item), ws.const("Q", f.amount), Why("given", label))
                continue
            deltas = {(f.src, f.item): "sub", (f.dst, f.item): "add"} if f.schema == "TRANSFER" else \
                {(f.owner[0], f.item): "add" if f.schema == "ACQUIRE" else "sub"}
            for (e, i) in pairs:
                op = deltas.get((e, i))
                new = ws.add(op, [qty(t, e, i), ws.const("Q", f.amount)]) if op else qty(t, e, i)
                ws.union(qty(t + 1, e, i), new, Why("event" if op else "frame", label))
            t += 1
        q = reading.question
        goal = qty(t, q.owner[0], q.item)
        for other in q.owner[1:]:
            goal = ws.add("add", [goal, qty(t, other, q.item)])
        saturate(ws, FOLD, max_firings=5000)
        negative = [c for c in ws.classes() if (v := ws.value(c)) is not None and isinstance(v, Fraction) and v < 0]
        blame = set()
        for c in negative:
            for tt in range(t + 1):
                for e, i in pairs:
                    node = ws.lookup("qty", [ws.const("T", tt), ent[e], itm[i]])
                    if node is not None and ws.same(node, c):
                        blame |= ws.assumptions(node, ws.const_node(c))
        hyp.derived = True
        if negative:
            hyp.weight = 0.0
            hyp.culprits = frozenset(blame)
            hyp.note = f"contradiction: a quantity becomes negative (blame {', '.join(sorted(blame))})"
            hyp.answer = None
        else:
            hyp.answer = ws.value(goal)
            if hyp.answer is None:
                hyp.note = "unknown: the story does not fix the quantity asked about"
        return ws.ops

    def verify(self, task: StoryTask, hyp: Hyp, mind: Mind) -> int:
        answer, problems = simulate(hyp.payload)
        hyp.verified = not problems and answer == hyp.answer
        if not hyp.verified:
            hyp.note = "; ".join(problems) or f"replay gives {answer}, derivation gave {hyp.answer}"
        return 5 * len(hyp.payload.frames)

    def questions(self, task, hyps, mind):
        return []

    def ask(self, task, question, hyps, mind) -> int:
        return 0

    def remember(self, task: StoryTask, result: Result, mind: Mind) -> None:
        mind.knowledge.setdefault(self.name, {"stories": 0})["stories"] += 1
