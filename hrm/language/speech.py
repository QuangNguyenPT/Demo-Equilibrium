"""Inner speech: the model thinking in language while it reasons.

When recognition fails, the reasoning section describes the demonstrations in
words and asks the language section which rule could explain them, phrased in
the domain's program language. Every line of the reply is parsed as a program;
anything malformed is dropped. Proposals are hypotheses only: the verifier
still decides, so a fluent but wrong proposal cannot become an answer.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from hrm.language.base import LanguageError, LanguageSection
from hrm.perception.base import Domain, Example, Program

log = logging.getLogger("hrm.language")

_BULLETS = "-*•0123456789.) \t"


class InnerSpeech:
    def __init__(self, language: LanguageSection, max_proposals: int = 3,
                 max_new_tokens: int = 64):
        self.language = language
        self.max_proposals = max_proposals
        self.max_new_tokens = max_new_tokens

    def prompt(self, domain: Domain, demos: Sequence[Example]) -> str:
        """Instruction + one worked example of the exact reply format. The example's
        output is computed by running its rule, so it can never be wrong."""
        example_in, example_rule = domain.format_example
        example_in = domain.from_json(example_in)
        example_out = domain.parse_program(example_rule).apply(example_in)
        shown = "\n".join(f"input: {domain.render(i)}  ->  output: {domain.render(o)}"
                          for i, o in demos)
        return (f"Find the rule that turns each input into its output.\n"
                f"Allowed operations: {domain.vocabulary()}.\n"
                f"Combine operations with ' . ' (applied left to right).\n"
                f"Reply with up to {self.max_proposals} rules, one per line, no explanation.\n\n"
                f"Example\ninput: {domain.render(example_in)}  ->  output: "
                f"{domain.render(example_out)}\nrules:\n{example_rule}\n\n"
                f"Task\n{shown}\nrules:\n")

    def propose(self, domain: Domain, demos: Sequence[Example]) -> Tuple[List[Program], str]:
        """(parsed proposals, raw reply). A failing backend yields no proposals."""
        try:
            said = self.language.generate(self.prompt(domain, demos), self.max_new_tokens)
        except LanguageError as exc:
            log.warning("[LANGUAGE] inner speech unavailable: %s", exc)
            return [], ""
        programs: List[Program] = []
        for line in said.splitlines():
            line = line.strip().strip("`").lstrip(_BULLETS).strip()
            if not line:
                continue
            try:
                programs.append(domain.parse_program(line))
            except ValueError:
                continue
            if len(programs) >= self.max_proposals:
                break
        return programs, said

    def explain(self, program: Program, n_demos: int, answer_text: str) -> str:
        """One sentence in plain language, followed by the literal statement."""
        literal = literal_explanation(program, n_demos, answer_text)
        try:
            said = self.language.generate(
                f"In one short sentence, explain what this rule does: {program.describe()}",
                self.max_new_tokens).strip()
        except LanguageError:
            return literal
        return f"{said} ({literal})" if said else literal


def literal_explanation(program: Program, n_demos: int, answer_text: str) -> str:
    return (f"The rule {program.describe()} reproduces all {n_demos} examples; "
            f"applied to the question it gives {answer_text}.")
