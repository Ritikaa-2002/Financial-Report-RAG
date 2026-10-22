"""Generate a cited answer from retrieved passages with Claude.

Needs the ANTHROPIC_API_KEY environment variable. Without it, ask.py simply shows
the retrieved passages and their page numbers.
"""

import os

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a financial analyst answering questions about a company filing.
Answer ONLY from the numbered passages provided. After every fact or number, cite the
page it came from like [p. 42]. If you calculate something, show the inputs and the
formula in one line. If the passages do not contain the answer, say exactly:
"I couldn't find this in the retrieved pages." Do not use outside knowledge."""


def build_prompt(question, passages):
    blocks = [
        f"[Passage {i} | {c.doc} | page {c.page + 1}]\n{c.text}"
        for i, (c, _score) in enumerate(passages, 1)
    ]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def generate_answer(question, passages, model=DEFAULT_MODEL, max_tokens=600):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
