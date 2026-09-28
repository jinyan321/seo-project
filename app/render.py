"""Render AI answers (untrusted LLM + web text) as safe HTML.

Raw HTML in the answer is escaped, never rendered. markdown-it's validateLink drops
javascript:/vbscript:/data: links. Every link opens in a new tab with no referrer.
"""

from markdown_it import MarkdownIt
from markupsafe import Markup

_md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def _link_open(renderer, tokens, idx, options, env):
    tok = tokens[idx]
    tok.attrSet("rel", "nofollow noopener noreferrer")
    tok.attrSet("target", "_blank")
    return renderer.renderToken(tokens, idx, options, env)


_md.add_render_rule("link_open", _link_open)


def render_answer(text: str | None) -> Markup:
    return Markup(_md.render(text or ""))  # noqa: S704 - output of the escaping renderer only
