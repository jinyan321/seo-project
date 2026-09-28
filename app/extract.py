"""Find brand mentions and URLs in an answer.

Bump EXTRACTOR_VERSION whenever these rules change, then run `python -m app.cli reextract`.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract

from app.config import Brand

EXTRACTOR_VERSION = 2  # 2: per-brand case_sensitive matching
SNIPPET_MAX = 300

# Bundled public-suffix snapshot only: never fetch the list over the network.
_tld = tldextract.TLDExtract(suffix_list_urls=())

_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_EMPHASIS = re.compile(r"[*_`~]+")
_STAR_BULLET = re.compile(r"(?m)^(\s*)\*\s+")
_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'()\[\]{}|\\^`]+")
_LIST_ITEM = re.compile(r"^( {0,1})(?:\d+[.)]|[-*•+])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_TRACKING_PARAMS = {"ref", "fbclid", "gclid", "mc_cid", "mc_eid", "srsltid"}


@dataclass
class MentionResult:
    brand: str
    is_own: bool
    count: int
    rank_first: int
    rank_list: int | None
    snippet: str
    first_pos: int


@dataclass
class CitationResult:
    url: str
    domain: str
    kind: str  # cited | retrieved | inline
    position: int


# ---------- text ----------


def normalize_text(text: str) -> str:
    """NFKC, markdown links -> link text, emphasis markers removed. Newlines are kept."""
    text = unicodedata.normalize("NFKC", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _STAR_BULLET.sub(r"\1- ", text)  # keep "* item" bullets before stripping emphasis
    return _MD_EMPHASIS.sub("", text)


@lru_cache(maxsize=256)
def brand_pattern(names: tuple[str, ...], case_sensitive: bool = False) -> re.Pattern[str]:
    """Whole-name match. Longest name first so an alias can't split a longer name."""
    alts = [
        re.escape(unicodedata.normalize("NFKC", n)).replace(r"\ ", r"\s+")
        for n in sorted(names, key=len, reverse=True)
    ]
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])", flags)


def _list_items(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the answer's top-level items, in order.

    If any heading exists, headings are the items (bullets under them are details).
    Otherwise top-level list lines and table body rows are the items. Nested lines
    belong to the item above them.
    """
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    headings = [i for i, ln in enumerate(lines) if _HEADING.match(ln)]
    if headings:
        starts = headings
    else:
        starts = []
        in_table = False
        for i, ln in enumerate(lines):
            if _TABLE_ROW.match(ln):
                if not in_table:  # first row of a table is the header
                    in_table = True
                    continue
                if not _TABLE_SEP.match(ln):
                    starts.append(i)
                continue
            in_table = False
            if _LIST_ITEM.match(ln):
                starts.append(i)

    spans = []
    for n, i in enumerate(starts):
        end_line = starts[n + 1] if n + 1 < len(starts) else len(lines)
        spans.append((offsets[i], offsets[end_line] - 1 if end_line < len(lines) else len(text)))
    return spans


def _snippet(text: str, pos: int) -> str:
    start = 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() > pos:
            end = m.start()
            break
        start = m.end()
    else:
        end = len(text)
    s = " ".join(text[start:end].split())
    return s if len(s) <= SNIPPET_MAX else s[: SNIPPET_MAX - 1] + "…"


def extract_mentions(answer: str, own: Brand, competitors: list[Brand]) -> list[MentionResult]:
    text = normalize_text(answer)
    items = _list_items(text)
    found: list[MentionResult] = []
    for brand in [own, *competitors]:
        matches = list(brand_pattern(tuple(brand.names), brand.case_sensitive).finditer(text))
        if not matches:
            continue
        first = matches[0].start()
        rank_list = None
        for idx, (s, e) in enumerate(items, 1):
            if any(s <= m.start() < e for m in matches):
                rank_list = idx
                break
        found.append(
            MentionResult(
                brand=brand.name,
                is_own=brand is own,
                count=len(matches),
                rank_first=0,
                rank_list=rank_list,
                snippet=_snippet(text, first),
                first_pos=first,
            )
        )
    for rank, m in enumerate(sorted(found, key=lambda m: m.first_pos), 1):
        m.rank_first = rank
    return found


# ---------- URLs ----------


def normalize_url(url: str) -> str | None:
    url = (url or "").strip().rstrip(".,;:!?")
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/")
    return urlunsplit(("https", host, path, urlencode(query), ""))


def domain_of(url: str) -> str:
    ext = _tld(url)
    registered = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
    return (registered or urlsplit(url).hostname or "").lower()


def extract_citations(
    answer: str, cited: list[str], retrieved: list[str]
) -> list[CitationResult]:
    inline = [m.group(0) for m in _URL_IN_TEXT.finditer(unicodedata.normalize("NFKC", answer))]
    out: list[CitationResult] = []
    for kind, urls in (("cited", cited), ("retrieved", retrieved), ("inline", inline)):
        seen: set[str] = set()
        for raw in urls:
            url = normalize_url(raw)
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                CitationResult(url=url, domain=domain_of(url), kind=kind, position=len(seen))
            )
    return out
