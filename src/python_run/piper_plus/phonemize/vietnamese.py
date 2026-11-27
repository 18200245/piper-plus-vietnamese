"""Vietnamese phonemizer for Piper TTS runtime (piper-plus).

Runtime counterpart to ``piper_plus_g2p.vietnamese``, backed by ``sea-g2p``.
Returns PUA-mapped single-codepoint tokens wrapped in BOS/EOS (``^`` ... ``$``),
matching the convention used by ``piper_plus.phonemize.swedish``.
"""

from __future__ import annotations

import re
import threading
import unicodedata

from ..text_sanitize import strip_control_chars
from .token_mapper import map_sequence

__all__ = ["phonemize_vietnamese"]

_SEA_PIPELINE = None
_SEA_PIPELINE_LOCK = threading.Lock()

_MULTI_TOKENS: tuple[str, ...] = (
    "t\u032a",
    "e-",
    "t\u0283",
    "d\u0292",
    "a\u026a",
    "e\u026a",
    "a\u028a",
    "o\u028a",
    "\u0254\u026a",
    "\u025b\u0303",
    "\u0251\u0303",
    "\u0254\u0303",
    "\u0153\u0303",
    "\u0250\u0303",
)

_TONE_DIGIT_MAP: dict[str, str] = {
    "1": "vi_tone1",
    "2": "vi_tone2",
    "4": "vi_tone4",
    "5": "vi_tone5",
    "6": "vi_tone6",
}

_RARE_SINGLE_NORM: dict[str, str] = {
    "y": "\u026f",
    "\u1d7b": "\u026a",
    "\u026c": "l",
    "\u02b2": "j",
    "\u00e7": "x",
    "\u0290": "\u027b",
}

_DROP_CHARS = frozenset({"\u0294", "\u0329", "\u032a", "\u0303", "-", "7"})

_PUNCT_MAP: dict[str, str] = {
    ",": ",",
    ".": ".",
    "!": "!",
    "?": "?",
    ";": ",",
    ":": ",",
    "\u2026": ".",
    "\u2025": ".",
    "\u2024": ".",
}

_PUNCT_STRIP_SET = ",.!?;:\u2026\u2025\u2024\"'()[]{}"
_RE_WHITESPACE = re.compile(r"\s+")


def _get_sea_pipeline():
    global _SEA_PIPELINE  # noqa: PLW0603
    if _SEA_PIPELINE is None:
        with _SEA_PIPELINE_LOCK:
            if _SEA_PIPELINE is None:
                try:
                    from sea_g2p import SEAPipeline  # noqa: PLC0415
                except ImportError as exc:
                    raise ImportError(
                        "sea-g2p is required for Vietnamese phonemization. "
                        "Install with: pip install sea-g2p"
                    ) from exc
                _SEA_PIPELINE = SEAPipeline(lang="vi")
    return _SEA_PIPELINE


def _peel_punctuation(chunk: str) -> tuple[list[str], str, list[str]]:
    lead: list[str] = []
    trail: list[str] = []

    start = 0
    end = len(chunk)
    while start < end and chunk[start] in _PUNCT_STRIP_SET:
        mapped = _PUNCT_MAP.get(chunk[start])
        if mapped is not None:
            lead.append(mapped)
        start += 1

    while end > start and chunk[end - 1] in _PUNCT_STRIP_SET:
        mapped = _PUNCT_MAP.get(chunk[end - 1])
        if mapped is not None:
            trail.append(mapped)
        end -= 1

    trail.reverse()
    return lead, chunk[start:end], trail


def _scan_body_tokens(body: str) -> list[str]:
    tokens: list[str] = []
    n = len(body)
    i = 0

    while i < n:
        if body.startswith("n\u0329", i):
            tokens.append("\u0259")
            tokens.append("n")
            i += 2
            continue

        matched_multi = False
        for multi in _MULTI_TOKENS:
            if body.startswith(multi, i):
                tokens.append(multi)
                i += len(multi)
                matched_multi = True
                break
        if matched_multi:
            continue

        ch = body[i]

        if ch == "\u025c":
            if i + 1 < n and body[i + 1] == "\u02d0":
                tokens.append("\u025c")
            else:
                tokens.append("vi_tone3")
            i += 1
            continue

        if ch in _TONE_DIGIT_MAP:
            tokens.append(_TONE_DIGIT_MAP[ch])
            i += 1
            continue

        if ch in _RARE_SINGLE_NORM:
            tokens.append(_RARE_SINGLE_NORM[ch])
            i += 1
            continue

        if ch in _DROP_CHARS or unicodedata.category(ch) == "Mn":
            i += 1
            continue

        tokens.append(ch)
        i += 1

    return tokens


def phonemize_vietnamese(text: str) -> list[str]:
    """Phonemize Vietnamese text. Returns PUA-mapped tokens wrapped in BOS/EOS."""
    text = strip_control_chars(text)
    if not text or not text.strip():
        return []

    raw_ipa = _get_sea_pipeline().run(text).strip()
    if not raw_ipa:
        return []

    phonemes: list[str] = []
    need_space = False

    for chunk in _RE_WHITESPACE.split(raw_ipa):
        if not chunk:
            continue
        lead_punct, body, trail_punct = _peel_punctuation(chunk)
        phonemes.extend(lead_punct)
        if body and not body.isdigit():
            body_tokens = _scan_body_tokens(body)
            if body_tokens:
                if need_space:
                    phonemes.append(" ")
                phonemes.extend(body_tokens)
                need_space = True
        phonemes.extend(trail_punct)

    if not phonemes:
        return []

    return map_sequence(["^"] + phonemes + ["$"])
