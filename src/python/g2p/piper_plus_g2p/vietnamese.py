"""Vietnamese phonemizer using sea-g2p for piper-plus-g2p.

Converts Vietnamese text (including Vietnamese-English code-switching, numbers,
dates, units, currency, URLs, and <en>/<math> tags) to clean IPA token lists
via ``sea_g2p.SEAPipeline(lang="vi")``.

Level-1 token disambiguation applied over raw ``sea-g2p`` output:
  * ``ɜ`` (U+025C) not followed by ``ː`` -> ``"vi_tone3"`` (sac tone; 100%
    distinct from English ``/ɜː/`` in ``sea_g2p.bin``).
  * ``t̪`` (U+0074 U+032A) preserved as single PUA token ``"t\u032a"``
    (distinct from Vietnamese ``th`` = ``"t"``).
  * ``e-`` preserved as single PUA token ``"e-"`` (anh/ach vs enh/ech).
  * ``y`` (U+0079) -> ``"ɯ"`` (Vietnamese close back unrounded vowel, avoiding
    collision with Japanese palatal glide ``"y"`` in multilingual mode).
  * Explicit tone markers ``1, 2, ɜ(?!ː), 4, 5, 6`` -> ``"vi_tone1".."vi_tone6"``.
  * Rare English loanword IPA characters (``ᵻ``, ``n̩``, ``ɬ``, ``ʲ``, ``ç``,
    ``ʐ``, ``ʔ``) normalized into the existing multilingual inventory so every
    entry in ``sea_g2p.bin`` encodes with zero unknown symbols.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata

from .base import Phonemizer, ProsodyInfo

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "VietnamesePhonemizer",
    "phonemize_vietnamese",
    "phonemize_vietnamese_with_prosody",
]

_SEA_PIPELINE = None
_SEA_PIPELINE_LOCK = threading.Lock()

# Multi-codepoint tokens emitted as a single token (checked longest-first).
# Keep long vowels like "aː", "uː", "ɜː" split into vowel + "ː" to stay aligned
# with EnglishPhonemizer and base checkpoint vocabularies.
_MULTI_TOKENS: tuple[str, ...] = (
    "t\u032a",          # Vietnamese dental plosive t (PUA 0xE065)
    "e-",               # Retracted vowel in anh/ach (PUA 0xE066)
    "t\u0283",          # tʃ (PUA 0xE054)
    "d\u0292",          # dʒ (PUA 0xE055)
    "a\u026a",          # aɪ (PUA 0xE028)
    "e\u026a",          # eɪ (PUA 0xE029)
    "a\u028a",          # aʊ (PUA 0xE02A)
    "o\u028a",          # oʊ (PUA 0xE02B)
    "\u0254\u026a",     # ɔɪ (PUA 0xE062)
    "\u025b\u0303",     # ɛ̃ (PUA 0xE056)
    "\u0251\u0303",     # ɑ̃ (PUA 0xE057)
    "\u0254\u0303",     # ɔ̃ (PUA 0xE058)
    "\u0153\u0303",     # œ̃ (PUA 0xE063)
    "\u0250\u0303",     # ɐ̃ (PUA 0xE064)
)

_TONE_DIGIT_MAP: dict[str, tuple[str, int]] = {
    "1": ("vi_tone1", 1),
    "2": ("vi_tone2", 2),
    "4": ("vi_tone4", 4),
    "5": ("vi_tone5", 5),
    "6": ("vi_tone6", 6),
}

_RARE_SINGLE_NORM: dict[str, str] = {
    "y": "\u026f",          # y (sea-g2p VI 'ư') -> ɯ
    "\u1d7b": "\u026a",     # ᵻ -> ɪ
    "\u026c": "l",          # ɬ -> l
    "\u02b2": "j",          # ʲ -> j
    "\u00e7": "x",          # ç -> x
    "\u0290": "\u027b",     # ʐ -> ɻ
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
    """Return a lazily initialized singleton ``SEAPipeline(lang='vi')``."""
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
    """Separate leading and trailing punctuation from a whitespace chunk."""
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


def _scan_body_tokens(body: str) -> tuple[list[str], int, int]:
    """Scan a single syllable/word body into tokens, tone (0..6), and stress (0..2)."""
    tokens: list[str] = []
    tone = 0
    stress = 0
    n = len(body)
    i = 0

    while i < n:
        # Syllabic nasal n̩ -> ə + n
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

        # ɜ not followed by ː is Vietnamese sac tone (tone 3); ɜː is English vowel.
        if ch == "\u025c":
            if i + 1 < n and body[i + 1] == "\u02d0":
                tokens.append("\u025c")
            else:
                tokens.append("vi_tone3")
                tone = 3
            i += 1
            continue

        if ch in _TONE_DIGIT_MAP:
            tone_tok, tone_val = _TONE_DIGIT_MAP[ch]
            tokens.append(tone_tok)
            tone = tone_val
            i += 1
            continue

        if ch == "\u02c8":  # ˈ primary stress
            tokens.append(ch)
            stress = max(stress, 2)
            i += 1
            continue

        if ch == "\u02cc":  # ˌ secondary stress
            tokens.append(ch)
            stress = max(stress, 1)
            i += 1
            continue

        if ch in _RARE_SINGLE_NORM:
            tokens.append(_RARE_SINGLE_NORM[ch])
            i += 1
            continue

        if ch in _DROP_CHARS:
            i += 1
            continue

        # Skip any unexpected combining marks rather than leaking multi-CP artifacts
        if unicodedata.category(ch) == "Mn":
            i += 1
            continue

        tokens.append(ch)
        i += 1

    return tokens, tone, stress


def tokenize_sea_g2p_output(
    ipa_str: str,
) -> tuple[list[str], list[ProsodyInfo | None]]:
    """Convert raw ``sea-g2p`` output string into ``(tokens, prosody_list)``."""
    cleaned = ipa_str.strip()
    if not cleaned:
        return [], []

    tokens: list[str] = []
    prosody_list: list[ProsodyInfo | None] = []
    need_space = False

    for chunk in _RE_WHITESPACE.split(cleaned):
        if not chunk:
            continue

        lead_punct, body, trail_punct = _peel_punctuation(chunk)

        for p in lead_punct:
            tokens.append(p)
            prosody_list.append(None)

        if body:
            # Skip converting unnormalized pure digit runs (e.g. from <en>123</en>)
            # into Vietnamese tone markers.
            if not body.isdigit():
                body_tokens, tone, stress = _scan_body_tokens(body)
                if body_tokens:
                    if need_space:
                        tokens.append(" ")
                        prosody_list.append(ProsodyInfo(a1=0, a2=0, a3=0))

                    phoneme_count = sum(
                        1 for t in body_tokens if t not in ("\u02c8", "\u02cc")
                    )
                    syl_prosody = ProsodyInfo(
                        a1=tone,
                        a2=stress,
                        a3=max(phoneme_count, 1),
                    )
                    for tok in body_tokens:
                        tokens.append(tok)
                        prosody_list.append(syl_prosody)
                    need_space = True

        for p in trail_punct:
            tokens.append(p)
            prosody_list.append(None)

    return tokens, prosody_list


def phonemize_vietnamese_with_prosody(
    text: str,
) -> tuple[list[str], list[ProsodyInfo | None]]:
    """Convert Vietnamese text to IPA tokens with prosody information."""
    if not text or not text.strip():
        return [], []

    pipeline = _get_sea_pipeline()
    raw_ipa = pipeline.run(text)
    return tokenize_sea_g2p_output(raw_ipa)


def phonemize_vietnamese(text: str) -> list[str]:
    """Convert Vietnamese text to IPA phoneme token list."""
    tokens, _ = phonemize_vietnamese_with_prosody(text)
    return tokens


class VietnamesePhonemizer(Phonemizer):
    """Vietnamese phonemizer backed by ``sea-g2p``."""

    def phonemize(self, text: str) -> list[str]:
        text = self._strip_control_chars(text)
        if not text or not text.strip():
            return []
        return phonemize_vietnamese(text)

    def phonemize_with_prosody(
        self, text: str
    ) -> tuple[list[str], list[ProsodyInfo | None]]:
        text = self._strip_control_chars(text)
        if not text or not text.strip():
            return [], []
        return phonemize_vietnamese_with_prosody(text)

    def get_phoneme_id_map(self) -> dict[str, list[int]] | None:
        return None
