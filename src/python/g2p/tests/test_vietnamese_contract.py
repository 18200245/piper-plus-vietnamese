"""Contract tests for Vietnamese phoneme inventory and full sea_g2p.bin coverage."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from piper_plus_g2p.encode.encoder import PiperEncoder
from piper_plus_g2p.encode.id_maps import _VIETNAMESE_PHONEMES, get_phoneme_id_map
from piper_plus_g2p.encode.pua import PUA_COMPAT_VERSION, map_token

try:
    from tests.conftest import requires_vi
except ImportError:
    from conftest import requires_vi


def _is_vi_word(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if code in (0x0102, 0x0103, 0x0110, 0x0111, 0x01A0, 0x01A1, 0x01AF, 0x01B0) or (
            0x1EA0 <= code <= 0x1EF9
        ):
            return True
    return False


def test_vietnamese_id_map_shape_and_prefixes():
    """Verify 199-symbol inventory, PUA v3 codepoints, and backwards-compatible prefixes."""
    vi_map = get_phoneme_id_map("vi")
    multi_map = get_phoneme_id_map("multilingual")
    assert vi_map == multi_map
    assert len(vi_map) == 199
    assert PUA_COMPAT_VERSION == 3

    all_ids = [ids[0] for ids in vi_map.values()]
    assert all(len(ids) == 1 for ids in vi_map.values())
    assert sorted(all_ids) == list(range(199))
    assert all(len(k) == 1 for k in vi_map)

    # New Vietnamese PUA v3 entries (0xE065..0xE06C) must exist as single codepoints
    expected_pua = {
        "t\u032a": "\ue065",
        "e-": "\ue066",
        "vi_tone1": "\ue067",
        "vi_tone2": "\ue068",
        "vi_tone3": "\ue069",
        "vi_tone4": "\ue06a",
        "vi_tone5": "\ue06b",
        "vi_tone6": "\ue06c",
    }
    for tok, expected_char in expected_pua.items():
        mapped = map_token(tok, strict=True)
        assert mapped == expected_char
        assert mapped in vi_map

    # All _VIETNAMESE_PHONEMES entries must resolve to valid single-codepoint keys
    for tok in _VIETNAMESE_PHONEMES:
        assert map_token(tok, strict=True) in vi_map

    # Earlier snapshots stay anchored below ID 185 (v1.0: 0..172, v1.1: 0..184)
    min_new_id = min(vi_map[chr(cp)][0] for cp in range(0xE065, 0xE06D))
    assert min_new_id >= 185


@requires_vi
def test_sea_g2p_bin_en_codepoint_and_pronunciation_encoding():
    """Verify every EN pronunciation codepoint and a broad slice encode cleanly."""
    import sea_g2p

    from piper_plus_g2p.vietnamese import tokenize_sea_g2p_output

    bin_path = Path(os.path.dirname(sea_g2p.__file__)) / "sea_g2p.bin"
    assert bin_path.is_file()

    data = bin_path.read_bytes()
    magic, _ver, _num_str, n2, _n3, off_str, *_rest = struct.unpack("<4s11I", data[:48])
    assert magic == b"SEAP"

    pool = data[48:off_str].decode("utf-8", "replace").split("\x00")[: n2 * 2]
    en_prons = sorted({s[4:] for s in pool if s.startswith("<en>")})
    assert len(en_prons) > 100_000

    id_map = get_phoneme_id_map("vi")
    encoder = PiperEncoder(id_map, strict=True)

    # Guarantee 100% character coverage + dense deterministic stride across 1M+ entries
    char_reps: dict[str, str] = {}
    for pron in en_prons:
        for ch in pron:
            if ch not in char_reps:
                char_reps[ch] = pron

    sample = sorted(set(char_reps.values()) | set(en_prons[::40]))
    for pron in sample:
        tokens, prosody = tokenize_sea_g2p_output(pron)
        ids, aligned = encoder.encode_with_prosody(tokens, prosody)
        assert len(ids) == len(aligned)
        assert len(ids) >= 2  # BOS + EOS


@requires_vi
def test_sea_g2p_vi_lexicon_zero_unknown_tokens():
    """Phonemize thousands of Vietnamese words from sea_g2p.bin with zero unknown tokens."""
    import sea_g2p
    from sea_g2p import SEAPipeline

    from piper_plus_g2p.vietnamese import tokenize_sea_g2p_output

    bin_path = Path(os.path.dirname(sea_g2p.__file__)) / "sea_g2p.bin"
    data = bin_path.read_bytes()
    _magic, _ver, _num_str, n2, _n3, off_str, *_rest = struct.unpack("<4s11I", data[:48])
    pool = data[48:off_str].decode("utf-8", "replace").split("\x00")[: n2 * 2]

    vi_words = sorted(
        {
            s
            for s in pool
            if not s.startswith("<en>") and 1 <= len(s) <= 24 and _is_vi_word(s)
        }
    )
    assert len(vi_words) > 1000

    # Test deterministic 5,000-word slice via Rust Rayon batch converter
    batch = vi_words[:: max(1, len(vi_words) // 5000)][:5000]
    pipeline = SEAPipeline(lang="vi")
    raw_outputs = pipeline.g2p.phonemize_batch(batch)

    id_map = get_phoneme_id_map("vi")
    encoder = PiperEncoder(id_map, strict=True)

    for word, raw_ipa in zip(batch, raw_outputs, strict=True):
        tokens, prosody = tokenize_sea_g2p_output(raw_ipa)
        ids, aligned = encoder.encode_with_prosody(tokens, prosody)
        assert len(ids) == len(aligned), f"Length mismatch for word {word!r}"
