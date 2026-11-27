"""Unit tests for Vietnamese G2P, disambiguation, multilingual routing, and runtime mirror."""

from __future__ import annotations

from piper_plus_g2p import available_languages, get_phonemizer
from piper_plus_g2p.base import ProsodyInfo
from piper_plus_g2p.encode.encoder import PiperEncoder
from piper_plus_g2p.encode.id_maps import get_phoneme_id_map
from piper_plus_g2p.multilingual import MultilingualPhonemizer
from piper_plus_g2p.vietnamese import VietnamesePhonemizer, tokenize_sea_g2p_output

try:
    from tests.conftest import requires_vi
except ImportError:
    from conftest import requires_vi


def test_vietnamese_registered_in_registry():
    assert "vi" in available_languages()
    p = get_phonemizer("vi")
    assert isinstance(p, VietnamesePhonemizer)


@requires_vi
def test_vietnamese_six_tones_and_prosody():
    p = VietnamesePhonemizer()

    # sea-g2p leaves level tone (ngang) unmarked; tones 2..6 emit explicit markers
    ma_tokens, ma_prosody = p.phonemize_with_prosody("ma")
    assert ma_tokens == ["m", "\u02c8", "a", "\u02d0"]
    assert all(tok not in ma_tokens for tok in (
        "vi_tone2", "vi_tone3", "vi_tone4", "vi_tone5", "vi_tone6"
    ))
    assert ma_prosody[0].a1 == 0

    marked_words = [
        ("mà", "vi_tone2", 2),
        ("má", "vi_tone3", 3),
        ("mả", "vi_tone4", 4),
        ("mã", "vi_tone5", 5),
        ("mạ", "vi_tone6", 6),
    ]
    for word, expected_tone_tok, expected_a1 in marked_words:
        tokens, prosody = p.phonemize_with_prosody(word)
        assert expected_tone_tok in tokens, f"{word!r} -> {tokens}"
        idx = tokens.index(expected_tone_tok)
        info = prosody[idx]
        assert isinstance(info, ProsodyInfo)
        assert info.a1 == expected_a1
        assert info.a3 >= 2

    # Explicit '1' digit marker in phonetic input also maps cleanly to vi_tone1
    explicit_t1_toks, explicit_t1_pros = tokenize_sea_g2p_output("m\u02c8a\u02d01")
    assert "vi_tone1" in explicit_t1_toks
    assert explicit_t1_pros[explicit_t1_toks.index("vi_tone1")].a1 == 1


@requires_vi
def test_english_long_er_not_confused_with_sac_tone():
    """English /ɜː/ ('bird', 'work') keeps 'ɜ' + 'ː' and must NOT emit vi_tone3."""
    p = VietnamesePhonemizer()
    for eng_word in ("bird", "work"):
        tokens = p.phonemize(eng_word)
        assert "vi_tone3" not in tokens, f"{eng_word!r} emitted vi_tone3: {tokens}"
        assert "\u025c" in tokens and "\u02d0" in tokens, (
            f"Expected ɜ + ː for {eng_word!r}, got {tokens}"
        )

    # Confirm tokenize_sea_g2p_output separates adjacent English ɜː and Vietnamese ɜ
    tokens, prosody = tokenize_sea_g2p_output("b\u02c8\u025c\u02d0d m\u02c8a\u02d0\u025c")
    assert "\u025c" in tokens
    assert "vi_tone3" in tokens
    assert prosody[tokens.index("\u025c")].a1 == 0
    assert prosody[tokens.index("vi_tone3")].a1 == 3


@requires_vi
def test_dental_t_vs_aspirated_th_disambiguation():
    p = VietnamesePhonemizer()
    t_toi = p.phonemize("tôi")
    t_ta = p.phonemize("ta")
    th_thoi = p.phonemize("thời")
    th_tho = p.phonemize("thơ")

    assert "t\u032a" in t_toi
    assert "t\u032a" in t_ta
    assert "t" in th_thoi and "t\u032a" not in th_thoi
    assert "t" in th_tho and "t\u032a" not in th_tho


@requires_vi
def test_retracted_e_minus_and_close_back_unrounded_u_horn():
    p = VietnamesePhonemizer()
    for w in ("anh", "sách"):
        tokens = p.phonemize(w)
        assert "e-" in tokens, f"{w!r} missing e-: {tokens}"

    for w in ("tư", "nước"):
        tokens = p.phonemize(w)
        assert "y" not in tokens, f"{w!r} leaked Japanese y: {tokens}"
        assert "\u026f" in tokens, f"{w!r} missing ɯ: {tokens}"


@requires_vi
def test_codeswitching_numbers_dates_currency_and_punctuation():
    p = VietnamesePhonemizer()
    encoder = PiperEncoder(get_phoneme_id_map("vi"), strict=True)

    sentences = [
        "Tôi dùng iPhone 15 Pro và GitHub mỗi ngày.",
        "Giá 500.000đ \u2013 giảm 50% (ngày 10/05/2025).",
        "\u201cXin chào!\u201d \u2014 anh ấy nói: được rồi\u2026",
    ]
    for text in sentences:
        tokens, prosody = p.phonemize_with_prosody(text)
        assert len(tokens) > 0
        ids, aligned = encoder.encode_with_prosody(tokens, prosody)
        assert len(ids) == len(aligned)
        assert ids[0] == 1 and ids[-1] == 2


@requires_vi
def test_empty_and_control_characters():
    p = VietnamesePhonemizer()
    assert p.phonemize("") == []
    assert p.phonemize("   \t\n  ") == []
    assert p.phonemize("\x00\x07\x1f") == []
    assert p.phonemize_with_prosody("\x00\x07") == ([], [])


@requires_vi
def test_multilingual_phonemizer_vi_routing_and_parity():
    multi_ja_en_vi = MultilingualPhonemizer(["ja", "en", "vi"], default_latin_language="en")
    segs = multi_ja_en_vi.segment_text("Tôi dùng GitHub và Python.")
    assert any(s["language"] == "vi" for s in segs)

    vi_only_multi = MultilingualPhonemizer(["vi"], default_latin_language="vi")
    vi_direct = VietnamesePhonemizer()
    sample = "Xin chào Việt Nam, hôm nay trời rất đẹp!"
    assert vi_only_multi.phonemize(sample) == vi_direct.phonemize(sample)


@requires_vi
def test_python_run_runtime_vietnamese_parity():
    from piper_plus.phonemize.token_mapper import map_sequence
    from piper_plus.phonemize.vietnamese import phonemize_vietnamese as runtime_phonemize_vi

    text = "Hôm nay trời Hà Nội mát mẻ, tôi uống cà phê!"
    expected = map_sequence(["^"] + VietnamesePhonemizer().phonemize(text) + ["$"])
    actual = runtime_phonemize_vi(text)
    assert actual == expected
    assert actual[0] == "^" and actual[-1] == "$"
    assert all(len(tok) == 1 for tok in actual)
