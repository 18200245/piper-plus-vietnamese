"""Tests for Vietnamese fine-tuning helper, embedding expansion, and prosody gating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from piper_plus_g2p.encode.id_maps import get_phoneme_id_map


pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _make_zs_v1_base_config(path: Path) -> dict:
    """Create a synthetic zs-v1 config (173 symbols, 6 langs, 3 raw multi-CP aliases)."""
    vi_map = get_phoneme_id_map("vi")
    ordered_tokens = [tok for tok, _ids in sorted(vi_map.items(), key=lambda kv: kv[1][0])]
    assert len(ordered_tokens) >= 173

    # Restore the exact 3 raw multi-codepoint keys present in published zs-v1/config.json
    pua_to_raw = {
        "\ue062": "\u0254\u026a",  # ID 94: ɔɪ
        "\ue063": "\u0153\u0303",  # ID 149: œ̃
        "\ue064": "\u0250\u0303",  # ID 153: ɐ̃
    }
    base_pm: dict[str, list[int]] = {}
    for idx in range(173):
        tok = pua_to_raw.get(ordered_tokens[idx], ordered_tokens[idx])
        base_pm[tok] = [idx]

    cfg = {
        "audio": {"sample_rate": 22050, "hop_size": 256},
        "num_symbols": 173,
        "num_speakers": 571,
        "num_languages": 6,
        "language_id_map": {"ja": 0, "en": 1, "zh": 2, "es": 3, "fr": 4, "pt": 5},
        "speaker_id_map": {"spk0": 0},
        "phoneme_id_map": base_pm,
        "prosody_dim": 16,
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def test_prepare_vietnamese_finetune_preserves_base_and_extends_vi(tmp_path: Path):
    """Verify prepare_vietnamese_finetune canonicalizes zs-v1 aliases and appends vi tokens."""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from prepare_vietnamese_finetune import prepare

    base_cfg_path = tmp_path / "base_config.json"
    base_cfg = _make_zs_v1_base_config(base_cfg_path)

    src_dir = tmp_path / "prepared_vi"
    cache_dir = src_dir / "cache"
    cache_dir.mkdir(parents=True)
    norm_file = cache_dir / "utt_001.norm.pt"
    spec_file = cache_dir / "utt_001.spec.pt"
    norm_file.write_bytes(b"norm")
    spec_file.write_bytes(b"spec")

    vi_map = get_phoneme_id_map("vi")
    src_cfg = {
        "audio": {"sample_rate": 22050, "hop_size": 256},
        "num_symbols": len(vi_map),
        "num_speakers": 1,
        "num_languages": 1,
        "language_id_map": {"vi": 0},
        "speaker_id_map": {},
        "phoneme_id_map": vi_map,
        "prosody_num_symbols": 11,
        "prosody_id_map": {str(i): [i] for i in range(11)},
    }
    (src_dir / "config.json").write_text(
        json.dumps(src_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sample_ids = [
        vi_map["^"][0],
        vi_map["\ue065"][0],  # dental t̪ (new PUA v3)
        vi_map["\ue062"][0],  # ɔɪ (was raw multi-codepoint in zs-v1 at ID 94)
        vi_map["\ue069"][0],  # vi_tone3 (new PUA v3)
        vi_map["$"][0],
    ]
    sample_prosody = [None, {"a1": 3, "a2": 2, "a3": 3}, {"a1": 3, "a2": 2, "a3": 3}, {"a1": 3, "a2": 2, "a3": 3}, None]
    utt = {
        "utterance_id": "utt_001",
        "text": "tôi",
        "phoneme_ids": sample_ids,
        "prosody_features": sample_prosody,
        "language_id": 0,
        "speaker_id": 0,
        "audio_norm_path": str(norm_file),
        "audio_spec_path": str(spec_file),
    }
    (src_dir / "dataset.jsonl").write_text(
        json.dumps(utt, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    out_dir = tmp_path / "aligned_vi"
    count = prepare(src_dir, base_cfg_path, out_dir)
    assert count == 1

    out_cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    out_pm = out_cfg["phoneme_id_map"]

    assert out_cfg["num_symbols"] == 199
    assert len(out_pm) == 199
    assert out_cfg["num_speakers"] == 1
    assert out_cfg["num_languages"] == 7
    assert out_cfg["language_id_map"]["vi"] == 6
    assert out_cfg["pua_compat_version"] == 3
    assert out_cfg["prosody_num_symbols"] == 11
    assert all(len(k) == 1 for k in out_pm)

    # All 173 base symbols keep their exact original IDs (with raw aliases canonicalized to PUA)
    assert out_pm["\ue062"] == base_cfg["phoneme_id_map"]["\u0254\u026a"] == [94]
    assert out_pm["\ue063"] == base_cfg["phoneme_id_map"]["\u0153\u0303"] == [149]
    assert out_pm["\ue064"] == base_cfg["phoneme_id_map"]["\u0250\u0303"] == [153]

    # New Vietnamese tokens start at 173+
    for cp in range(0xE065, 0xE06D):
        assert out_pm[chr(cp)][0] >= 173

    out_lines = (out_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(out_lines) == 1
    out_utt = json.loads(out_lines[0])
    assert out_utt["language_id"] == 6
    assert "speaker_id" not in out_utt
    assert len(out_utt["phoneme_ids"]) == len(sample_ids)
    assert len(out_utt["prosody_features"]) == len(out_utt["phoneme_ids"])
    assert out_utt["phoneme_ids"][2] == 94
    assert out_utt["phoneme_ids"][1] == out_pm["\ue065"][0]
    assert out_utt["phoneme_ids"][3] == out_pm["\ue069"][0]


def test_synthesizer_trn_prosody_language_ids_gating_for_vi():
    """Verify SynthesizerTrn applies prosody conditioning for lid=0,6 and zeros lid=1."""
    torch = pytest.importorskip("torch")
    from piper_train.vits.models import SynthesizerTrn

    model_g = SynthesizerTrn(
        n_vocab=199,
        spec_channels=513,
        segment_size=8192,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(4, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16),
        n_speakers=1,
        n_languages=7,
        gin_channels=512,
        prosody_dim=16,
        prosody_language_ids={0, 6},
    )
    assert model_g.prosody_language_ids == {0, 6}

    batch_size, time_steps = 3, 5
    x = torch.zeros(batch_size, 192, time_steps)
    x_mask = torch.ones(batch_size, 1, time_steps)
    lid = torch.tensor([0, 1, 6], dtype=torch.long)

    zero_prosody = torch.zeros(batch_size, time_steps, 3)
    active_prosody = torch.full((batch_size, time_steps, 3), 3.0)

    out_zero = model_g._prepare_prosody_input(x, x_mask, zero_prosody, lid=lid)
    out_active = model_g._prepare_prosody_input(x, x_mask, active_prosody, lid=lid)

    assert not torch.allclose(out_active[0], out_zero[0])
    assert not torch.allclose(out_active[2], out_zero[2])
    assert torch.allclose(out_active[1], out_zero[1])


def test_expand_checkpoint_embeddings_preserves_pretrained_and_seeds_vi():
    """Verify expand_checkpoint_embeddings retains 0..172 symbols and seeds row 6."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from piper_train.__main__ import expand_checkpoint_embeddings

    torch.manual_seed(0)
    saved_emb = torch.randn(173, 192)
    saved_lang = torch.randn(6, 512)
    saved_spk = torch.randn(571, 512)

    target_emb = torch.zeros(199, 192)
    target_lang = torch.zeros(7, 512)
    other_weight = torch.randn(16, 16)

    saved_sd = {
        "model_g.enc_p.emb.weight": saved_emb,
        "model_g.emb_lang.weight": saved_lang,
        "model_g.emb_g.weight": saved_spk,
        "model_g.spk_proj.0.weight": torch.randn(512, 192),
        "model_g.other": other_weight,
    }
    model_sd = {
        "model_g.enc_p.emb.weight": target_emb,
        "model_g.emb_lang.weight": target_lang,
        "model_g.other": torch.zeros(16, 16),
    }

    adapted = expand_checkpoint_embeddings(saved_sd, model_sd)

    assert adapted["model_g.enc_p.emb.weight"].shape == (199, 192)
    assert torch.equal(adapted["model_g.enc_p.emb.weight"][:173], saved_emb)
    assert torch.equal(adapted["model_g.enc_p.emb.weight"][173:], target_emb[173:])

    assert adapted["model_g.emb_lang.weight"].shape == (7, 512)
    assert torch.equal(adapted["model_g.emb_lang.weight"][:6], saved_lang)
    # Row 6 (vi) is seeded from row 2 (tonal pretrained language)
    assert torch.equal(adapted["model_g.emb_lang.weight"][6], saved_lang[2])
    assert torch.equal(adapted["model_g.other"], other_weight)


def test_vits_model_prosody_language_ids_gating_for_vi():
    """Verify VitsModel forwards prosody_language_ids=[0, 6] and masks non-listed languages."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from piper_train.vits.lightning import VitsModel

    model = VitsModel(
        num_symbols=199,
        num_speakers=1,
        num_languages=7,
        gin_channels=512,
        upsample_rates=(4, 4),
        upsample_kernel_sizes=(16, 16),
        prosody_dim=16,
        prosody_language_ids=[0, 6],
        use_wavlm_discriminator=False,
    )
    assert model.model_g.prosody_language_ids == {0, 6}

    batch_size, time_steps = 3, 5
    x = torch.zeros(batch_size, 192, time_steps)
    x_mask = torch.ones(batch_size, 1, time_steps)
    lid = torch.tensor([0, 1, 6], dtype=torch.long)

    zero_prosody = torch.zeros(batch_size, time_steps, 3)
    active_prosody = torch.full((batch_size, time_steps, 3), 3.0)

    out_zero = model.model_g._prepare_prosody_input(x, x_mask, zero_prosody, lid=lid)
    out_active = model.model_g._prepare_prosody_input(x, x_mask, active_prosody, lid=lid)

    # JA (lid=0) and VI (lid=6) receive active prosody conditioning
    assert not torch.allclose(out_active[0], out_zero[0])
    assert not torch.allclose(out_active[2], out_zero[2])
    # EN (lid=1) is masked out before prosody_proj
    assert torch.allclose(out_active[1], out_zero[1])


def test_synthesize_random_dataset_sample_writes_wav_and_restores_train_mode(tmp_path: Path):
    """Verify per-epoch random dataset TTS sampling produces a valid WAV and preserves train()."""
    import random

    import scipy.io.wavfile
    torch = pytest.importorskip("torch")
    from torch.utils.data import Subset

    from piper_train.vits.dataset import UtteranceTensors
    from piper_train.vits.epoch_sample import resolve_sample_pool, synthesize_random_dataset_sample
    from piper_train.vits.models import SynthesizerTrn

    class _DummyDataset:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    torch.manual_seed(0)
    model_g = SynthesizerTrn(
        n_vocab=199,
        spec_channels=513,
        segment_size=8192,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(4, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16),
        n_speakers=1,
        n_languages=7,
        gin_channels=512,
        prosody_dim=16,
        prosody_language_ids={0, 6},
    )
    model_g.train()

    items = [
        UtteranceTensors(
            phoneme_ids=torch.tensor([1, 20, 21, 2], dtype=torch.long),
            spectrogram=torch.zeros(513, 4),
            audio_norm=torch.zeros(1, 256),
            language_id=torch.tensor([6], dtype=torch.long),
            text=txt,
            prosody_features=torch.tensor([[0, 0, 0], [3, 2, 2], [3, 2, 2], [0, 0, 0]], dtype=torch.long),
        )
        for txt in ("Xin chào", "Trời hôm nay đẹp", "Tôi thích nghe nhạc")
    ]
    base_ds = _DummyDataset(items)
    train_subset = Subset(base_ds, [0, 1])
    val_subset = Subset(base_ds, [2])

    pool = resolve_sample_pool(train_subset, val_subset, None)
    assert pool is base_ds

    record = synthesize_random_dataset_sample(
        model_g,
        pool,
        tmp_path,
        epoch=1,
        global_step=10,
        sample_rate=22050,
        num_languages=7,
        rng=random.Random(42),
    )

    assert record is not None
    assert model_g.training is True
    assert record["epoch"] == 1
    assert record["global_step"] == 10
    assert record["text"] in ("Xin chào", "Trời hôm nay đẹp", "Tôi thích nghe nhạc")
    assert record["duration_sec"] > 0

    wav_path = Path(record["wav_path"])
    assert wav_path == tmp_path / "epoch_samples" / "epoch_0001.wav"
    assert wav_path.is_file()

    sr, pcm = scipy.io.wavfile.read(str(wav_path))
    assert sr == 22050
    assert pcm.dtype.name == "int16"
    assert pcm.size > 0

    manifest_lines = (tmp_path / "epoch_samples" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    logged = json.loads(manifest_lines[0])
    assert logged["epoch"] == 1
    assert logged["text"] == record["text"]
    assert "audio_float" not in logged
