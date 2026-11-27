"""Per-epoch random dataset TTS sampling during training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile
import torch


def resolve_sample_pool(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
) -> Any:
    """Return a dataset pool covering the full dataset when possible."""
    for candidate in (train_dataset, val_dataset):
        base = getattr(candidate, "dataset", None)
        if base is not None and len(base) > 0:
            return base

    for candidate in (val_dataset, train_dataset, test_dataset):
        if candidate is not None and len(candidate) > 0:
            return candidate

    return None


def _to_long_tensor(value: Any, *, device: torch.device | str | None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.long()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.long)
    return tensor.to(device) if device is not None else tensor


def _to_optional_lid(
    raw_lid: Any,
    *,
    num_languages: int,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if num_languages <= 1 or raw_lid is None:
        return None
    if isinstance(raw_lid, torch.Tensor):
        lid = raw_lid.long().view(-1)[:1]
    else:
        lid = torch.tensor([int(raw_lid)], dtype=torch.long)
    return lid.to(device) if device is not None else lid


def synthesize_random_dataset_sample(
    model_g: torch.nn.Module,
    dataset: Any,
    output_dir: str | Path,
    *,
    epoch: int,
    global_step: int,
    sample_rate: int,
    num_speakers: int = 1,
    num_languages: int = 1,
    spk_embed_dim: int = 192,
    device: torch.device | str | None = None,
    rng: random.Random | None = None,
    noise_scale: float = 0.4,
    length_scale: float = 1.0,
    noise_scale_w: float = 0.5,
    max_len: int = 1000,
) -> dict[str, Any] | None:
    """Pick a random dataset utterance, synthesize WAV audio, and log metadata."""
    if dataset is None:
        return None
    total = len(dataset)
    if total == 0:
        return None

    picker = rng if rng is not None else random.Random()  # noqa: S311
    dataset_index = picker.randrange(total)
    sample = dataset[dataset_index]

    raw_g = getattr(model_g, "_orig_mod", model_g)
    target_device = device
    if target_device is None:
        try:
            target_device = next(raw_g.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

    phoneme_ids = _to_long_tensor(sample.phoneme_ids, device=target_device)
    if phoneme_ids.ndim != 1:
        phoneme_ids = phoneme_ids.view(-1)
    seq_len = int(phoneme_ids.numel())
    if seq_len == 0:
        return None

    text = phoneme_ids.unsqueeze(0)
    text_lengths = torch.tensor([seq_len], dtype=torch.long, device=target_device)
    lid = _to_optional_lid(
        getattr(sample, "language_id", None),
        num_languages=num_languages,
        device=target_device,
    )

    prosody_features = None
    if getattr(raw_g, "prosody_dim", 0) > 0:
        raw_prosody = getattr(sample, "prosody_features", None)
        if raw_prosody is not None:
            prosody_tensor = _to_long_tensor(raw_prosody, device=target_device)
            if prosody_tensor.ndim == 2:
                prosody_tensor = prosody_tensor.unsqueeze(0)
            prosody_features = prosody_tensor

    spk_emb = None
    if num_speakers > 1:
        raw_spk_emb = getattr(sample, "speaker_embedding", None)
        if raw_spk_emb is not None:
            spk_tensor = (
                raw_spk_emb.float()
                if isinstance(raw_spk_emb, torch.Tensor)
                else torch.as_tensor(np.asarray(raw_spk_emb), dtype=torch.float32)
            )
            spk_emb = spk_tensor.view(1, -1).to(target_device)
        else:
            spk_emb = torch.zeros(1, spk_embed_dim, device=target_device)

    was_training = raw_g.training
    raw_g.eval()
    try:
        with torch.no_grad():
            audio, *_ = raw_g.infer(
                text,
                text_lengths,
                sid=None,
                lid=lid,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                max_len=max_len,
                prosody_features=prosody_features,
                speaker_embeddings=spk_emb,
            )
    finally:
        if was_training:
            raw_g.train()

    audio_np = np.nan_to_num(
        audio.detach().float().squeeze().cpu().numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    audio_np = np.atleast_1d(audio_np).astype(np.float32, copy=False)
    if audio_np.size == 0:
        return None

    peak = float(np.abs(audio_np).max())
    if peak > 0.01:
        audio_np = audio_np / peak * 0.95
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = np.clip(audio_np * 32767.0, -32768.0, 32767.0).astype(np.int16)

    samples_dir = Path(output_dir) / "epoch_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    wav_path = samples_dir / f"epoch_{epoch:04d}.wav"
    scipy.io.wavfile.write(str(wav_path), int(sample_rate), audio_i16)

    raw_text = getattr(sample, "text", None)
    text_str = raw_text.strip() if isinstance(raw_text, str) and raw_text.strip() else f"sample_{dataset_index}"
    duration_sec = round(float(audio_i16.size) / float(sample_rate), 3)

    record: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "dataset_index": int(dataset_index),
        "text": text_str,
        "wav_path": str(wav_path),
        "duration_sec": duration_sec,
    }
    manifest_path = samples_dir / "samples.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as mf:
        mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {**record, "audio_float": audio_np}
