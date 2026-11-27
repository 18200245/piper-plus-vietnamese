#!/usr/bin/env python3
"""Align and extend a preprocessed Vietnamese dataset for fine-tuning from a base checkpoint.

Preserves every existing phoneme ID in the base model's ``phoneme_id_map``
(for example ``ayousanz/piper-plus-base`` ``releases/zs-v1`` with 173 symbols
and 6 languages), canonicalizes any raw multi-codepoint aliases via
``map_token()``, appends newly introduced Vietnamese tokens starting at
``base["num_symbols"]``, and appends ``"vi"`` to ``language_id_map``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from piper_plus_g2p.encode.pua import PUA_COMPAT_VERSION, map_token


def _canonicalize_base_map(base_pm: dict[str, list[int]]) -> dict[str, list[int]]:
    """Map raw multi-codepoint aliases (e.g. ɔɪ, œ̃, ɐ̃) to single PUA chars."""
    canonical: dict[str, list[int]] = {}
    for token, ids in base_pm.items():
        mapped = map_token(token, strict=True)
        if mapped in canonical and canonical[mapped] != ids:
            raise ValueError(
                f"Conflicting IDs for canonical token {mapped!r}: "
                f"{canonical[mapped]} vs {ids}"
            )
        canonical[mapped] = list(ids)
    return canonical


def prepare(source: Path, base_config: Path, output: Path) -> int:
    """Remap *source* dataset onto *base_config* extended with Vietnamese tokens."""
    source, base_config, output = map(Path, (source, base_config, output))
    if output.exists():
        raise FileExistsError(output)

    old = json.loads((source / "config.json").read_text(encoding="utf-8"))
    base = json.loads(base_config.read_text(encoding="utf-8"))

    if old["num_speakers"] != 1:
        raise ValueError("This helper requires a single-speaker input dataset")

    # Build inverse map: source_id -> token
    inverse: dict[int, str] = {}
    for token, ids in old["phoneme_id_map"].items():
        if len(ids) != 1 or ids[0] in inverse:
            raise ValueError("Source vocabulary must assign a unique ID to each token")
        inverse[ids[0]] = token

    # Preserve base IDs identically (canonicalizing known raw aliases to PUA)
    extended_pm = _canonicalize_base_map(base["phoneme_id_map"])
    next_id = int(base["num_symbols"])

    # Append any source tokens missing from base in deterministic source ID order
    for _src_id, token in sorted((ids[0], tok) for tok, ids in old["phoneme_id_map"].items()):
        canonical_tok = map_token(token, strict=True)
        if canonical_tok not in extended_pm:
            extended_pm[canonical_tok] = [next_id]
            next_id += 1

    # Verify all keys are single codepoints
    bad_keys = [k for k in extended_pm if len(k) != 1]
    if bad_keys:
        raise ValueError(f"Extended phoneme_id_map has multi-codepoint keys: {bad_keys!r}")

    old_languages = {v: k for k, v in old.get("language_id_map", {"vi": 0}).items()}
    base_languages = dict(base.get("language_id_map", {}))
    if "vi" not in base_languages:
        next_lang_id = (max(base_languages.values()) + 1) if base_languages else 0
        base_languages["vi"] = next_lang_id

    entries = []
    for number, line in enumerate(
        (source / "dataset.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        remapped: list[int] = []
        for old_id in entry["phoneme_ids"]:
            token = inverse.get(old_id)
            if token is None:
                raise ValueError(f"Line {number}: unknown source ID {old_id}")
            canonical_tok = map_token(token, strict=True)
            target = extended_pm.get(canonical_tok)
            if target is None or len(target) != 1:
                raise ValueError(
                    f"Line {number}: token {canonical_tok!r} missing from extended map"
                )
            remapped.append(target[0])

        prosody = entry.get("prosody_features")
        if prosody is not None and len(prosody) != len(remapped):
            raise ValueError(
                f"Line {number}: prosody length ({len(prosody)}) != phoneme_ids ({len(remapped)})"
            )

        language = old_languages.get(entry.get("language_id", 0), "vi")
        if language not in base_languages:
            raise ValueError(f"Line {number}: unsupported language {language!r}")

        entry["phoneme_ids"] = remapped
        entry["language_id"] = base_languages[language]
        entry.pop("speaker_embedding_path", None)
        entry.pop("speaker_id", None)

        for key in ("audio_norm_path", "audio_spec_path"):
            candidates = {(source / entry[key]).resolve(), Path(entry[key]).resolve()}
            matches = [p for p in candidates if p.is_file()]
            if not matches:
                raise ValueError(f"Line {number}: missing cache {entry[key]}")
            entry[key] = str(matches[0])

        entries.append(entry)

    if not entries:
        raise ValueError("Input dataset is empty")

    output.mkdir(parents=True)
    out_config = dict(base)
    out_config.update(
        dataset=output.name,
        num_symbols=len(extended_pm),
        phoneme_id_map=extended_pm,
        num_speakers=1,
        speaker_id_map={},
        language_id_map=base_languages,
        num_languages=max(base_languages.values()) + 1,
        pua_compat_version=PUA_COMPAT_VERSION,
        prosody_num_symbols=old.get("prosody_num_symbols", 11),
        prosody_id_map=old.get(
            "prosody_id_map", {str(i): [i] for i in range(11)}
        ),
    )
    (output / "config.json").write_text(
        json.dumps(out_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "dataset.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in entries),
        encoding="utf-8",
    )
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {"utterances": prepare(args.input_dir, args.base_config, args.output_dir)}
        )
    )


if __name__ == "__main__":
    main()
