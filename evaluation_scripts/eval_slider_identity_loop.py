#!/usr/bin/env python
"""
Identity-preservation loop evaluation for AudioLDM2 LoRA sliders.

Loop for each combination:
  x0 = original footstep
  p  = perturbation sample
  x_mix = x0 + alpha * p
  y = slider_edit(x_mix, strength=s)
  y_minus_p = y - alpha * p
  z = slider_edit(y_minus_p, strength=-s)
  score = cosine_similarity(logmel(x0), logmel(z))

The script is restart-proof and parallel-job friendly:
  - reads an existing central CSV and skips completed keys
  - appends each completed row immediately
  - uses a Unix fcntl lock around CSV writes
  - optional --shard_index / --num_shards for parallel jobs

Expected perturbation directory structure:
  perturbation_root/
    piano/*.wav
    guitar/*.wav
    footsteps/*.wav
    speech/*.wav

Example:
python eval_slider_identity_loop.py \
  --project_root "$WORK/AudioLDM2/audioldm2_slider_project" \
  --model_path "$WORK/models/audioldm2" \
  --lora_path "lora_slider_training_cluster_surface/lora_best.pt" \
  --lora_rank 4 \
  --original_dir "identity_eval/original_footsteps" \
  --perturbation_root "identity_eval/perturbations" \
  --output_csv "identity_loop_results.csv" \
  --steps 50 \
  --train_cfg_scale 3.5 \
  --max_originals 10 \
  --max_perturbations_per_type 2 \
  --strengths -4 -2 -1 -0.5 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Identity loop evaluation for AudioLDM2 LoRA sliders.")

    # Project/model paths
    p.add_argument("--project_root", required=True, help="Path containing the audioldm2_slider package.")
    p.add_argument("--model_path", required=True, help="Local AudioLDM2 model path.")
    p.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint .pt.")
    p.add_argument("--lora_rank", type=int, required=True)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # Data paths
    p.add_argument("--original_dir", required=True, help="Folder with original 10s footstep wavs.")
    p.add_argument("--perturbation_root", required=True, help="Root folder containing perturbation type subfolders.")
    p.add_argument("--perturbation_types", nargs="+", default=["piano", "guitar", "footsteps", "speech"])
    p.add_argument("--output_csv", required=True, help="Central CSV to create/append to.")
    p.add_argument("--output_dir", default=None, help="Optional folder for logs/config/intermediate debug wavs.")

    # Evaluation grid
    p.add_argument("--strengths", nargs="+", type=float, default=[-4, -2, -1, -0.5, 0.5, 1, 2, 4])
    p.add_argument("--max_originals", type=int, default=10)
    p.add_argument("--max_perturbations_per_type", type=int, default=50)
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac"])

    # Audio handling
    p.add_argument("--target_audio_length_s", type=float, default=10.0)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--perturbation_gain", type=float, default=0.5, help="Linear gain applied to perturbation before mixing.")
    p.add_argument("--mix_peak_normalize", action="store_true", help="Peak normalize mixture only if it exceeds [-1, 1].")
    p.add_argument("--save_debug_wavs", action="store_true", help="Save a few intermediate wavs for inspection.")
    p.add_argument("--max_debug_wavs", type=int, default=20)

    # Slider/inference config
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--train_cfg_scale", type=float, default=3.5)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--invert_prompt", default="")

    # Trainer prompts: required by TrainerConfig even if forward_edit_latent may not use all text semantically.
    p.add_argument("--prompt_target", default="a version of this sound that preserves its rhythm, timing, and overall structure")
    p.add_argument("--prompt_neutral", default="the same sound, preserving its rhythm, timing, and overall structure")
    p.add_argument("--prompt_piano", default="this sound as the positive slider concept, preserving its rhythm, timing, and overall structure")
    p.add_argument("--prompt_guitar", default="this sound as the negative slider concept, preserving its rhythm, timing, and overall structure")

    # Spectrogram similarity config
    p.add_argument("--spec_type", choices=["logmel", "stft"], default="logmel")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--mel_fmin", type=float, default=0.0)
    p.add_argument("--mel_fmax", type=float, default=8000.0)

    # Parallelization / restart
    p.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards/jobs.")
    p.add_argument("--shard_index", type=int, default=0, help="This job shard index in [0, num_shards-1].")
    p.add_argument("--resume", action="store_true", help="Skip rows already found in output_csv.")
    p.add_argument("--overwrite_existing", action="store_true", help="Ignore existing CSV rows and recompute assigned rows.")

    # Runtime diagnostics
    p.add_argument("--device", default=None, help="cuda/cpu. Defaults to AudioLDM2Editor auto-resolution.")
    p.add_argument("--progress_every", type=int, default=10)

    return p.parse_args()


CSV_FIELDS = [
    "original_sound_name",
    "original_path",
    "slider_movement",
    "perturbation_type",
    "perturbation_sample_name",
    "perturbation_path",
    "perturbation_gain",
    "cosine_similarity_spec",
    "spec_type",
    "first_edit_strength",
    "second_edit_strength",
    "runtime_sec",
    "shard_index",
    "num_shards",
    "status",
    "error",
]

KEY_FIELDS = [
    "original_sound_name",
    "slider_movement",
    "perturbation_type",
    "perturbation_sample_name",
    "perturbation_gain",
]


def row_key_from_values(original_name: str, strength: float, perturb_type: str, perturb_name: str, gain: float) -> Tuple[str, str, str, str, str]:
    return (original_name, f"{strength:.8g}", perturb_type, perturb_name, f"{gain:.8g}")


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return tuple(str(row[k]) for k in KEY_FIELDS)  # type: ignore


def list_audio_files(folder: Path, extensions: Iterable[str], max_files: int | None = None) -> List[Path]:
    exts = {e.lower() for e in extensions}
    files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    if max_files is not None:
        files = files[:max_files]
    return files


def load_existing_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") == "ok":
                keys.add(row_key(row))
    return keys


def append_row_locked(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_audio_fixed(path: Path, sr: int, length_s: float) -> np.ndarray:
    wav, file_sr = sf.read(str(path), always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr).astype(np.float32)

    target_len = int(round(sr * length_s))
    if len(wav) > target_len:
        wav = wav[:target_len]
    elif len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32)


def safe_mix(original: np.ndarray, perturb: np.ndarray, gain: float, normalize_if_needed: bool) -> Tuple[np.ndarray, float]:
    added = gain * perturb
    mix = original + added
    scale = 1.0
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if normalize_if_needed and peak > 1.0:
        scale = peak
        mix = mix / scale
        added = added / scale
    return mix.astype(np.float32), float(scale)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def spectrogram_features(wav: np.ndarray, sr: int, args: argparse.Namespace) -> np.ndarray:
    if args.spec_type == "logmel":
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmin=args.mel_fmin,
            fmax=args.mel_fmax,
            power=2.0,
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    else:
        stft = librosa.stft(wav, n_fft=args.n_fft, hop_length=args.hop_length)
        mag = np.abs(stft)
        return np.log1p(mag).astype(np.float32)


def stable_combo_hash(combo: Tuple[Path, float, str, Path]) -> int:
    text = "|".join([str(combo[0]), f"{combo[1]:.8g}", combo[2], str(combo[3])])
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def build_combinations(args: argparse.Namespace) -> List[Tuple[Path, float, str, Path]]:
    original_files = list_audio_files(Path(args.original_dir), args.extensions, args.max_originals)
    if not original_files:
        raise RuntimeError(f"No original audio files found in {args.original_dir}")

    combos = []
    for perturb_type in args.perturbation_types:
        pdir = Path(args.perturbation_root) / perturb_type
        pfiles = list_audio_files(pdir, args.extensions, args.max_perturbations_per_type)
        if not pfiles:
            print(f"[WARN] No perturbation files found for type={perturb_type} in {pdir}")
            continue
        for original in original_files:
            for strength in args.strengths:
                for perturb in pfiles:
                    combos.append((original, float(strength), perturb_type, perturb))

    # Deterministic stable ordering.
    combos = sorted(combos, key=lambda x: (x[0].name, x[2], x[3].name, x[1]))
    return combos


def maybe_shard(combos: List[Tuple[Path, float, str, Path]], num_shards: int, shard_index: int):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [c for c in combos if stable_combo_hash(c) % num_shards == shard_index]


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root))

    from audioldm2_slider import (  # noqa: WPS433
        AudioLDM2Editor,
        EditorConfig,
        LoRAConfig,
        LoRASliderTrainer,
        TrainerConfig,
    )

    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir) if args.output_dir else output_csv.parent / (output_csv.stem + "_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"config_shard{args.shard_index}.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    all_combos = build_combinations(args)
    assigned_combos = maybe_shard(all_combos, args.num_shards, args.shard_index)

    existing_keys = set() if args.overwrite_existing else load_existing_keys(output_csv)
    if args.resume and existing_keys:
        before = len(assigned_combos)
        assigned_combos = [
            c for c in assigned_combos
            if row_key_from_values(c[0].name, c[1], c[2], c[3].name, args.perturbation_gain) not in existing_keys
        ]
        print(f"Resume mode: skipped {before - len(assigned_combos)} already completed rows.")

    print("Total combinations:", len(all_combos))
    print(f"Assigned to shard {args.shard_index}/{args.num_shards}:", len(assigned_combos))
    approx_edits = len(assigned_combos) * 2
    print("Approx slider edits for this shard:", approx_edits)

    print("Loading editor...")
    editor_cfg = EditorConfig(
        model_path=args.model_path,
        device=args.device,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        target_audio_length_s=args.target_audio_length_s,
    )
    editor = AudioLDM2Editor(editor_cfg)

    print("Injecting LoRA...")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    editor.inject_lora(lora_cfg, verbose=True)

    print(f"Loading LoRA checkpoint: {args.lora_path}")
    state_dict = torch.load(args.lora_path, map_location="cpu")
    missing, unexpected = editor.pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"LoRA checkpoint loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    trainer_cfg = TrainerConfig(
        dataset_root="",
        train_songs=[],
        val_songs=[],
        prompt_target=args.prompt_target,
        prompt_neutral=args.prompt_neutral,
        prompt_piano=args.prompt_piano,
        prompt_guitar=args.prompt_guitar,
        train_steps=args.steps,
        train_cfg_scale=args.train_cfg_scale,
        num_epochs=1,
        lr_lora=1e-4,
        max_train_pairs=1,
        max_val_pairs=1,
        out_dir=str(output_dir),
    )
    trainer = LoRASliderTrainer(editor, trainer_cfg)

    # Important after Trainer construction: keep inference modules on GPU/device.
    editor.pipe.unet.to(editor.device, dtype=torch.float32)
    editor.pipe.vae.to(editor.device, dtype=torch.float32)
    try:
        editor.pipe.text_encoder.to(editor.device, dtype=torch.float32)
        editor.pipe.text_encoder_2.to(editor.device, dtype=torch.float32)
        editor.pipe.projection_model.to(editor.device, dtype=torch.float32)
        editor.pipe.language_model.to(editor.device, dtype=torch.float32)
    except Exception:
        pass

    print("editor.device:", editor.device)
    print("UNet device:", next(editor.pipe.unet.parameters()).device)

    debug_count = 0
    t_global = time.perf_counter()
    completed = 0
    failed = 0
    runtimes = []

    for combo in tqdm(assigned_combos, desc=f"Identity loop shard {args.shard_index}"):
        original_path, strength, perturb_type, perturb_path = combo
        key = row_key_from_values(original_path.name, strength, perturb_type, perturb_path.name, args.perturbation_gain)

        if args.resume and not args.overwrite_existing and key in load_existing_keys(output_csv):
            continue

        t0 = time.perf_counter()
        row = {
            "original_sound_name": original_path.name,
            "original_path": str(original_path),
            "slider_movement": f"{strength:.8g}",
            "perturbation_type": perturb_type,
            "perturbation_sample_name": perturb_path.name,
            "perturbation_path": str(perturb_path),
            "perturbation_gain": f"{args.perturbation_gain:.8g}",
            "first_edit_strength": f"{strength:.8g}",
            "second_edit_strength": f"{-strength:.8g}",
            "spec_type": args.spec_type,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }

        try:
            x0 = load_audio_fixed(original_path, args.target_sr, args.target_audio_length_s)
            p = load_audio_fixed(perturb_path, args.target_sr, args.target_audio_length_s)
            x_mix, mix_scale = safe_mix(x0, p, args.perturbation_gain, args.mix_peak_normalize)
            added_effective = (args.perturbation_gain * p) / mix_scale

            # Step 2: edit mixed sound with strength s.
            z0_mix = editor.encode_audio_to_z0(x_mix, args.target_sr)
            zT_mix = editor.invert(z0_mix, invert_prompt=args.invert_prompt)
            zT_mix = zT_mix.to(editor.device, dtype=torch.float32)
            z0_first = trainer.forward_edit_latent(zT_mix, direction=1, lora_multiplier=float(strength))
            y = editor.decode_latents(z0_first, wav_len=len(x0)).astype(np.float32)
            y = y[: len(x0)] if len(y) >= len(x0) else np.pad(y, (0, len(x0) - len(y)))

            # Step 3: subtract the same perturbation signal.
            y_minus_p = (y - added_effective).astype(np.float32)
            peak = float(np.max(np.abs(y_minus_p))) if y_minus_p.size else 0.0
            if peak > 1.0:
                y_minus_p = y_minus_p / peak

            # Step 4: edit back with strength -s.
            z0_back_in = editor.encode_audio_to_z0(y_minus_p, args.target_sr)
            zT_back_in = editor.invert(z0_back_in, invert_prompt=args.invert_prompt)
            zT_back_in = zT_back_in.to(editor.device, dtype=torch.float32)
            z0_final = trainer.forward_edit_latent(zT_back_in, direction=1, lora_multiplier=float(-strength))
            z = editor.decode_latents(z0_final, wav_len=len(x0)).astype(np.float32)
            z = z[: len(x0)] if len(z) >= len(x0) else np.pad(z, (0, len(x0) - len(z)))

            spec0 = spectrogram_features(x0, args.target_sr, args)
            specz = spectrogram_features(z, args.target_sr, args)
            score = cosine_similarity(spec0, specz)

            runtime = time.perf_counter() - t0
            runtimes.append(runtime)
            row.update({
                "cosine_similarity_spec": score,
                "runtime_sec": runtime,
                "status": "ok",
                "error": "",
            })
            append_row_locked(output_csv, row)
            completed += 1

            if args.save_debug_wavs and debug_count < args.max_debug_wavs:
                debug_dir = output_dir / "debug_wavs"
                debug_dir.mkdir(exist_ok=True)
                stem = f"{debug_count:04d}_{original_path.stem}_s{strength:.3g}_{perturb_type}_{perturb_path.stem}"
                sf.write(debug_dir / f"{stem}_original.wav", x0, args.target_sr)
                sf.write(debug_dir / f"{stem}_mixed.wav", x_mix, args.target_sr)
                sf.write(debug_dir / f"{stem}_after_first_edit.wav", y, args.target_sr)
                sf.write(debug_dir / f"{stem}_after_subtract.wav", y_minus_p, args.target_sr)
                sf.write(debug_dir / f"{stem}_final.wav", z, args.target_sr)
                debug_count += 1

        except Exception as e:
            runtime = time.perf_counter() - t0
            row.update({
                "cosine_similarity_spec": "",
                "runtime_sec": runtime,
                "status": "error",
                "error": repr(e),
            })
            append_row_locked(output_csv, row)
            failed += 1
            print(f"[ERROR] {original_path.name}, s={strength}, {perturb_type}/{perturb_path.name}: {e}")

        total_done = completed + failed
        if args.progress_every > 0 and total_done % args.progress_every == 0:
            elapsed = time.perf_counter() - t_global
            mean_rt = float(np.mean(runtimes)) if runtimes else float("nan")
            remaining = max(0, len(assigned_combos) - total_done)
            eta = remaining * mean_rt if runtimes else float("nan")
            print(
                f"[PROGRESS] shard={args.shard_index}/{args.num_shards} "
                f"done={total_done}/{len(assigned_combos)} ok={completed} failed={failed} "
                f"elapsed={elapsed/60:.1f}min mean_row={mean_rt:.1f}s eta={eta/3600:.2f}h"
            )

    elapsed = time.perf_counter() - t_global
    summary = {
        "completed": completed,
        "failed": failed,
        "assigned_combinations": len(assigned_combos),
        "elapsed_sec": elapsed,
        "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_sec": float(np.median(runtimes)) if runtimes else None,
        "output_csv": str(output_csv),
    }
    with open(output_dir / f"summary_shard{args.shard_index}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
