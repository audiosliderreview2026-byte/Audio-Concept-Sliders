#!/usr/bin/env python
"""
Optimized identity-preservation loop evaluation for AudioLDM2 LoRA sliders.

For each base tuple (original x0, perturbation p):
  x_mix = x0 + gain*p
  zT_mix = invert(encode(x_mix))       # computed ONCE and reused for all strengths

For each slider strength s:
  y = slider_edit(zT_mix, strength=s)
  y_minus_p = y - gain*p
  zT_back = invert(encode(y_minus_p))
  z = slider_edit(zT_back, strength=-s)
  score = cosine_similarity(spec(x0), spec(z))

This is restart-proof and SLURM-array friendly:
  - central CSV append with fcntl lock
  - --resume skips already completed rows
  - --num_shards / --shard_index split base tuples, not individual strengths,
    so the expensive mixed inversion is shared across all strengths in a shard.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import gc
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized identity loop evaluation for AudioLDM2 LoRA sliders.")

    p.add_argument("--project_root", required=True, help="Path containing the audioldm2_slider package.")
    p.add_argument("--model_path", required=True, help="Local AudioLDM2 model path.")
    p.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint .pt.")
    p.add_argument("--lora_rank", type=int, required=True)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--original_dir", required=True, help="Folder with original 10s footstep wavs.")
    p.add_argument("--perturbation_root", required=True, help="Root folder containing perturbation type subfolders.")
    p.add_argument("--perturbation_types", nargs="+", default=["piano", "guitar", "footsteps", "speech"])
    p.add_argument("--output_csv", required=True, help="Central CSV to create/append to.")
    p.add_argument("--output_dir", default=None, help="Folder for configs, summaries, and optional debug wavs.")
    p.add_argument("--scratch_dir", default=None, help="Scratch folder for temporary/debug wavs. Defaults to output_dir.")

    p.add_argument("--strengths", nargs="+", type=float, default=[-4, -2, -1, -0.5, 0.5, 1, 2, 4])
    p.add_argument("--max_originals", type=int, default=10)
    p.add_argument("--max_perturbations_per_type", type=int, default=50)
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac"])

    p.add_argument("--target_audio_length_s", type=float, default=10.0)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--perturbation_gain", type=float, default=0.5)
    p.add_argument("--mix_peak_normalize", action="store_true")
    p.add_argument("--save_debug_wavs", action="store_true")
    p.add_argument("--max_debug_wavs", type=int, default=20)

    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--train_cfg_scale", type=float, default=3.5)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--invert_prompt", default="")

    p.add_argument("--target_prompt", default="a version of this sound that preserves its rhythm, timing, and overall structure")
    p.add_argument("--neutral_prompt", default="footsteps, preserving their rhythm, timing, and overall structure")
    p.add_argument("--positive_prompt", default="this sound as very angry and agitated footsteps, preserving its rhythm, timing, and overall structure")
    p.add_argument("--negative_prompt", default="this sound as peacefully calm footsteps, preserving its rhythm, timing, and overall structure")

    p.add_argument("--spec_type", choices=["logmel", "stft"], default="logmel")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--mel_fmin", type=float, default=0.0)
    p.add_argument("--mel_fmax", type=float, default=8000.0)

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite_existing", action="store_true")

    p.add_argument("--device", default=None, help="cuda/cpu. Defaults to AudioLDM2Editor auto-resolution.")
    p.add_argument("--progress_every", type=int, default=5, help="Progress print every N base tuples.")

    return p.parse_args()


CSV_FIELDS = [
    "original_sound_name", "original_path", "slider_movement",
    "perturbation_type", "perturbation_sample_name", "perturbation_path",
    "perturbation_gain", "cosine_similarity_spec", "spec_type",
    "first_edit_strength", "second_edit_strength",
    "runtime_sec", "runtime_base_sec", "runtime_strength_sec",
    "shard_index", "num_shards", "status", "error",
]

KEY_FIELDS = [
    "original_sound_name", "slider_movement", "perturbation_type",
    "perturbation_sample_name", "perturbation_gain",
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


def safe_mix(original: np.ndarray, perturb: np.ndarray, gain: float, normalize_if_needed: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    added = gain * perturb
    mix = original + added
    scale = 1.0
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if normalize_if_needed and peak > 1.0:
        scale = peak
        mix = mix / scale
        added = added / scale
    return mix.astype(np.float32), added.astype(np.float32), float(scale)


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
            y=wav, sr=sr, n_fft=args.n_fft, hop_length=args.hop_length,
            n_mels=args.n_mels, fmin=args.mel_fmin, fmax=args.mel_fmax, power=2.0,
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    stft = librosa.stft(wav, n_fft=args.n_fft, hop_length=args.hop_length)
    return np.log1p(np.abs(stft)).astype(np.float32)


def stable_base_hash(base: Tuple[Path, str, Path]) -> int:
    text = "|".join([str(base[0]), base[1], str(base[2])])
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def build_base_combinations(args: argparse.Namespace) -> List[Tuple[Path, str, Path]]:
    originals = list_audio_files(Path(args.original_dir), args.extensions, args.max_originals)
    if not originals:
        raise RuntimeError(f"No original audio files found in {args.original_dir}")
    bases = []
    for perturb_type in args.perturbation_types:
        pdir = Path(args.perturbation_root) / perturb_type
        pfiles = list_audio_files(pdir, args.extensions, args.max_perturbations_per_type)
        if not pfiles:
            print(f"[WARN] No perturbation files found for type={perturb_type} in {pdir}")
            continue
        for original in originals:
            for perturb in pfiles:
                bases.append((original, perturb_type, perturb))
    return sorted(bases, key=lambda x: (x[0].name, x[1], x[2].name))


def maybe_shard_bases(bases: List[Tuple[Path, str, Path]], num_shards: int, shard_index: int):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [b for b in bases if stable_base_hash(b) % num_shards == shard_index]


def pad_or_trim(wav: np.ndarray, n: int) -> np.ndarray:
    if len(wav) >= n:
        return wav[:n].astype(np.float32)
    return np.pad(wav, (0, n - len(wav))).astype(np.float32)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root))

    from audioldm2_slider import AudioLDM2Editor, EditorConfig, LoRAConfig, LoRASliderTrainer, TrainerConfig  # noqa

    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir) if args.output_dir else output_csv.parent / (output_csv.stem + "_artifacts")
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"config_shard{args.shard_index}.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    all_bases = build_base_combinations(args)
    assigned_bases = maybe_shard_bases(all_bases, args.num_shards, args.shard_index)
    existing_keys = set() if args.overwrite_existing else load_existing_keys(output_csv)

    # Drop base tuples whose every strength is already done.
    if args.resume and existing_keys:
        before = len(assigned_bases)
        assigned_bases = [
            b for b in assigned_bases
            if any(row_key_from_values(b[0].name, s, b[1], b[2].name, args.perturbation_gain) not in existing_keys for s in args.strengths)
        ]
        print(f"Resume mode: skipped {before - len(assigned_bases)} fully completed base tuples.")

    print("Total base tuples:", len(all_bases))
    print(f"Assigned base tuples to shard {args.shard_index}/{args.num_shards}:", len(assigned_bases))
    print("Strengths per base:", len(args.strengths))
    print("Naive slider edits for this shard:", len(assigned_bases) * len(args.strengths) * 2)
    print("Optimized mixed inversions for this shard:", len(assigned_bases), "instead of", len(assigned_bases) * len(args.strengths))
    print("Scratch dir:", scratch_dir)

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
        dataset_root="", train_songs=[], val_songs=[],
        prompt_target=args.target_prompt, prompt_neutral=args.neutral_prompt,
        prompt_piano=args.positive_prompt, prompt_guitar=args.negative_prompt,
        train_steps=args.steps, train_cfg_scale=args.train_cfg_scale,
        num_epochs=1, lr_lora=1e-4, max_train_pairs=1, max_val_pairs=1,
        out_dir=str(output_dir),
    )
    trainer = LoRASliderTrainer(editor, trainer_cfg)

    # Evaluation only: never build autograd graphs.
    torch.set_grad_enabled(False)
    
    #editor.pipe.eval()
    editor.pipe.unet.eval()
    editor.pipe.vae.eval()
    
    for module in [
        editor.pipe.unet,
        editor.pipe.vae,
        editor.pipe.text_encoder,
        editor.pipe.text_encoder_2,
        editor.pipe.projection_model,
        editor.pipe.language_model,
    ]:
        try:
            module.requires_grad_(False)
        except Exception:
            pass

    
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

    t_global = time.perf_counter()
    completed = 0
    failed = 0
    skipped = 0
    base_runtimes = []
    strength_runtimes = []
    debug_count = 0

    for base_idx, base in enumerate(tqdm(assigned_bases, desc=f"Base tuples shard {args.shard_index}")):
        original_path, perturb_type, perturb_path = base
        t_base0 = time.perf_counter()

        pending_strengths = []
        for s in args.strengths:
            key = row_key_from_values(original_path.name, float(s), perturb_type, perturb_path.name, args.perturbation_gain)
            if args.resume and not args.overwrite_existing and key in load_existing_keys(output_csv):
                skipped += 1
            else:
                pending_strengths.append(float(s))

        if not pending_strengths:
            continue

        try:
            x0 = load_audio_fixed(original_path, args.target_sr, args.target_audio_length_s)
            p = load_audio_fixed(perturb_path, args.target_sr, args.target_audio_length_s)
            x_mix, added_effective, mix_scale = safe_mix(x0, p, args.perturbation_gain, args.mix_peak_normalize)
            spec0 = spectrogram_features(x0, args.target_sr, args)

            with torch.inference_mode():
                z0_mix = editor.encode_audio_to_z0(x_mix, args.target_sr)
                zT_mix = editor.invert(z0_mix, invert_prompt=args.invert_prompt).to(editor.device, dtype=torch.float32)

        except Exception as e:
            # If base preparation fails, write one error row per pending strength.
            runtime_base = time.perf_counter() - t_base0
            for strength in pending_strengths:
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
                    "runtime_base_sec": runtime_base,
                    "runtime_strength_sec": "",
                    "runtime_sec": runtime_base,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "status": "error",
                    "error": repr(e),
                }
                append_row_locked(output_csv, row)
                failed += 1
            print(f"[ERROR base] {original_path.name}, {perturb_type}/{perturb_path.name}: {e}")
            continue

        runtime_base = time.perf_counter() - t_base0
        base_runtimes.append(runtime_base)

        for strength in pending_strengths:
            t_s0 = time.perf_counter()
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
                "runtime_base_sec": runtime_base,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            try:
                # Step 2: first edit from the shared mixed inversion.
                with torch.inference_mode():
                    z0_first = trainer.forward_edit_latent(
                        zT_mix,
                        direction=1,
                        lora_multiplier=float(strength),
                    )
                    y = editor.decode_latents(z0_first, wav_len=len(x0)).astype(np.float32)
                y = pad_or_trim(y, len(x0))

                # Step 3: subtract perturbation.
                y_minus_p = (y - added_effective).astype(np.float32)
                peak = float(np.max(np.abs(y_minus_p))) if y_minus_p.size else 0.0
                if peak > 1.0:
                    y_minus_p = y_minus_p / peak

                # Step 4: invert edited-minus-perturbation and move back.
                with torch.inference_mode():
                    z0_back_in = editor.encode_audio_to_z0(y_minus_p, args.target_sr)
                    zT_back_in = editor.invert(
                        z0_back_in,
                        invert_prompt=args.invert_prompt,
                    ).to(editor.device, dtype=torch.float32)
                
                    z0_final = trainer.forward_edit_latent(
                        zT_back_in,
                        direction=1,
                        lora_multiplier=float(-strength),
                    )
                    z = editor.decode_latents(z0_final, wav_len=len(x0)).astype(np.float32)
                z = pad_or_trim(z, len(x0))

                specz = spectrogram_features(z, args.target_sr, args)
                score = cosine_similarity(spec0, specz)

                runtime_strength = time.perf_counter() - t_s0
                strength_runtimes.append(runtime_strength)
                row.update({
                    "cosine_similarity_spec": score,
                    "runtime_strength_sec": runtime_strength,
                    "runtime_sec": runtime_base + runtime_strength,
                    "status": "ok",
                    "error": "",
                })
                append_row_locked(output_csv, row)
                completed += 1

                if args.save_debug_wavs and debug_count < args.max_debug_wavs:
                    debug_dir = scratch_dir / f"debug_wavs_shard{args.shard_index}"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{debug_count:04d}_{original_path.stem}_s{strength:.3g}_{perturb_type}_{perturb_path.stem}"
                    sf.write(debug_dir / f"{stem}_original.wav", x0, args.target_sr)
                    sf.write(debug_dir / f"{stem}_mixed.wav", x_mix, args.target_sr)
                    sf.write(debug_dir / f"{stem}_after_first_edit.wav", y, args.target_sr)
                    sf.write(debug_dir / f"{stem}_after_subtract.wav", y_minus_p, args.target_sr)
                    sf.write(debug_dir / f"{stem}_final.wav", z, args.target_sr)
                    debug_count += 1

            except Exception as e:
                runtime_strength = time.perf_counter() - t_s0
                row.update({
                    "cosine_similarity_spec": "",
                    "runtime_strength_sec": runtime_strength,
                    "runtime_sec": runtime_base + runtime_strength,
                    "status": "error",
                    "error": repr(e),
                })
                append_row_locked(output_csv, row)
                failed += 1
                print(f"[ERROR] {original_path.name}, s={strength}, {perturb_type}/{perturb_path.name}: {e}")
                
        del z0_first, y, y_minus_p, z0_back_in, zT_back_in, z0_final, z
        gc.collect()
        
        if args.progress_every > 0 and (base_idx + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t_global
            mean_base = float(np.mean(base_runtimes)) if base_runtimes else float("nan")
            mean_strength = float(np.mean(strength_runtimes)) if strength_runtimes else float("nan")
            remaining_bases = max(0, len(assigned_bases) - (base_idx + 1))
            approx_eta = remaining_bases * (mean_base + len(args.strengths) * mean_strength)
            print(
                f"[PROGRESS] shard={args.shard_index}/{args.num_shards} "
                f"base={base_idx+1}/{len(assigned_bases)} ok={completed} failed={failed} skipped={skipped} "
                f"elapsed={elapsed/60:.1f}min mean_base={mean_base:.1f}s mean_strength={mean_strength:.1f}s eta={approx_eta/3600:.2f}h"
            )
            
    del z0_mix, zT_mix, x0, p, x_mix, added_effective, spec0
    gc.collect()

    elapsed = time.perf_counter() - t_global
    summary = {
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "assigned_base_tuples": len(assigned_bases),
        "elapsed_sec": elapsed,
        "mean_base_runtime_sec": float(np.mean(base_runtimes)) if base_runtimes else None,
        "median_base_runtime_sec": float(np.median(base_runtimes)) if base_runtimes else None,
        "mean_strength_runtime_sec": float(np.mean(strength_runtimes)) if strength_runtimes else None,
        "median_strength_runtime_sec": float(np.median(strength_runtimes)) if strength_runtimes else None,
        "output_csv": str(output_csv),
        "scratch_dir": str(scratch_dir),
    }
    with open(output_dir / f"summary_shard{args.shard_index}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
