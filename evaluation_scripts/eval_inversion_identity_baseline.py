#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--output_csv", default="inversion_identity_baseline.csv")
    p.add_argument("--output_json", default="inversion_identity_baseline_summary.json")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--target_audio_length_s", type=float, default=10.0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def list_audio_files(audio_dir: Path):
    exts = {".wav", ".flac", ".mp3"}
    return sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])


def logmel_cosine_similarity(
    wav_a,
    wav_b,
    sr,
    n_fft=1024,
    hop_length=160,
    n_mels=64,
    eps=1e-8,
):
    min_len = min(len(wav_a), len(wav_b))
    wav_a = wav_a[:min_len]
    wav_b = wav_b[:min_len]

    mel_a = librosa.feature.melspectrogram(
        y=wav_a.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_b = librosa.feature.melspectrogram(
        y=wav_b.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    log_a = np.log(mel_a + eps).reshape(-1)
    log_b = np.log(mel_b + eps).reshape(-1)

    return float(np.dot(log_a, log_b) / ((np.linalg.norm(log_a) * np.linalg.norm(log_b)) + eps))


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root))

    from audioldm2_slider import EditorConfig, AudioLDM2Editor

    audio_files = list_audio_files(Path(args.audio_dir))
    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]

    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in {args.audio_dir}")

    print(f"Found {len(audio_files)} audio files.")

    editor_cfg = EditorConfig(
        model_path=args.model_path,
        device=args.device,
        steps=args.steps,
        target_audio_length_s=args.target_audio_length_s,
    )
    editor = AudioLDM2Editor(editor_cfg)

    torch.set_grad_enabled(False)
    for module in [
        editor.pipe.unet,
        editor.pipe.vae,
        editor.pipe.text_encoder,
        editor.pipe.text_encoder_2,
        editor.pipe.projection_model,
        editor.pipe.language_model,
    ]:
        try:
            module.eval()
            module.requires_grad_(False)
        except Exception:
            pass

    print("editor.device:", editor.device)
    print("UNet device:", next(editor.pipe.unet.parameters()).device)

    rows = []

    for path in tqdm(audio_files, desc="Inversion baseline"):
        t0 = time.time()

        try:
            with torch.inference_mode():
                wav, sr = editor.load_audio(str(path))
                z0 = editor.encode_audio_to_z0(wav, sr)
                zT = editor.invert(z0, invert_prompt="")
                z0_hat = editor.sanity_forward(zT, invert_prompt="")
                wav_rec = editor.decode_latents(z0_hat, wav_len=len(wav))

            cos = logmel_cosine_similarity(wav, wav_rec, sr)

            rows.append({
                "filename": path.name,
                "path": str(path),
                "steps": args.steps,
                "cosine_similarity_spec": cos,
                "runtime_sec": time.time() - t0,
                "status": "ok",
                "error": "",
            })

            print(f"[OK] {path.name}: cos={cos:.6f}")

        except Exception as e:
            rows.append({
                "filename": path.name,
                "path": str(path),
                "steps": args.steps,
                "cosine_similarity_spec": np.nan,
                "runtime_sec": time.time() - t0,
                "status": "error",
                "error": repr(e),
            })
            print(f"[ERROR] {path.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    ok = df[df["status"] == "ok"].copy()

    summary = {
        "n_total": int(len(df)),
        "n_ok": int(len(ok)),
        "steps": int(args.steps),
        "mean_cosine_similarity_spec": float(ok["cosine_similarity_spec"].mean()) if len(ok) else None,
        "median_cosine_similarity_spec": float(ok["cosine_similarity_spec"].median()) if len(ok) else None,
        "q25_cosine_similarity_spec": float(ok["cosine_similarity_spec"].quantile(0.25)) if len(ok) else None,
        "q75_cosine_similarity_spec": float(ok["cosine_similarity_spec"].quantile(0.75)) if len(ok) else None,
        "mean_runtime_sec": float(ok["runtime_sec"].mean()) if len(ok) else None,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved CSV:  {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()