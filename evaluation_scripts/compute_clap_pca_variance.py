#!/usr/bin/env python
# compute_clap_pca_variance.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
from sklearn.decomposition import PCA


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract CLAP audio embeddings from WAV files and plot PCA explained variance."
    )

    p.add_argument("--audio_dir", required=True, help="Folder containing reference audio files.")
    p.add_argument("--clap_model_path", required=True, help="Local path to CLAP checkpoint.")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".mp3"])
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--pca_max_components", type=int, default=256)
    p.add_argument("--variance_thresholds", nargs="+", type=float, default=[0.8, 0.9, 0.95, 0.99])

    return p.parse_args()


def list_audio_files(audio_dir: Path, extensions):
    extensions = {e.lower() for e in extensions}
    files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions]
    return sorted(files)


def load_audio_for_clap(path: Path, target_sr: int):
    wav, sr = sf.read(str(path), always_2d=False)

    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if wav.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / peak

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    return wav


@torch.no_grad()
def extract_clap_embeddings(files, processor, model, device, batch_size: int):
    target_sr = int(processor.feature_extractor.sampling_rate)
    all_embs = []
    kept_files = []

    for start in tqdm(range(0, len(files), batch_size), desc="CLAP batches"):
        batch_files = files[start:start + batch_size]

        audios = []
        valid_files = []

        for path in batch_files:
            try:
                wav = load_audio_for_clap(path, target_sr=target_sr)
                audios.append(wav)
                valid_files.append(path)
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

        if not audios:
            continue

        inputs = processor(
            audios=audios,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        emb = model.get_audio_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)

        all_embs.append(emb.cpu().numpy())
        kept_files.extend(valid_files)

    if not all_embs:
        raise RuntimeError("No embeddings extracted.")

    return np.concatenate(all_embs, axis=0), kept_files


def main():
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(audio_dir, args.extensions)

    if args.max_files is not None:
        files = files[:args.max_files]

    print(f"Found {len(files)} audio files.")

    processor = ClapProcessor.from_pretrained(
        args.clap_model_path,
        local_files_only=args.local_files_only,
    )
    model = ClapModel.from_pretrained(
        args.clap_model_path,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()

    embeddings, kept_files = extract_clap_embeddings(
        files=files,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("Embedding matrix:", embeddings.shape)

    np.save(output_dir / "clap_embeddings.npy", embeddings)

    with open(output_dir / "files.txt", "w", encoding="utf-8") as f:
        for p in kept_files:
            f.write(str(p) + "\n")

    n_samples, n_dims = embeddings.shape
    n_components = min(args.pca_max_components, n_samples, n_dims)

    pca = PCA(n_components=n_components, svd_solver="full")
    z = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    np.save(output_dir / "pca_embeddings.npy", z)
    np.save(output_dir / "pca_explained_variance_ratio.npy", explained)
    np.save(output_dir / "pca_cumulative_explained_variance.npy", cumulative)

    summary = {
        "n_files": len(kept_files),
        "embedding_dim": int(n_dims),
        "pca_components": int(n_components),
        "threshold_components": {},
    }

    for th in args.variance_thresholds:
        k = int(np.searchsorted(cumulative, th) + 1)
        summary["threshold_components"][str(th)] = k
        print(f"{th:.2%} variance -> {k} PCA components")

    with open(output_dir / "pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot cumulative explained variance
    xs = np.arange(1, n_components + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, cumulative, marker="o", markersize=2)
    for th in args.variance_thresholds:
        plt.axhline(th, linestyle="--", linewidth=1)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance of CLAP audio embeddings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_cumulative_variance.png", dpi=200)
    plt.savefig(output_dir / "pca_cumulative_variance.pdf")
    plt.close()

    # Plot per-component explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(xs, explained, marker="o", markersize=2)
    plt.xlabel("PCA component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance per component")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance_per_component.png", dpi=200)
    plt.savefig(output_dir / "pca_explained_variance_per_component.pdf")
    plt.close()

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()