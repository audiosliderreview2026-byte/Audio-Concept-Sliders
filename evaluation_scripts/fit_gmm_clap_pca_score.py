#!/usr/bin/env python
# fit_gmm_clap_pca_score.py

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


def parse_args():
    p = argparse.ArgumentParser()

    # Reference set used to fit PCA + GMM, e.g. AudioCaps
    p.add_argument("--reference_audio_dir", required=True)

    # Audio files to score
    p.add_argument("--score_audio_dir", required=True)

    # Local CLAP checkpoint
    p.add_argument("--clap_model_path", required=True)
    p.add_argument("--local_files_only", action="store_true")

    # Output/cache
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="clap_pca_gmm")

    # PCA / GMM config
    p.add_argument("--pca_dim", type=int, required=True)
    p.add_argument("--n_components", type=int, default=32)
    p.add_argument("--covariance_type", default="diag", choices=["full", "tied", "diag", "spherical"])
    p.add_argument("--reg_covar", type=float, default=1e-5)
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--random_state", type=int, default=0)

    # Runtime
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_reference_files", type=int, default=None)
    p.add_argument("--max_score_files", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".mp3"])

    # Recompute controls
    p.add_argument("--force_recompute_embeddings", action="store_true")
    p.add_argument("--force_refit_gmm", action="store_true")

    return p.parse_args()


def list_audio_files(audio_dir: Path, extensions):
    extensions = {e.lower() for e in extensions}
    return sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions])


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

    for start in tqdm(range(0, len(files), batch_size), desc="CLAP embedding batches"):
        batch_files = files[start:start + batch_size]

        audios = []
        valid_files = []

        for path in batch_files:
            try:
                wav = load_audio_for_clap(path, target_sr)
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
        raise RuntimeError("No valid audio embeddings extracted.")

    return np.concatenate(all_embs, axis=0), kept_files


def save_file_list(paths, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


def load_or_compute_embeddings(
    files,
    cache_path: Path,
    filelist_path: Path,
    processor,
    model,
    device,
    batch_size,
    force=False,
):
    if cache_path.exists() and filelist_path.exists() and not force:
        print(f"Loading cached embeddings: {cache_path}")
        embeddings = np.load(cache_path)
        with open(filelist_path, "r", encoding="utf-8") as f:
            kept_files = [Path(line.strip()) for line in f if line.strip()]
        return embeddings, kept_files

    embeddings, kept_files = extract_clap_embeddings(
        files=files,
        processor=processor,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    np.save(cache_path, embeddings)
    save_file_list(kept_files, filelist_path)

    return embeddings, kept_files


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "models"
    emb_dir = output_dir / "embeddings"
    score_dir = output_dir / "scores"

    model_dir.mkdir(exist_ok=True)
    emb_dir.mkdir(exist_ok=True)
    score_dir.mkdir(exist_ok=True)

    reference_audio_dir = Path(args.reference_audio_dir)
    score_audio_dir = Path(args.score_audio_dir)

    ref_files = list_audio_files(reference_audio_dir, args.extensions)
    score_files = list_audio_files(score_audio_dir, args.extensions)

    if args.max_reference_files is not None:
        ref_files = ref_files[:args.max_reference_files]

    if args.max_score_files is not None:
        score_files = score_files[:args.max_score_files]

    print(f"Reference files: {len(ref_files)}")
    print(f"Score files:     {len(score_files)}")

    processor = ClapProcessor.from_pretrained(
        args.clap_model_path,
        local_files_only=args.local_files_only,
    )
    model = ClapModel.from_pretrained(
        args.clap_model_path,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()

    ref_embeddings, ref_kept_files = load_or_compute_embeddings(
        files=ref_files,
        cache_path=emb_dir / "reference_clap_embeddings.npy",
        filelist_path=emb_dir / "reference_files.txt",
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_recompute_embeddings,
    )

    score_embeddings, score_kept_files = load_or_compute_embeddings(
        files=score_files,
        cache_path=emb_dir / "score_clap_embeddings.npy",
        filelist_path=emb_dir / "score_files.txt",
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_recompute_embeddings,
    )

    print("Reference embedding shape:", ref_embeddings.shape)
    print("Score embedding shape:    ", score_embeddings.shape)

    if args.pca_dim > ref_embeddings.shape[1]:
        raise ValueError(f"pca_dim={args.pca_dim} > embedding_dim={ref_embeddings.shape[1]}")

    if args.pca_dim > len(ref_embeddings):
        raise ValueError(f"pca_dim={args.pca_dim} > n_reference_samples={len(ref_embeddings)}")

    tag = (
        f"{args.model_name}"
        f"_pca{args.pca_dim}"
        f"_gmm{args.n_components}"
        f"_{args.covariance_type}"
        f"_reg{args.reg_covar}"
    )

    pca_path = model_dir / f"{tag}_pca.pkl"
    gmm_path = model_dir / f"{tag}_gmm.pkl"
    metadata_path = model_dir / f"{tag}_metadata.json"

    if pca_path.exists() and gmm_path.exists() and not args.force_refit_gmm:
        print(f"Loading cached PCA: {pca_path}")
        print(f"Loading cached GMM: {gmm_path}")

        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(gmm_path, "rb") as f:
            gmm = pickle.load(f)

    else:
        print(f"Fitting PCA with dim={args.pca_dim}")
        pca = PCA(n_components=args.pca_dim, svd_solver="full", random_state=args.random_state)
        ref_z = pca.fit_transform(ref_embeddings)

        print(f"Fitting GMM with n_components={args.n_components}, covariance_type={args.covariance_type}")
        gmm = GaussianMixture(
            n_components=args.n_components,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            max_iter=args.max_iter,
            random_state=args.random_state,
            verbose=1,
        )
        gmm.fit(ref_z)

        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

        with open(gmm_path, "wb") as f:
            pickle.dump(gmm, f)

        metadata = {
            "reference_audio_dir": str(reference_audio_dir),
            "n_reference_files": len(ref_kept_files),
            "embedding_dim": int(ref_embeddings.shape[1]),
            "pca_dim": args.pca_dim,
            "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            "gmm_n_components": args.n_components,
            "gmm_covariance_type": args.covariance_type,
            "gmm_reg_covar": args.reg_covar,
            "gmm_converged": bool(gmm.converged_),
            "gmm_n_iter": int(gmm.n_iter_),
            "gmm_lower_bound": float(gmm.lower_bound_),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    score_z = pca.transform(score_embeddings)

    # score_samples returns log p(x) under the fitted GMM
    log_likelihoods = gmm.score_samples(score_z)

    df = pd.DataFrame({
        "path": [str(p) for p in score_kept_files],
        "filename": [p.name for p in score_kept_files],
        "log_likelihood": log_likelihoods,
        "avg_log_likelihood": log_likelihoods / args.pca_dim,
        "pca_dim": args.pca_dim,
        "gmm_n_components": args.n_components,
        "gmm_covariance_type": args.covariance_type,
    })

    csv_path = score_dir / f"{tag}_scores.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_scored_files": int(len(df)),
        "mean_log_likelihood": float(df["log_likelihood"].mean()),
        "std_log_likelihood": float(df["log_likelihood"].std()),
        "median_log_likelihood": float(df["log_likelihood"].median()),
        "mean_avg_log_likelihood": float(df["avg_log_likelihood"].mean()),
        "csv_path": str(csv_path),
        "pca_path": str(pca_path),
        "gmm_path": str(gmm_path),
    }

    with open(score_dir / f"{tag}_score_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(df.head())
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved scores to: {csv_path}")


if __name__ == "__main__":
    main()s