#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score existing slider-edited WAVs under a CLAP+PCA+GMM model."
    )

    p.add_argument("--edited_wavs_dir", required=True, help="Folder containing already generated edited WAVs.")
    p.add_argument("--footstep_baseline_dir", required=True, help="Folder of real footstep WAVs used as baseline.")

    p.add_argument("--pca_path", required=True)
    p.add_argument("--gmm_path", required=True)
    p.add_argument("--clap_model_path", required=True)
    p.add_argument("--clap_local_files_only", action="store_true")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".mp3"])

    p.add_argument("--clap_batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max_edited_files", type=int, default=None)
    p.add_argument("--max_baseline_files", type=int, default=None)
    p.add_argument("--overwrite_scores", action="store_true")

    # Strength parsing from filenames such as:
    # something_strength_m5d000.wav, something_strength_p2d368.wav
    p.add_argument("--strength_token", default="_strength_")

    return p.parse_args()


def list_audio_files(audio_dir: Path, extensions: List[str], max_files=None) -> List[Path]:
    exts = {e.lower() for e in extensions}
    files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])
    if max_files is not None:
        files = files[:max_files]
    return files


def parse_strength_from_filename(path: Path, strength_token: str = "_strength_") -> float:
    stem = path.stem
    if strength_token not in stem:
        raise ValueError(f"Could not find strength token {strength_token!r} in filename: {path.name}")

    s = stem.split(strength_token)[-1]
    s = s.replace("p", "+").replace("m", "-").replace("d", ".")
    return float(s)


class ClapGMMScorer:
    def __init__(self, clap_model_path: str, pca_path: str, gmm_path: str, device: str, local_files_only: bool):
        self.device = device

        self.processor = ClapProcessor.from_pretrained(
            clap_model_path,
            local_files_only=local_files_only,
        )
        self.model = ClapModel.from_pretrained(
            clap_model_path,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()

        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        with open(gmm_path, "rb") as f:
            self.gmm = pickle.load(f)

        self.target_sr = int(self.processor.feature_extractor.sampling_rate)
        self.pca_dim = int(getattr(self.pca, "n_components_", getattr(self.pca, "n_components", -1)))

    def _load_for_clap(self, path: Path) -> np.ndarray:
        wav, sr = sf.read(str(path), always_2d=False)

        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)

        wav = wav.astype(np.float32)

        if wav.size == 0:
            raise ValueError(f"Empty audio: {path}")

        peak = float(np.max(np.abs(wav)))
        if peak > 1.0:
            wav = wav / peak

        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        return wav.astype(np.float32)

    @torch.no_grad()
    def embed_files(self, paths: List[Path], batch_size: int = 16) -> Tuple[np.ndarray, List[Path]]:
        all_embs = []
        kept = []

        for start in tqdm(range(0, len(paths), batch_size), desc="CLAP embedding batches"):
            batch_paths = paths[start:start + batch_size]

            audios = []
            valid_paths = []

            for path in batch_paths:
                try:
                    audios.append(self._load_for_clap(path))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"[WARN] skipping {path}: {e}")

            if not audios:
                continue

            inputs = self.processor(
                audios=audios,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            emb = self.model.get_audio_features(**inputs)
            emb = torch.nn.functional.normalize(emb, dim=-1)

            all_embs.append(emb.cpu().numpy())
            kept.extend(valid_paths)

        if not all_embs:
            return np.empty((0, 0), dtype=np.float32), []

        return np.concatenate(all_embs, axis=0), kept

    def score_files(self, paths: List[Path], batch_size: int = 16) -> pd.DataFrame:
        emb, kept = self.embed_files(paths, batch_size=batch_size)

        if len(kept) == 0:
            return pd.DataFrame(columns=["path", "filename", "log_likelihood", "avg_log_likelihood"])

        z = self.pca.transform(emb)
        ll = self.gmm.score_samples(z)

        return pd.DataFrame({
            "path": [str(p) for p in kept],
            "filename": [p.name for p in kept],
            "log_likelihood": ll,
            "avg_log_likelihood": ll / z.shape[1],
        })


def plot_curve(grouped: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    baseline_median = float(baseline_df["log_likelihood"].median())
    baseline_q25 = float(baseline_df["log_likelihood"].quantile(0.25))
    baseline_q75 = float(baseline_df["log_likelihood"].quantile(0.75))

    x = grouped["strength"].to_numpy()
    y = grouped["median_log_likelihood"].to_numpy()
    y_low = grouped["q25_log_likelihood"].to_numpy()
    y_high = grouped["q75_log_likelihood"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, label="Edited outputs median")
    plt.fill_between(x, y_low, y_high, alpha=0.2, label="Edited outputs IQR")

    plt.axhline(
        baseline_median,
        linestyle="--",
        linewidth=2,
        color="black",
        label="Real footsteps median",
    )
    plt.axhspan(
        baseline_q25,
        baseline_q75,
        alpha=0.12,
        color="black",
        label="Real footsteps IQR",
    )

    plt.xlabel("Slider strength")
    plt.ylabel("GMM log-likelihood in CLAP-PCA space")
    plt.title("Footstep GMM likelihood vs slider strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "gmm_likelihood_vs_slider_strength_footstep_ref.png", dpi=250)
    plt.savefig(output_dir / "gmm_likelihood_vs_slider_strength_footstep_ref.pdf")
    plt.close()

    # Normalized per PCA dimension
    baseline_median = float(baseline_df["avg_log_likelihood"].median())
    baseline_q25 = float(baseline_df["avg_log_likelihood"].quantile(0.25))
    baseline_q75 = float(baseline_df["avg_log_likelihood"].quantile(0.75))

    y = grouped["median_avg_log_likelihood"].to_numpy()
    y_low = grouped["q25_avg_log_likelihood"].to_numpy()
    y_high = grouped["q75_avg_log_likelihood"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, label="Edited outputs median")
    plt.fill_between(x, y_low, y_high, alpha=0.2, label="Edited outputs IQR")

    plt.axhline(
        baseline_median,
        linestyle="--",
        linewidth=2,
        color="black",
        label="Real footsteps median",
    )
    plt.axhspan(
        baseline_q25,
        baseline_q75,
        alpha=0.12,
        color="black",
        label="Real footsteps IQR",
    )

    plt.xlabel("Slider strength")
    plt.ylabel("Average log-likelihood per PCA dimension")
    plt.title("Normalized footstep GMM likelihood vs slider strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "gmm_avg_likelihood_vs_slider_strength_footstep_ref.png", dpi=250)
    plt.savefig(output_dir / "gmm_avg_likelihood_vs_slider_strength_footstep_ref.pdf")
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edited_score_csv = output_dir / "gmm_scores_existing_edits.csv"
    baseline_score_csv = output_dir / "gmm_scores_footstep_baseline.csv"
    strength_csv = output_dir / "gmm_scores_by_strength_footstep_ref.csv"

    edited_files = list_audio_files(
        Path(args.edited_wavs_dir),
        args.extensions,
        args.max_edited_files,
    )
    baseline_files = list_audio_files(
        Path(args.footstep_baseline_dir),
        args.extensions,
        args.max_baseline_files,
    )

    print(f"Edited files:   {len(edited_files)}")
    print(f"Baseline files: {len(baseline_files)}")

    if len(edited_files) == 0:
        raise RuntimeError(f"No edited audio files found in {args.edited_wavs_dir}")
    if len(baseline_files) == 0:
        raise RuntimeError(f"No baseline audio files found in {args.footstep_baseline_dir}")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    scorer = ClapGMMScorer(
        clap_model_path=args.clap_model_path,
        pca_path=args.pca_path,
        gmm_path=args.gmm_path,
        device=args.device,
        local_files_only=args.clap_local_files_only,
    )

    if baseline_score_csv.exists() and not args.overwrite_scores:
        print(f"Loading baseline scores: {baseline_score_csv}")
        baseline_df = pd.read_csv(baseline_score_csv)
    else:
        baseline_df = scorer.score_files(baseline_files, batch_size=args.clap_batch_size)
        baseline_df["set"] = "real_footsteps_baseline"
        baseline_df.to_csv(baseline_score_csv, index=False)

    if edited_score_csv.exists() and not args.overwrite_scores:
        print(f"Loading edited scores: {edited_score_csv}")
        edited_df = pd.read_csv(edited_score_csv)
    else:
        edited_df = scorer.score_files(edited_files, batch_size=args.clap_batch_size)
        edited_df = edited_df.rename(columns={"path": "edited_path", "filename": "edited_filename"})

        strengths = []
        for p in edited_df["edited_path"]:
            strengths.append(parse_strength_from_filename(Path(p), args.strength_token))

        edited_df["strength"] = strengths
        edited_df.to_csv(edited_score_csv, index=False)

    grouped = edited_df.groupby("strength", as_index=False).agg(
        n=("log_likelihood", "count"),
        mean_log_likelihood=("log_likelihood", "mean"),
        std_log_likelihood=("log_likelihood", "std"),
        median_log_likelihood=("log_likelihood", "median"),
        q25_log_likelihood=("log_likelihood", lambda x: x.quantile(0.25)),
        q75_log_likelihood=("log_likelihood", lambda x: x.quantile(0.75)),
        mean_avg_log_likelihood=("avg_log_likelihood", "mean"),
        std_avg_log_likelihood=("avg_log_likelihood", "std"),
        median_avg_log_likelihood=("avg_log_likelihood", "median"),
        q25_avg_log_likelihood=("avg_log_likelihood", lambda x: x.quantile(0.25)),
        q75_avg_log_likelihood=("avg_log_likelihood", lambda x: x.quantile(0.75)),
    )

    grouped["sem_log_likelihood"] = grouped["std_log_likelihood"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["sem_avg_log_likelihood"] = grouped["std_avg_log_likelihood"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped = grouped.sort_values("strength")
    grouped.to_csv(strength_csv, index=False)

    summary = {
        "n_edited_files": int(len(edited_df)),
        "n_baseline_files": int(len(baseline_df)),
        "baseline_median_log_likelihood": float(baseline_df["log_likelihood"].median()),
        "baseline_q25_log_likelihood": float(baseline_df["log_likelihood"].quantile(0.25)),
        "baseline_q75_log_likelihood": float(baseline_df["log_likelihood"].quantile(0.75)),
        "baseline_mean_log_likelihood": float(baseline_df["log_likelihood"].mean()),
        "baseline_std_log_likelihood": float(baseline_df["log_likelihood"].std()),
        "pca_dim": scorer.pca_dim,
    }

    with open(output_dir / "summary_footstep_ref.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_curve(grouped, baseline_df, output_dir)

    print("\nSaved:")
    print(" ", edited_score_csv)
    print(" ", baseline_score_csv)
    print(" ", strength_csv)
    print(" ", output_dir / "gmm_likelihood_vs_slider_strength_footstep_ref.png")
    print(" ", output_dir / "gmm_avg_likelihood_vs_slider_strength_footstep_ref.png")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()