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
        description="Score existing edited WAVs under CLAP+PCA+GMM and plot GMM log-likelihood."
    )

    p.add_argument("--edited_wavs_dir", required=True)

    p.add_argument("--baseline_audioset_dir", required=True)
    p.add_argument("--baseline_plugin_dir", required=True)
    p.add_argument("--baseline_synthetic_dir", required=True)

    p.add_argument("--pca_path", required=True)
    p.add_argument("--gmm_path", required=True)
    p.add_argument("--clap_model_path", required=True)
    p.add_argument("--clap_local_files_only", action="store_true")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".mp3"])

    p.add_argument("--max_edited_files", type=int, default=None)
    p.add_argument("--max_baseline_files_per_set", type=int, default=None)

    p.add_argument("--clap_batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--strength_token", default="_strength_")
    p.add_argument("--overwrite_scores", action="store_true")
    p.add_argument("--target_duration_s", type=float, default=10.0)

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
    def __init__(
        self,
        clap_model_path: str,
        pca_path: str,
        gmm_path: str,
        device: str,
        local_files_only: bool,
        target_duration_s: float | None = 10.0,
    ):
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
        self.target_duration_s = target_duration_s

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

        if self.target_duration_s is not None:
            n = int(round(self.target_duration_s * self.target_sr))
            if len(wav) < n:
                wav = np.pad(wav, (0, n - len(wav)))
            else:
                wav = wav[:n]

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

    def score_files(self, paths: List[Path], batch_size: int = 16, set_name: str = "") -> pd.DataFrame:
        emb, kept = self.embed_files(paths, batch_size=batch_size)

        if len(kept) == 0:
            return pd.DataFrame(columns=["set", "path", "filename", "log_likelihood", "avg_log_likelihood"])

        z = self.pca.transform(emb)
        ll = self.gmm.score_samples(z)

        return pd.DataFrame({
            "set": set_name,
            "path": [str(p) for p in kept],
            "filename": [p.name for p in kept],
            "log_likelihood": ll,
            "avg_log_likelihood": ll / z.shape[1],
        })


def plot_curve(grouped: pd.DataFrame, baseline_summary: pd.DataFrame, output_dir: Path):
    colors = {
        "edited": "tab:blue",
        "audioset": "tab:green",
        "plugin": "tab:orange",
        "synthetic": "tab:red",
    }

    linestyles = {
        "audioset": "--",
        "plugin": "-.",
        "synthetic": ":",
    }

    plt.figure(figsize=(8.8, 5))

    x = grouped["strength"].to_numpy()
    y = grouped["median_log_likelihood"].to_numpy()

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2.8,
        color=colors["edited"],
        label="Edited outputs",
    )

    for set_name in ["audioset", "plugin", "synthetic"]:
        sub = baseline_summary[baseline_summary["set"] == set_name]
        if len(sub) == 0:
            print(f"[WARN] No baseline summary for {set_name}; skipping line.")
            continue

        val = float(sub["median_log_likelihood"].iloc[0])

        plt.axhline(
            val,
            linestyle=linestyles[set_name],
            linewidth=2.3,
            color=colors[set_name],
            label=f"{set_name} baseline ({val:.1f})",
        )

    plt.xlabel("Slider strength")
    plt.ylabel("GMM log-likelihood in CLAP-PCA space")
    plt.title("Footstep GMM log-likelihood vs slider strength")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    plt.savefig(output_dir / "gmm_loglikelihood_vs_slider_strength.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "gmm_loglikelihood_vs_slider_strength.pdf", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edited_files = list_audio_files(
        Path(args.edited_wavs_dir),
        args.extensions,
        args.max_edited_files,
    )

    baseline_sets = {
        "audioset": list_audio_files(Path(args.baseline_audioset_dir), args.extensions, args.max_baseline_files_per_set),
        "plugin": list_audio_files(Path(args.baseline_plugin_dir), args.extensions, args.max_baseline_files_per_set),
        "synthetic": list_audio_files(Path(args.baseline_synthetic_dir), args.extensions, args.max_baseline_files_per_set),
    }

    print(f"Edited files: {len(edited_files)}")
    for name, files in baseline_sets.items():
        print(f"Baseline {name}: {len(files)} files")
        if len(files) == 0:
            print(f"[WARN] No supported audio files found for baseline '{name}'.")

    if len(edited_files) == 0:
        raise RuntimeError(f"No edited files found in {args.edited_wavs_dir}")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    edited_score_csv = output_dir / "gmm_scores_existing_edits.csv"
    baseline_score_csv = output_dir / "gmm_scores_baselines.csv"
    strength_csv = output_dir / "gmm_scores_by_strength.csv"
    baseline_summary_csv = output_dir / "gmm_baseline_summary.csv"

    scorer = ClapGMMScorer(
        clap_model_path=args.clap_model_path,
        pca_path=args.pca_path,
        gmm_path=args.gmm_path,
        device=args.device,
        local_files_only=args.clap_local_files_only,
        target_duration_s=args.target_duration_s,
    )

    if edited_score_csv.exists() and baseline_score_csv.exists() and not args.overwrite_scores:
        print("Loading existing scores. Use --overwrite_scores to recompute baselines.")
        edited_df = pd.read_csv(edited_score_csv)
        baseline_df = pd.read_csv(baseline_score_csv)
    else:
        edited_df = scorer.score_files(
            edited_files,
            batch_size=args.clap_batch_size,
            set_name="edited",
        )

        edited_df = edited_df.rename(columns={"path": "edited_path", "filename": "edited_filename"})
        edited_df["strength"] = [
            parse_strength_from_filename(Path(p), args.strength_token)
            for p in edited_df["edited_path"]
        ]

        baseline_dfs = []
        for set_name, files in baseline_sets.items():
            if len(files) == 0:
                print(f"[WARN] Skipping empty baseline set: {set_name}")
                continue

            df_set = scorer.score_files(
                files,
                batch_size=args.clap_batch_size,
                set_name=set_name,
            )
            baseline_dfs.append(df_set)

        if len(baseline_dfs) == 0:
            raise RuntimeError("No valid baseline scores computed.")

        baseline_df = pd.concat(baseline_dfs, ignore_index=True)

        edited_df.to_csv(edited_score_csv, index=False)
        baseline_df.to_csv(baseline_score_csv, index=False)

    grouped = edited_df.groupby("strength", as_index=False).agg(
        n=("log_likelihood", "count"),
        median_log_likelihood=("log_likelihood", "median"),
        mean_log_likelihood=("log_likelihood", "mean"),
        q25_log_likelihood=("log_likelihood", lambda x: x.quantile(0.25)),
        q75_log_likelihood=("log_likelihood", lambda x: x.quantile(0.75)),
        median_avg_log_likelihood=("avg_log_likelihood", "median"),
        mean_avg_log_likelihood=("avg_log_likelihood", "mean"),
    ).sort_values("strength")

    grouped.to_csv(strength_csv, index=False)

    baseline_summary = (
        baseline_df.groupby("set", as_index=False)
        .agg(
            n=("log_likelihood", "count"),
            median_log_likelihood=("log_likelihood", "median"),
            mean_log_likelihood=("log_likelihood", "mean"),
            q25_log_likelihood=("log_likelihood", lambda x: x.quantile(0.25)),
            q75_log_likelihood=("log_likelihood", lambda x: x.quantile(0.75)),
            median_avg_log_likelihood=("avg_log_likelihood", "median"),
            mean_avg_log_likelihood=("avg_log_likelihood", "mean"),
        )
    )

    baseline_summary.to_csv(baseline_summary_csv, index=False)

    summary = {
        "n_edited_files": int(len(edited_df)),
        "n_baseline_files": int(len(baseline_df)),
        "baseline_sets_present": sorted(baseline_df["set"].unique().tolist()),
        "pca_dim": scorer.pca_dim,
        "metric": "GMM log-likelihood in CLAP-PCA space",
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_curve(grouped, baseline_summary, output_dir)

    print("\nSaved:")
    print(" ", edited_score_csv)
    print(" ", baseline_score_csv)
    print(" ", strength_csv)
    print(" ", baseline_summary_csv)
    print(" ", output_dir / "gmm_loglikelihood_vs_slider_strength.png")
    print(" ", output_dir / "gmm_loglikelihood_vs_slider_strength.pdf")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()