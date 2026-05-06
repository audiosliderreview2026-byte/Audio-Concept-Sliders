#!/usr/bin/env python
"""
Plot Experiment 3 identity-loop figures and optional random-pair sanity baseline.

This script is a CLI version of the XP3 notebook. It avoids hard-coded paths so it
can be stored in a git repository and reused on cluster/local machines.

Typical usage:

1) Raw loop-consistency figure:
python plot_xp3_identity_loop.py \
  --results_csv identity_loop_results.csv \
  --baseline_json inversion_identity_baseline_200steps_summary.json \
  --out_dir identity_loop_figures \
  --plot_type raw \
  --random_pair_baseline 0.810 \
  --show_inversion_baseline \
  --show_random_pair_baseline

2) Normalized identity-loss figure:
python plot_xp3_identity_loop.py \
  --results_csv identity_loop_results.csv \
  --baseline_json inversion_identity_baseline_200steps_summary.json \
  --out_dir identity_loop_figures \
  --plot_type normalized \
  --random_pair_baseline 0.810

3) Random-pair sanity check:
python plot_xp3_identity_loop.py \
  --plot_type random_pairs \
  --footstep_dir path/to/footstep_wavs \
  --out_dir identity_loop_figures \
  --n_random_pairs 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


PERTURBATION_ORDER_DEFAULT = ["speech", "guitar", "piano", "footsteps"]
AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make XP3 identity-loop figures and random-pair sanity checks."
    )

    p.add_argument(
        "--plot_type",
        choices=["raw", "normalized", "random_pairs", "all"],
        default="raw",
        help=(
            "raw: cosine similarity curves; normalized: normalized identity loss; "
            "random_pairs: compute random-pair footstep baseline; all: raw+normalized and random pairs if --footstep_dir is given."
        ),
    )

    # Main identity-loop inputs
    p.add_argument("--results_csv", type=Path, default=None, help="CSV produced by identity-loop experiment.")
    p.add_argument(
        "--baseline_json",
        type=Path,
        default=None,
        help="JSON summary from inversion-only baseline; must contain median_cosine_similarity_spec.",
    )
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory for figures and summary CSVs.")

    # Figure options
    p.add_argument("--include_footsteps", action="store_true", help="Include footsteps perturbation curve.")
    p.add_argument("--show_inversion_baseline", action="store_true", help="Show inversion-only baseline on raw plot.")
    p.add_argument("--show_random_pair_baseline", action="store_true", help="Show random-pair baseline on raw plot.")
    p.add_argument(
        "--random_pair_baseline",
        type=float,
        default=None,
        help="Median random-pair log-mel cosine similarity, e.g. 0.810.",
    )
    p.add_argument(
        "--perturbation_order",
        nargs="+",
        default=PERTURBATION_ORDER_DEFAULT,
        help="Order of perturbation curves in plots and legend.",
    )
    p.add_argument("--raw_ylim", nargs=2, type=float, default=None, help="Y-axis limits for raw plot, e.g. --raw_ylim 0.93 1.0")
    p.add_argument(
        "--normalized_ylim",
        nargs=2,
        type=float,
        default=None,
        help="Y-axis limits for normalized-loss plot.",
    )
    p.add_argument("--fig_width", type=float, default=10.5)
    p.add_argument("--fig_height", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=250)
    p.add_argument("--no_iqr", action="store_true", help="Do not draw IQR shading.")

    # Random pair sanity-check options
    p.add_argument("--footstep_dir", type=Path, default=None, help="Folder of footstep sounds for random-pair baseline.")
    p.add_argument("--n_random_pairs", type=int, default=500)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--target_duration_s", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--n_mels", type=int, default=64)

    return p.parse_args()


def list_audio_files(audio_dir: Path, extensions: Iterable[str] = AUDIO_EXTS) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])


def load_identity_results(results_csv: Path, include_footsteps: bool) -> pd.DataFrame:
    if results_csv is None:
        raise ValueError("--results_csv is required for raw/normalized plots.")
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    df = pd.read_csv(results_csv)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df["slider_movement"] = pd.to_numeric(df["slider_movement"], errors="coerce")
    df["cosine_similarity_spec"] = pd.to_numeric(df["cosine_similarity_spec"], errors="coerce")
    df = df.dropna(subset=["slider_movement", "cosine_similarity_spec", "perturbation_type"])

    if not include_footsteps:
        df = df[df["perturbation_type"] != "footsteps"].copy()

    if len(df) == 0:
        raise RuntimeError("No valid identity-loop rows after filtering.")

    return df


def load_inversion_baseline(baseline_json: Path | None) -> float | None:
    if baseline_json is None:
        return None
    if not baseline_json.exists():
        raise FileNotFoundError(baseline_json)
    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    return float(baseline["median_cosine_similarity_spec"])


def ordered_plot_types(summary: pd.DataFrame, perturbation_order: list[str]) -> list[str]:
    available = list(summary["perturbation_type"].unique())
    plot_types = [p for p in perturbation_order if p in available]
    plot_types += [p for p in available if p not in plot_types]
    return plot_types


def plot_raw_identity_loop(args: argparse.Namespace) -> None:
    df = load_identity_results(args.results_csv, args.include_footsteps)
    baseline_value = load_inversion_baseline(args.baseline_json)

    summary = (
        df.groupby(["perturbation_type", "slider_movement"])["cosine_similarity_spec"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            n="count",
        )
        .reset_index()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "identity_loop_similarity_summary.csv", index=False)

    plt.figure(figsize=(args.fig_width, args.fig_height))

    for ptype in ordered_plot_types(summary, args.perturbation_order):
        sub = summary[summary["perturbation_type"] == ptype].sort_values("slider_movement")
        x = sub["slider_movement"].to_numpy()
        y = sub["median"].to_numpy()
        y_low = sub["q25"].to_numpy()
        y_high = sub["q75"].to_numpy()

        plt.plot(x, y, marker="o", linewidth=2.3, label=ptype)
        if not args.no_iqr:
            plt.fill_between(x, y_low, y_high, alpha=0.18)

    if args.show_inversion_baseline:
        if baseline_value is None:
            raise ValueError("--show_inversion_baseline requires --baseline_json.")
        plt.axhline(
            baseline_value,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"inversion-only ({baseline_value:.3f})",
        )

    if args.show_random_pair_baseline:
        if args.random_pair_baseline is None:
            raise ValueError("--show_random_pair_baseline requires --random_pair_baseline.")
        if args.raw_ylim is not None and args.random_pair_baseline < args.raw_ylim[0]:
            plt.text(
                0.02,
                0.03,
                f"random footstep-pair baseline = {args.random_pair_baseline:.3f}\n(below plotted range)",
                transform=plt.gca().transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
            )
        else:
            plt.axhline(
                args.random_pair_baseline,
                linestyle=":",
                linewidth=2,
                color="gray",
                label=f"random footstep pairs ({args.random_pair_baseline:.3f})",
            )

    plt.axvline(0, linestyle=":", linewidth=1, color="black", alpha=0.5)
    plt.xlabel("Slider strength")
    plt.ylabel("Loop consistency\nlog-mel spectrogram cosine similarity")
    plt.title("Identity preservation under additive perturbations")
    plt.grid(True, alpha=0.3)

    if args.raw_ylim is not None:
        plt.ylim(*args.raw_ylim)

    plt.legend(title="Perturbation", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()

    suffix = "with_footsteps" if args.include_footsteps else "no_footsteps"
    if args.show_inversion_baseline or args.show_random_pair_baseline:
        suffix += "_with_refs"

    out_png = args.out_dir / f"identity_loop_similarity_{suffix}.png"
    out_pdf = args.out_dir / f"identity_loop_similarity_{suffix}.pdf"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    if baseline_value is not None:
        print(f"Inversion baseline median: {baseline_value:.6f}")


def plot_normalized_identity_loss(args: argparse.Namespace) -> None:
    if args.random_pair_baseline is None:
        raise ValueError("Normalized identity loss requires --random_pair_baseline.")

    df = load_identity_results(args.results_csv, args.include_footsteps)
    inversion_baseline = load_inversion_baseline(args.baseline_json)
    if inversion_baseline is None:
        raise ValueError("Normalized identity loss requires --baseline_json.")

    denom = inversion_baseline - args.random_pair_baseline
    if denom <= 0:
        raise ValueError("Invalid baselines: inversion baseline must be larger than random-pair baseline.")

    df["normalized_identity_loss"] = (inversion_baseline - df["cosine_similarity_spec"]) / denom

    summary = (
        df.groupby(["perturbation_type", "slider_movement"])["normalized_identity_loss"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            n="count",
        )
        .reset_index()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "normalized_identity_loss_summary.csv", index=False)

    plt.figure(figsize=(args.fig_width, args.fig_height))

    for ptype in ordered_plot_types(summary, args.perturbation_order):
        sub = summary[summary["perturbation_type"] == ptype].sort_values("slider_movement")
        x = sub["slider_movement"].to_numpy()
        y = sub["median"].to_numpy()
        y_low = sub["q25"].to_numpy()
        y_high = sub["q75"].to_numpy()

        plt.plot(x, y, marker="o", linewidth=2.3, label=ptype)
        if not args.no_iqr:
            plt.fill_between(x, y_low, y_high, alpha=0.18)

    plt.axhline(0.0, linestyle="--", linewidth=1.8, color="black", label="inversion-only")
    plt.axhline(1.0, linestyle=":", linewidth=1.8, color="gray", label="random footstep pairs")
    plt.axvline(0, linestyle=":", linewidth=1, color="black", alpha=0.5)

    plt.xlabel("Slider strength")
    plt.ylabel("Normalized identity loss")
    plt.title("Slider-induced identity loss relative to reconstruction and random-pair baselines")
    plt.grid(True, alpha=0.3)

    if args.normalized_ylim is not None:
        plt.ylim(*args.normalized_ylim)

    plt.legend(title="Perturbation", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()

    suffix = "with_footsteps" if args.include_footsteps else "no_footsteps"
    out_png = args.out_dir / f"normalized_identity_loss_{suffix}.png"
    out_pdf = args.out_dir / f"normalized_identity_loss_{suffix}.pdf"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Inversion baseline:   {inversion_baseline:.6f}")
    print(f"Random-pair baseline: {args.random_pair_baseline:.6f}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print("\nMedian normalized identity loss by perturbation:")
    print(
        summary.groupby("perturbation_type")["median"]
        .median()
        .sort_values()
        .to_frame("median_normalized_identity_loss")
    )


def load_mono_fixed(path: Path, target_sr: int, duration_s: float) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)

    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    n = int(round(duration_s * target_sr))
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    else:
        wav = wav[:n]

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak

    return wav


def logmel_vector(wav: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int, eps: float = 1e-8) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    return np.log(mel + eps).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def compute_random_pair_baseline(args: argparse.Namespace) -> float:
    if args.footstep_dir is None:
        raise ValueError("--footstep_dir is required for random_pairs mode.")
    if not args.footstep_dir.exists():
        raise FileNotFoundError(args.footstep_dir)

    rng = random.Random(args.seed)
    files = list_audio_files(args.footstep_dir)
    print(f"Found {len(files)} files in {args.footstep_dir}")

    if len(files) < 2:
        raise RuntimeError("Need at least two files to compute random pair similarities.")

    vectors = {}
    for path in tqdm(files, desc="Computing log-mel vectors"):
        wav = load_mono_fixed(path, args.target_sr, args.target_duration_s)
        vectors[path] = logmel_vector(wav, args.target_sr, args.n_fft, args.hop_length, args.n_mels)

    rows = []
    for i in tqdm(range(args.n_random_pairs), desc="Random pairs"):
        a, b = rng.sample(files, 2)
        sim = cosine(vectors[a], vectors[b])
        rows.append(
            {
                "pair_index": i,
                "path_a": str(a),
                "path_b": str(b),
                "filename_a": a.name,
                "filename_b": b.name,
                "cosine_similarity_spec": sim,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_df = pd.DataFrame(rows)
    out_csv = args.out_dir / "random_footstep_pair_logmel_cosine.csv"
    pair_df.to_csv(out_csv, index=False)

    summary = pair_df["cosine_similarity_spec"].describe(percentiles=[0.25, 0.5, 0.75])
    median = float(pair_df["cosine_similarity_spec"].median())
    mean = float(pair_df["cosine_similarity_spec"].mean())

    out_json = args.out_dir / "random_footstep_pair_logmel_cosine_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_random_pairs": args.n_random_pairs,
                "target_sr": args.target_sr,
                "target_duration_s": args.target_duration_s,
                "median_cosine_similarity_spec": median,
                "mean_cosine_similarity_spec": mean,
                "summary": {k: float(v) for k, v in summary.to_dict().items()},
            },
            f,
            indent=2,
        )

    print(summary)
    print(f"\nMedian random-pair similarity: {median:.6f}")
    print(f"Mean random-pair similarity:   {mean:.6f}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    return median


def main() -> None:
    args = parse_args()

    if args.plot_type in {"raw", "all"}:
        plot_raw_identity_loop(args)

    if args.plot_type in {"normalized", "all"}:
        plot_normalized_identity_loss(args)

    if args.plot_type in {"random_pairs", "all"}:
        if args.footstep_dir is not None:
            compute_random_pair_baseline(args)
        elif args.plot_type == "random_pairs":
            raise ValueError("--footstep_dir is required for --plot_type random_pairs.")
        else:
            print("[INFO] Skipping random-pair baseline because --footstep_dir was not provided.")


if __name__ == "__main__":
    main()
