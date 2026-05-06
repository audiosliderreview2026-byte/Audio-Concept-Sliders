#!/usr/bin/env python
# clap_reference_barplot.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--neutral_dir", required=True)
    p.add_argument("--calm_dir", required=True)
    p.add_argument("--angry_dir", required=True)

    p.add_argument("--neutral_prompt", required=True)
    p.add_argument("--calm_prompt", required=True)
    p.add_argument("--angry_prompt", required=True)

    p.add_argument("--clap_model_path", required=True)
    p.add_argument("--clap_local_files_only", action="store_true")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--extensions", nargs="+", default=[".wav", ".flac", ".mp3"])
    p.add_argument("--max_files_per_set", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def list_audio_files(audio_dir: Path, extensions: List[str], max_files=None):
    exts = {e.lower() for e in extensions}
    files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in exts])
    if max_files is not None:
        files = files[:max_files]
    return files


def load_audio_for_clap(path: Path, target_sr: int):
    wav, sr = sf.read(str(path), always_2d=False)

    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if wav.size == 0:
        raise ValueError(f"Empty file: {path}")

    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        wav = wav / peak

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    return wav.astype(np.float32)


class CLAPScorer:
    def __init__(self, model_path: str, device: str, local_files_only: bool):
        self.device = device

        self.processor = ClapProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.model = ClapModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()

        self.target_sr = int(self.processor.feature_extractor.sampling_rate)

    @torch.no_grad()
    def encode_texts(self, prompts: Dict[str, str]):
        inputs = self.processor(
            text=list(prompts.values()),
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        emb = self.model.get_text_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)

        return {
            name: emb[i].detach().cpu()
            for i, name in enumerate(prompts.keys())
        }

    @torch.no_grad()
    def encode_audio_files(self, files: List[Path], batch_size: int):
        all_embs = []
        kept_files = []

        for start in tqdm(range(0, len(files), batch_size), desc="CLAP audio batches"):
            batch_files = files[start:start + batch_size]

            audios = []
            valid_files = []

            for path in batch_files:
                try:
                    audios.append(load_audio_for_clap(path, self.target_sr))
                    valid_files.append(path)
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

            all_embs.append(emb.cpu())
            kept_files.extend(valid_files)

        if not all_embs:
            return torch.empty(0), []

        return torch.cat(all_embs, dim=0), kept_files


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets = {
        "calm": Path(args.calm_dir),
        "neutral": Path(args.neutral_dir),
        "angry": Path(args.angry_dir),
    }
    
    prompts = {
        "calm": args.calm_prompt,
        "neutral": args.neutral_prompt,
        "angry": args.angry_prompt,
    }

    scorer = CLAPScorer(
        model_path=args.clap_model_path,
        device=args.device,
        local_files_only=args.clap_local_files_only,
    )

    text_embeds = scorer.encode_texts(prompts)

    all_rows = []

    for set_name, set_dir in sets.items():
        files = list_audio_files(set_dir, args.extensions, args.max_files_per_set)
        print(f"{set_name}: {len(files)} files")

        audio_embeds, kept_files = scorer.encode_audio_files(files, args.batch_size)

        if len(kept_files) == 0:
            continue

        for i, path in enumerate(kept_files):
            audio_emb = audio_embeds[i]

            for prompt_name, text_emb in text_embeds.items():
                sim = torch.nn.functional.cosine_similarity(
                    audio_emb[None, :],
                    text_emb[None, :],
                    dim=-1,
                ).item()

                all_rows.append({
                    "set": set_name,
                    "audio_path": str(path),
                    "audio_filename": path.name,
                    "prompt": prompt_name,
                    "prompt_text": prompts[prompt_name],
                    "clap_similarity": sim,
                })

    df = pd.DataFrame(all_rows)
    scores_path = output_dir / "clap_reference_scores_per_file.csv"
    df.to_csv(scores_path, index=False)

    summary = (
        df.groupby(["set", "prompt"])["clap_similarity"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    summary_path = output_dir / "clap_reference_scores_summary.csv"
    summary.to_csv(summary_path, index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # =========================
    # Plot
    # =========================

    set_order = ["calm", "neutral", "angry"]
    prompt_order = ["calm", "neutral", "angry"]

    x = np.arange(len(set_order))
    width = 0.24

    plt.figure(figsize=(8.5, 5))

    for j, prompt_name in enumerate(prompt_order):
        vals = []
        lows = []
        highs = []

        for set_name in set_order:
            row = summary[(summary["set"] == set_name) & (summary["prompt"] == prompt_name)]
            if len(row) == 0:
                vals.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
            else:
                med = float(row["median"].iloc[0])
                q25 = float(row["q25"].iloc[0])
                q75 = float(row["q75"].iloc[0])

                vals.append(med)
                lows.append(med - q25)
                highs.append(q75 - med)

        offset = (j - 1) * width

        plt.bar(
            x + offset,
            vals,
            width,
            yerr=np.array([lows, highs]),
            capsize=3,
            label=f"{prompt_name} prompt",
        )

    plt.xticks(x, set_order)
    plt.xlabel("Audio set")
    plt.ylabel("CLAP cosine similarity")
    plt.title("Reference CLAP similarities between datasets and prompts")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Text prompt", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    fig_png = output_dir / "clap_reference_barplot.png"
    fig_pdf = output_dir / "clap_reference_barplot.pdf"

    plt.savefig(fig_png, dpi=250, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(" ", scores_path)
    print(" ", summary_path)
    print(" ", fig_png)
    print(" ", fig_pdf)


if __name__ == "__main__":
    main()