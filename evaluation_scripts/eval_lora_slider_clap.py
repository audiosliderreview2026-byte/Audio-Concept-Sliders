#!/usr/bin/env python
"""
Evaluate an AudioLDM2 LoRA slider with CLAP similarity curves.

For each input audio and slider strength, this script:
  1. encodes + inverts the source audio,
  2. generates an edited audio with the trained LoRA slider,
  3. saves the edited WAV to disk,
  4. computes CLAP cosine similarity to three prompts:
       - neutral prompt
       - positive prompt
       - negative prompt
  5. averages scores over source audios,
  6. saves CSV files and a figure.

Expected to be launched from the project root so `audioldm2_slider` is importable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import librosa

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLAP evaluation curves for an AudioLDM2 LoRA slider.")

    # Project / model / LoRA
    p.add_argument("--project_root", type=str, default=".", help="Path containing the audioldm2_slider package.")
    p.add_argument("--model_path", required=True, help="Local AudioLDM2 model path.")
    p.add_argument("--lora_path", required=True, help="Path to trained LoRA checkpoint .pt.")
    p.add_argument("--lora_rank", type=int, required=True)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # Audio inputs / outputs
    p.add_argument("--audio_dir", required=True, help="Folder containing source wav files to edit.")
    p.add_argument("--audio_glob", default="*.wav")
    p.add_argument("--max_audios", type=int, default=None)
    p.add_argument("--output_dir", default="slider_clap_eval_outputs")
    p.add_argument("--overwrite_audio", action="store_true", help="Regenerate edited wavs even if cached files exist.")

    # Diffusion/editing config
    p.add_argument("--steps", type=int, default=50, help="Number of DDIM steps used for inversion/editing.")
    p.add_argument("--target_audio_length_s", type=float, default=10.0)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--train_cfg_scale", type=float, default=3.5)
    p.add_argument("--invert_prompt", default="")

    # Slider grid
    p.add_argument("--strength_min", type=float, default=-5.0)
    p.add_argument("--strength_max", type=float, default=5.0)
    p.add_argument("--num_strengths", type=int, default=21, help="Number of slider points between min and max.")
    p.add_argument("--strengths", nargs="+", type=float, default=None, help="Optional explicit strength list.")

    # Prompts for CLAP metric and for the trainer prompt bundle
    p.add_argument("--neutral_prompt", required=True)
    p.add_argument("--positive_prompt", required=True)
    p.add_argument("--negative_prompt", required=True)
    p.add_argument(
        "--target_prompt",
        default="a version of this sound that preserves its rhythm, timing, and overall structure",
    )

    # CLAP model
    p.add_argument(
        "--clap_model_path",
        default="laion/clap-htsat-unfused",
        help="HF id or local path for a transformers ClapModel/ClapProcessor.",
    )
    p.add_argument("--clap_local_files_only", action="store_true")
    p.add_argument("--clap_batch_size", type=int, default=8)

    # Misc
    p.add_argument("--device", default=None, help="Override device, e.g. cuda, cpu.")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def safe_float_tag(x: float) -> str:
    # -2.50 -> m2p50, +1.00 -> p1p00
    return (f"{x:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p"))


def sanitize_stem(stem: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)


def list_audio_files(audio_dir: Path, pattern: str, max_audios: int | None) -> List[Path]:
    files = sorted(audio_dir.glob(pattern))
    if max_audios is not None:
        files = files[:max_audios]
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir} with pattern {pattern!r}")
    return files


def load_lora_checkpoint_into_unet(unet, checkpoint_path: Path, LoRALinear, strict: bool = True) -> Dict[str, object]:
    """
    Loads exported LoRA checkpoints of the form:
        module.name.lora_A.weight
        module.name.lora_B.weight

    Falls back to unet.load_state_dict(strict=False) if the checkpoint looks like a broader state dict.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")

    lora_keys = [k for k in state.keys() if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")]
    if not lora_keys:
        missing, unexpected = unet.load_state_dict(state, strict=False)
        return {
            "mode": "unet_state_dict",
            "num_loaded": len(state),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "wrong_shape": [],
        }

    modules = dict(unet.named_modules())
    num_loaded = 0
    missing = []
    wrong_shape = []

    for key in lora_keys:
        value = state[key]
        if key.endswith(".lora_A.weight"):
            module_name = key[: -len(".lora_A.weight")]
            sub_name = "lora_A"
        else:
            module_name = key[: -len(".lora_B.weight")]
            sub_name = "lora_B"

        module = modules.get(module_name, None)
        if module is None or not isinstance(module, LoRALinear):
            missing.append(key)
            continue

        target = getattr(module, sub_name).weight
        if tuple(target.shape) != tuple(value.shape):
            wrong_shape.append((key, tuple(value.shape), tuple(target.shape)))
            continue

        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        num_loaded += 1

    if strict and (missing or wrong_shape):
        msg = [f"LoRA checkpoint did not load cleanly: loaded={num_loaded}"]
        if missing:
            msg.append(f"missing={len(missing)} example={missing[:3]}")
        if wrong_shape:
            msg.append(f"wrong_shape={len(wrong_shape)} example={wrong_shape[:3]}")
        raise RuntimeError(" | ".join(msg))

    return {
        "mode": "exported_lora",
        "num_loaded": num_loaded,
        "missing": missing,
        "unexpected": [],
        "wrong_shape": wrong_shape,
    }


# -----------------------------------------------------------------------------
# CLAP scoring with transformers
# -----------------------------------------------------------------------------

class TransformersCLAPScorer:
    def __init__(self, model_path: str, device: str, local_files_only: bool = False):
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device)
        self.processor = ClapProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = ClapModel.from_pretrained(model_path, local_files_only=local_files_only).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def encode_audio_files(self, paths: List[Path]) -> torch.Tensor:

        target_sr = int(self.processor.feature_extractor.sampling_rate)  # should be 48000 
        audios = []
        for p in paths:
            wav, sr = sf.read(str(p), always_2d=False)
        
            if getattr(wav, "ndim", 1) == 2:
                wav = wav.mean(axis=1)
        
            wav = wav.astype(np.float32)
        
            peak = np.max(np.abs(wav)) if wav.size > 0 else 0.0
            if peak > 1.0:
                wav = wav / peak
        
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        
            audios.append(wav)
        
        inputs = self.processor(
            audios=audios,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_audio_features(**inputs)
        return F.normalize(emb.float(), dim=-1)

    def score_audio_files(self, paths: List[Path], text_embeds: torch.Tensor, batch_size: int) -> np.ndarray:
        all_scores = []
        for i in tqdm(range(0, len(paths), batch_size), desc="CLAP audio batches"):
            batch = paths[i : i + batch_size]
            audio_emb = self.encode_audio_files(batch)
            scores = audio_emb @ text_embeds.T
            all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores, axis=0)


# -----------------------------------------------------------------------------
# Main generation + evaluation
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from audioldm2_slider.editor import AudioLDM2Editor, EditorConfig
    from audioldm2_slider.trainer import LoRASliderTrainer, TrainerConfig
    from audioldm2_slider.lora import LoRAConfig, LoRALinear
    from audioldm2_slider.utils import decode_latents_to_audio, save_wav

    output_dir = Path(args.output_dir)
    edited_dir = output_dir / "edited_wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list_audio_files(Path(args.audio_dir), args.audio_glob, args.max_audios)
    strengths = args.strengths if args.strengths is not None else np.linspace(args.strength_min, args.strength_max, args.num_strengths).tolist()
    strengths = [float(s) for s in strengths]

    print("=== Evaluation configuration ===")
    print("Audio files:", len(audio_files))
    print("Strengths:", strengths)
    print("Output dir:", output_dir)
    print("LoRA:", args.lora_path)

    # Save config for reproducibility.
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load AudioLDM2 editor and LoRA.
    print("\nLoading AudioLDM2 editor...")
    editor_cfg = EditorConfig(
        model_path=args.model_path,
        device=args.device,
        target_audio_length_s=args.target_audio_length_s,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
    )
    editor = AudioLDM2Editor(editor_cfg)

    lora_cfg = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_keywords=["to_q", "to_k", "to_v", "to_out.0"],
    )
    editor.inject_lora(lora_cfg, verbose=True)

    print("Loading LoRA checkpoint...")
    load_info = load_lora_checkpoint_into_unet(editor.pipe.unet, Path(args.lora_path), LoRALinear, strict=True)
    print("LoRA load info:", load_info)

    trainer_cfg = TrainerConfig(
        training_regime="prompt",
        prompt_target=args.target_prompt,
        prompt_neutral=args.neutral_prompt,
        prompt_piano=args.positive_prompt,
        prompt_guitar=args.negative_prompt,
        train_audio_length_s=args.target_audio_length_s,
        train_cfg_scale=args.train_cfg_scale,
        train_steps=args.steps,
        train_max_new_tokens=args.max_new_tokens,
        num_epochs=1,
    )
    trainer = LoRASliderTrainer(editor, trainer_cfg)
    # Make sure all inference modules are on the same device.
    editor.pipe.unet.to(editor.device, dtype=torch.float32)
    editor.pipe.vae.to(editor.device, dtype=torch.float32)
    editor.pipe.text_encoder.to(editor.device, dtype=torch.float32)
    editor.pipe.text_encoder_2.to(editor.device, dtype=torch.float32)
    editor.pipe.projection_model.to(editor.device, dtype=torch.float32)
    editor.pipe.language_model.to(editor.device, dtype=torch.float32)
    editor.pipe.unet.eval()

    # Generate/cache edited files.
    rows = []
    edited_paths = []

    print("\nGenerating or loading edited WAVs...")
    for audio_path in tqdm(audio_files, desc="Source audios"):
        stem = sanitize_stem(audio_path.stem)
        source_hash = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:10]
        source_out_dir = edited_dir / f"{stem}_{source_hash}"
        source_out_dir.mkdir(parents=True, exist_ok=True)
        print("doing",str(audio_path))
        # Only invert if at least one requested edited file is missing or overwrite is enabled.
        expected_paths = [source_out_dir / f"{stem}_strength_{safe_float_tag(s)}.wav" for s in strengths]
        need_generation = args.overwrite_audio or any(not p.exists() for p in expected_paths)

        encoded = None
        zT = None
        wav_len = None
        sr = None
        print("editor.device:", editor.device)
        print("UNet device:", next(editor.pipe.unet.parameters()).device)
        if need_generation:
            encoded = editor.encode_and_invert_audio(str(audio_path), invert_prompt=args.invert_prompt)
            zT = encoded["zT"].to(editor.pipe.device, dtype=torch.float32)
            wav_len = int(encoded["wav_len"])
            sr = int(encoded["sr"])

        for s, out_path in zip(strengths, expected_paths):
            print(str(s))
            if need_generation and (args.overwrite_audio or not out_path.exists()):
                direction = +1 if s >= 0 else -1
                with torch.no_grad():
                    zT = zT.to(editor.device, dtype=torch.float32)
                    z0_edit = trainer.forward_edit_latent(zT, direction=direction, lora_multiplier=float(s))
                    wav_edit = decode_latents_to_audio(editor.pipe, z0_edit, wav_len=wav_len)
                save_wav(out_path, wav_edit, sr)
                del z0_edit, wav_edit
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            rows.append({
                "source_path": str(audio_path),
                "edited_path": str(out_path),
                "source_stem": stem,
                "strength": float(s),
            })
            edited_paths.append(out_path)

        del encoded, zT
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = output_dir / "edited_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "edited_path", "source_stem", "strength"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved manifest:", manifest_path)

    # CLAP score.
    print("\nLoading CLAP scorer...")
    clap_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    clap = TransformersCLAPScorer(args.clap_model_path, clap_device, local_files_only=args.clap_local_files_only)

    prompts = [args.negative_prompt, args.neutral_prompt, args.positive_prompt]
    prompt_names = ["negative", "neutral", "positive"]
    text_embeds = clap.encode_texts(prompts)

    print("Computing CLAP similarities...")
    scores = clap.score_audio_files(edited_paths, text_embeds, batch_size=args.clap_batch_size)

    detailed_rows = []
    for row, score_vec in zip(rows, scores):
        out = dict(row)
        for name, score in zip(prompt_names, score_vec):
            out[f"clap_{name}"] = float(score)
        detailed_rows.append(out)

    detailed_csv = output_dir / "clap_scores_per_file.csv"
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_path", "edited_path", "source_stem", "strength", "clap_negative", "clap_neutral", "clap_positive"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print("Saved detailed scores:", detailed_csv)

    # Aggregate by strength.
    agg_rows = []
    for s in strengths:
        subset = [r for r in detailed_rows if abs(float(r["strength"]) - float(s)) < 1e-9]
        agg = {"strength": float(s), "n": len(subset)}
        for name in prompt_names:
            vals = np.array([r[f"clap_{name}"] for r in subset], dtype=np.float64)
            agg[f"clap_{name}_mean"] = float(vals.mean())
            agg[f"clap_{name}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[f"clap_{name}_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        agg_rows.append(agg)

    agg_csv = output_dir / "clap_scores_by_strength.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "strength", "n",
            "clap_negative_mean", "clap_negative_std", "clap_negative_sem",
            "clap_neutral_mean", "clap_neutral_std", "clap_neutral_sem",
            "clap_positive_mean", "clap_positive_std", "clap_positive_sem",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print("Saved aggregate scores:", agg_csv)

    # Plot.
    xs = np.array([r["strength"] for r in agg_rows], dtype=np.float64)
    plt.figure(figsize=(8, 5))
    for name, label in [("negative", "Negative prompt"), ("neutral", "Neutral prompt"), ("positive", "Positive prompt")]:
        ys = np.array([r[f"clap_{name}_mean"] for r in agg_rows], dtype=np.float64)
        sem = np.array([r[f"clap_{name}_sem"] for r in agg_rows], dtype=np.float64)
        plt.plot(xs, ys, marker="o", label=label)
        plt.fill_between(xs, ys - sem, ys + sem, alpha=0.2)

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Slider strength")
    plt.ylabel("CLAP cosine similarity")
    plt.title("CLAP prompt similarity across LoRA slider strengths")
    plt.legend()
    plt.tight_layout()

    fig_path = output_dir / "clap_similarity_by_slider_strength.png"
    pdf_path = output_dir / "clap_similarity_by_slider_strength.pdf"
    plt.savefig(fig_path, dpi=200)
    plt.savefig(pdf_path)
    print("Saved figure:", fig_path)
    print("Saved figure:", pdf_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
