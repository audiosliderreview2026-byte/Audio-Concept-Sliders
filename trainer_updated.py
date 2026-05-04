from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc
import hashlib
import random
import time

import torch
from tqdm.auto import tqdm

from .lora import freeze_non_lora_params, export_lora_state_dict, set_all_lora_multipliers
from .prompts import encode_prompt_fixed
from .diffusion import predict_eps_cfg
from .utils import decode_latents_to_audio


@dataclass
class TrainerConfig:
    dataset_root: str = "lora_dataset"
    train_songs: Optional[List[str]] = None
    val_songs: Optional[List[str]] = None

    # Prompt design: closer to Concept Sliders target / positive / unconditional / neutral setup.
    prompt_target: str = (
        "a version of this sound that preserves its exact timing, rhythm, melody, pacing, "
        "and overall event structure"
    )
    prompt_neutral: str = (
        "a version of this sound that preserves its exact timing, rhythm, melody, pacing, "
        "and overall event structure"
    )
    prompt_piano: str = (
        "a piano version of this sound, with piano notes matching the exact timing, rhythm, "
        "melody, pacing, and overall event structure of the original"
    )
    prompt_guitar: str = (
        "a guitar version of this sound, with guitar notes matching the exact timing, rhythm, "
        "melody, pacing, and overall event structure of the original"
    )

    train_audio_length_s: float = 10.0
    train_cfg_scale: float = 3.5
    train_steps: int = 50
    train_max_new_tokens: int = 8

    # Concept-Sliders-like loss controls
    concept_guidance_scale: float = 4.0
    target_denoise_guidance_scale: float = 3.0
    identity_loss_weight: float = 1.0
    pair_loss_weight: float = 0.0
    train_timestep_mode: str = "random"   # "random" or "middle"
    val_timestep_mode: str = "middle"     # "random" or "middle"

    num_epochs: int = 10
    lr_lora: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    max_train_pairs: Optional[int] = None
    max_val_pairs: Optional[int] = None

    out_dir: str = "lora_slider_training"
    checkpoint_every: int = 1

    seed: int = 42
    cache_dir_name: Optional[str] = None
    use_inner_tqdm: bool = False

    # memory-related flags
    enable_gradient_checkpointing: bool = True
    enable_attention_slicing: bool = False
    empty_cuda_cache_each_step: bool = True


class LoRASliderTrainer:
    def __init__(self, editor, config: TrainerConfig):
        self.editor = editor
        self.pipe = editor.pipe
        self.config = config
        self.latent_cache: Dict[str, dict] = {}
        self.history = {"train": [], "val": []}

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.pipe.scheduler.set_timesteps(config.train_steps, device=self.pipe.device)
        self.timesteps = self.pipe.scheduler.timesteps
        self.extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

        # Target / neutral prompt pair: in Concept Sliders target is usually the base entity,
        # while positive / unconditional define the concept direction.
        self.prompt_bundle = {}
        self._cache_prompt_bundles()

    def _encode_prompt_triplet(self, target_prompt: str, positive_prompt: str, unconditional_prompt: str, neutral_prompt: str):
        pt_target, am_target, gt_target, pu_target, amu_target, gu_target = encode_prompt_fixed(
            self.pipe,
            target_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )
        pt_positive, am_positive, gt_positive, pu_positive, amu_positive, gu_positive = encode_prompt_fixed(
            self.pipe,
            positive_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )
        pt_uncond, am_uncond, gt_uncond, pu_uncond, amu_uncond, gu_uncond = encode_prompt_fixed(
            self.pipe,
            unconditional_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )
        pt_neutral, am_neutral, gt_neutral, pu_neutral, amu_neutral, gu_neutral = encode_prompt_fixed(
            self.pipe,
            neutral_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )

        return {
            "target": (pt_target, am_target, gt_target, pu_target, amu_target, gu_target),
            "positive": (pt_positive, am_positive, gt_positive, pu_positive, amu_positive, gu_positive),
            "unconditional": (pt_uncond, am_uncond, gt_uncond, pu_uncond, amu_uncond, gu_uncond),
            "neutral": (pt_neutral, am_neutral, gt_neutral, pu_neutral, amu_neutral, gu_neutral),
        }

    def _cache_prompt_bundles(self):
        # +1 means "toward piano", -1 means "toward guitar"
        self.prompt_bundle[+1] = self._encode_prompt_triplet(
            target_prompt=self.config.prompt_target,
            positive_prompt=self.config.prompt_piano,
            unconditional_prompt=self.config.prompt_guitar,
            neutral_prompt=self.config.prompt_neutral,
        )
        self.prompt_bundle[-1] = self._encode_prompt_triplet(
            target_prompt=self.config.prompt_target,
            positive_prompt=self.config.prompt_guitar,
            unconditional_prompt=self.config.prompt_piano,
            neutral_prompt=self.config.prompt_neutral,
        )

    def build_optimizer(self):
        freeze_non_lora_params(self.pipe.unet)
        self.pipe.unet.to(self.pipe.device, dtype=torch.float32)

        if self.config.enable_gradient_checkpointing:
            try:
                self.pipe.unet.enable_gradient_checkpointing()
                print("[INFO] Enabled UNet gradient checkpointing.")
            except Exception as e:
                print(f"[WARN] Could not enable gradient checkpointing: {e}")

        if self.config.enable_attention_slicing:
            try:
                self.pipe.enable_attention_slicing()
                print("[INFO] Enabled attention slicing.")
            except Exception as e:
                print(f"[WARN] Could not enable attention slicing: {e}")

        self.pipe.unet.train()
        params = [p for p in self.pipe.unet.parameters() if p.requires_grad]
        print(f"[INFO] Trainable parameter tensors: {len(params)}")
        return torch.optim.AdamW(params, lr=self.config.lr_lora, weight_decay=self.config.weight_decay)

    def get_lora_named_params(self):
        out = []
        for name, p in self.pipe.unet.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                out.append((name, p))
        return out

    def get_lora_grad_stats(self):
        grads = []
        for _, p in self.get_lora_named_params():
            if p.grad is not None:
                grads.append(float(p.grad.detach().norm().item()))
        if len(grads) == 0:
            return {"num_with_grad": 0, "mean_grad_norm": 0.0, "max_grad_norm": 0.0}
        return {
            "num_with_grad": len(grads),
            "mean_grad_norm": sum(grads) / len(grads),
            "max_grad_norm": max(grads),
        }

    def snapshot_lora_params(self):
        snap = {}
        for name, p in self.get_lora_named_params():
            snap[name] = p.detach().clone().cpu()
        return snap

    def compare_lora_snapshots(self, before, after):
        deltas = []
        for name in before:
            delta = (after[name] - before[name]).abs().mean().item()
            deltas.append(delta)
        if len(deltas) == 0:
            return {"mean_abs_param_delta": 0.0, "max_abs_param_delta": 0.0}
        return {
            "mean_abs_param_delta": sum(deltas) / len(deltas),
            "max_abs_param_delta": max(deltas),
        }

    def make_latent_cache_key(self, audio_path: str) -> str:
        return hashlib.md5(audio_path.encode("utf-8")).hexdigest()

    def get_default_cache_dir(self, invert_prompt: str = "") -> Path:
        prompt_tag = "empty" if invert_prompt == "" else hashlib.md5(invert_prompt.encode("utf-8")).hexdigest()[:8]
        dir_name = self.config.cache_dir_name or (
            f"latent_cache_steps{self.config.train_steps}_len{self.config.train_audio_length_s:g}_inv_{prompt_tag}"
        )
        return Path(self.config.out_dir) / dir_name

    def get_latent_cache_file(self, audio_path: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self.make_latent_cache_key(audio_path)
        stem = Path(audio_path).stem
        return cache_dir / f"{stem}_{key}.pt"

    def save_latent_cache_item(self, cache_file: Path, audio_path: str, sr: int, wav_len: int, z0, zT):
        payload = {
            "audio_path": audio_path,
            "sr": int(sr),
            "wav_len": int(wav_len),
            "z0": z0.detach().cpu(),
            "zT": zT.detach().cpu(),
        }
        torch.save(payload, cache_file)

    def load_latent_cache_item(self, cache_file: Path):
        payload = torch.load(cache_file, map_location="cpu")
        return {
            "wav": None,
            "sr": int(payload["sr"]),
            "wav_len": int(payload["wav_len"]),
            "z0": payload["z0"].to(dtype=torch.float32),
            "zT": payload["zT"].to(dtype=torch.float32),
        }

    def get_audio_for_path(self, audio_path: str):
        return self.editor.load_audio(audio_path)

    def precompute_latent_cache(
        self,
        pairs,
        invert_prompt: str = "",
        cache_dir: Optional[str | Path] = None,
        store_wav: bool = False,
        overwrite: bool = False,
    ):
        cache_dir = Path(cache_dir) if cache_dir is not None else self.get_default_cache_dir(invert_prompt=invert_prompt)
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_paths = sorted({item["piano"] for item in pairs} | {item["guitar"] for item in pairs})
        print(f"Unique wav files to cache: {len(all_paths)}")
        print(f"Latent cache dir: {cache_dir}")

        for path in tqdm(all_paths, desc="Precomputing / loading latents"):
            if (path in self.latent_cache) and (not overwrite):
                if store_wav and self.latent_cache[path].get("wav") is None:
                    wav, sr = self.get_audio_for_path(path)
                    self.latent_cache[path]["wav"] = wav
                    self.latent_cache[path]["sr"] = sr
                continue

            cache_file = self.get_latent_cache_file(path, cache_dir)

            if cache_file.exists() and (not overwrite):
                item = self.load_latent_cache_item(cache_file)
                if store_wav:
                    wav, sr = self.get_audio_for_path(path)
                    item["wav"] = wav
                    item["sr"] = sr
                self.latent_cache[path] = item
                continue

            encoded = self.editor.encode_and_invert_audio(path, invert_prompt=invert_prompt)

            self.save_latent_cache_item(
                cache_file=cache_file,
                audio_path=path,
                sr=encoded["sr"],
                wav_len=encoded["wav_len"],
                z0=encoded["z0"],
                zT=encoded["zT"],
            )

            self.latent_cache[path] = {
                "wav": encoded["wav"] if store_wav else None,
                "sr": encoded["sr"],
                "wav_len": encoded["wav_len"],
                "z0": encoded["z0"].detach().cpu(),
                "zT": encoded["zT"].detach().cpu(),
            }

            del encoded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Latent cache ready.")
        print(f"Cached items in memory: {len(self.latent_cache)}")
        return self.latent_cache

    def get_endpoint_prompt_tensors(self, direction: int):
        # Kept for backward compatibility.
        bundle = self.prompt_bundle[direction]
        return bundle["positive"]

    def _get_prompt_triplet(self, direction: int):
        return self.prompt_bundle[direction]

    def _select_timestep_index(self, mode: str) -> int:
        if len(self.timesteps) < 2:
            return 0
        if mode == "middle":
            return max(1, len(self.timesteps) // 2)
        # default random in [1, len-1)
        return random.randint(1, len(self.timesteps) - 1)

    def _partial_denoise(self, zT, branch_tensors, lora_multiplier: float, stop_index: int, cfg_scale: float):
        pt, am, gt, pu, amu, gu = branch_tensors
        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)

        x = zT
        iterator = range(stop_index)
        if self.config.use_inner_tqdm:
            iterator = tqdm(iterator, desc="Partial denoise", leave=False)

        for i in iterator:
            t = self.timesteps[i]
            eps = predict_eps_cfg(
                self.pipe,
                x,
                t,
                pu,
                pt,
                amu,
                am,
                gu,
                gt,
                cfg_scale,
                force_cpu_unet=False,
                cpu_grad_mode=False,
            )
            x = self.pipe.scheduler.step(eps, t, x, **self.extra_step_kwargs).prev_sample
        return x

    def _predict_branch_eps(self, x, timestep, branch_tensors, lora_multiplier: float, cfg_scale: float = 1.0):
        pt, am, gt, pu, amu, gu = branch_tensors
        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)
        return predict_eps_cfg(
            self.pipe,
            x,
            timestep,
            pu,
            pt,
            amu,
            am,
            gu,
            gt,
            cfg_scale,
            force_cpu_unet=False,
            cpu_grad_mode=False,
        )

    def _continue_forward_from_index(self, x, start_index: int, branch_tensors, lora_multiplier: float, cfg_scale: float):
        pt, am, gt, pu, amu, gu = branch_tensors
        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)
        for i in range(start_index, len(self.timesteps) - 1):
            t = self.timesteps[i]
            eps = predict_eps_cfg(
                self.pipe,
                x,
                t,
                pu,
                pt,
                amu,
                am,
                gu,
                gt,
                cfg_scale,
                force_cpu_unet=False,
                cpu_grad_mode=False,
            )
            x = self.pipe.scheduler.step(eps, t, x, **self.extra_step_kwargs).prev_sample
        return x

    def _compute_concept_slider_loss(self, zT_src, direction: int, z0_tgt=None, mode: str = "random"):
        bundle = self._get_prompt_triplet(direction)
        step_idx = self._select_timestep_index(mode)
        current_timestep = self.timesteps[step_idx]

        target_tensors = bundle["target"]
        positive_tensors = bundle["positive"]
        unconditional_tensors = bundle["unconditional"]
        neutral_tensors = bundle["neutral"]

        # Concept-Sliders-like setup:
        # 1) move from source zT to an intermediate latent using the target prompt with LoRA active
        x_target = self._partial_denoise(
            zT_src,
            target_tensors,
            lora_multiplier=1.0,
            stop_index=step_idx,
            cfg_scale=self.config.target_denoise_guidance_scale,
        )

        # 2) build a base / neutral path without LoRA to preserve structure
        x_neutral_base = self._partial_denoise(
            zT_src,
            neutral_tensors,
            lora_multiplier=0.0,
            stop_index=step_idx,
            cfg_scale=self.config.target_denoise_guidance_scale,
        )

        # 3) evaluate concept directions at the same intermediate point
        positive_eps = self._predict_branch_eps(
            x_target.detach(), current_timestep, positive_tensors, lora_multiplier=0.0, cfg_scale=1.0
        )
        neutral_eps = self._predict_branch_eps(
            x_target.detach(), current_timestep, neutral_tensors, lora_multiplier=0.0, cfg_scale=1.0
        )
        unconditional_eps = self._predict_branch_eps(
            x_target.detach(), current_timestep, unconditional_tensors, lora_multiplier=0.0, cfg_scale=1.0
        )
        target_eps = self._predict_branch_eps(
            x_target, current_timestep, target_tensors, lora_multiplier=1.0, cfg_scale=1.0
        )

        concept_target = neutral_eps + self.config.concept_guidance_scale * (positive_eps - unconditional_eps)
        loss_concept = torch.nn.functional.mse_loss(target_eps, concept_target)

        # Identity preservation: keep the LoRA-altered trajectory near the neutral/base trajectory.
        loss_identity = torch.nn.functional.mse_loss(x_target, x_neutral_base)

        # Optional auxiliary paired endpoint loss.
        if self.config.pair_loss_weight > 0.0 and z0_tgt is not None:
            z0_slider = self._continue_forward_from_index(
                x_target,
                start_index=step_idx,
                branch_tensors=target_tensors,
                lora_multiplier=1.0,
                cfg_scale=self.config.train_cfg_scale,
            )
            loss_pair = torch.nn.functional.mse_loss(z0_slider, z0_tgt)
        else:
            z0_slider = None
            loss_pair = target_eps.new_tensor(0.0)

        loss_total = loss_concept + self.config.identity_loss_weight * loss_identity + self.config.pair_loss_weight * loss_pair

        return loss_total, {
            "loss_total": float(loss_total.detach().item()),
            "loss_concept": float(loss_concept.detach().item()),
            "loss_identity": float(loss_identity.detach().item()),
            "loss_pair": float(loss_pair.detach().item()),
            "step_idx": int(step_idx),
        }

    def forward_edit_latent(self, zT, direction: int, lora_multiplier: float):
        # For inference: use the target/neutral prompt and control with the slider multiplier.
        target_tensors = self._get_prompt_triplet(direction)["target"]
        pt_edit, am_edit, gt_edit, pu_edit, amu_edit, gu_edit = target_tensors
        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)

        x = zT
        iterator = range(len(self.timesteps) - 1)
        if self.config.use_inner_tqdm:
            iterator = tqdm(iterator, desc="Forward edit", leave=False)

        for i in iterator:
            t = self.timesteps[i]
            eps = predict_eps_cfg(
                self.pipe,
                x,
                t,
                pu_edit,
                pt_edit,
                amu_edit,
                am_edit,
                gu_edit,
                gt_edit,
                self.config.train_cfg_scale,
                force_cpu_unet=False,
                cpu_grad_mode=False,
            )
            x = self.pipe.scheduler.step(eps, t, x, **self.extra_step_kwargs).prev_sample

        return x

    def training_step_on_sample(self, sample):
        source = self.latent_cache[sample["source_path"]]
        target = self.latent_cache[sample["target_path"]]

        zT_src = source["zT"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        z0_tgt = target["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        direction = int(sample["direction"])

        return self._compute_concept_slider_loss(
            zT_src=zT_src,
            direction=direction,
            z0_tgt=z0_tgt,
            mode=self.config.train_timestep_mode,
        )

    @torch.no_grad()
    def validation_step_on_sample(self, sample):
        self.pipe.unet.eval()

        source = self.latent_cache[sample["source_path"]]
        target = self.latent_cache[sample["target_path"]]

        zT_src = source["zT"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        z0_tgt = target["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        direction = int(sample["direction"])

        loss, logs = self._compute_concept_slider_loss(
            zT_src=zT_src,
            direction=direction,
            z0_tgt=z0_tgt,
            mode=self.config.val_timestep_mode,
        )

        del zT_src, z0_tgt, loss
        gc.collect()
        if torch.cuda.is_available() and self.config.empty_cuda_cache_each_step:
            torch.cuda.empty_cache()

        self.pipe.unet.train()
        return logs

    def train(self, train_samples, val_samples=None):
        val_samples = [] if val_samples is None else val_samples
        optimizer = self.build_optimizer()

        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")

        for epoch in range(1, self.config.num_epochs + 1):
            self.pipe.unet.train()
            freeze_non_lora_params(self.pipe.unet)
            random.shuffle(train_samples)

            train_loss_sum = 0.0
            train_concept_sum = 0.0
            train_identity_sum = 0.0
            train_pair_sum = 0.0
            train_count = 0
            t0 = time.time()

            pbar = tqdm(train_samples, desc=f"Epoch {epoch}/{self.config.num_epochs} [train]")
            for step_idx, sample in enumerate(pbar):
                optimizer.zero_grad(set_to_none=True)

                before_step = None
                if epoch == 1 and step_idx == 0:
                    before_step = self.snapshot_lora_params()

                loss = None
                try:
                    loss, logs = self.training_step_on_sample(sample)
                    loss.backward()

                    if epoch == 1 and step_idx == 0:
                        grad_stats = self.get_lora_grad_stats()
                        print("[DIAG] First backward pass:")
                        print(f"  LoRA tensors with grad: {grad_stats['num_with_grad']}")
                        print(f"  Mean grad norm:         {grad_stats['mean_grad_norm']:.6e}")
                        print(f"  Max grad norm:          {grad_stats['max_grad_norm']:.6e}")

                    if self.config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.pipe.unet.parameters() if p.requires_grad],
                            self.config.grad_clip_norm,
                        )

                    optimizer.step()

                    if epoch == 1 and step_idx == 0 and before_step is not None:
                        after_step = self.snapshot_lora_params()
                        delta_stats = self.compare_lora_snapshots(before_step, after_step)
                        print("[DIAG] First optimizer step:")
                        print(f"  Mean abs param delta:   {delta_stats['mean_abs_param_delta']:.6e}")
                        print(f"  Max abs param delta:    {delta_stats['max_abs_param_delta']:.6e}")

                    train_loss_sum += logs["loss_total"]
                    train_concept_sum += logs["loss_concept"]
                    train_identity_sum += logs["loss_identity"]
                    train_pair_sum += logs["loss_pair"]
                    train_count += 1
                    pbar.set_postfix(
                        loss=f"{logs['loss_total']:.6f}",
                        concept=f"{logs['loss_concept']:.6f}",
                        ident=f"{logs['loss_identity']:.6f}",
                        avg=f"{train_loss_sum / train_count:.6f}",
                    )

                finally:
                    if loss is not None:
                        del loss
                    gc.collect()
                    if torch.cuda.is_available() and self.config.empty_cuda_cache_each_step:
                        torch.cuda.empty_cache()

            train_avg = train_loss_sum / max(train_count, 1)
            train_concept_avg = train_concept_sum / max(train_count, 1)
            train_identity_avg = train_identity_sum / max(train_count, 1)
            train_pair_avg = train_pair_sum / max(train_count, 1)
            self.history["train"].append(train_avg)

            if len(val_samples) > 0:
                val_loss_sum = 0.0
                val_concept_sum = 0.0
                val_identity_sum = 0.0
                val_pair_sum = 0.0
                val_count = 0
                for sample in tqdm(val_samples, desc=f"Epoch {epoch}/{self.config.num_epochs} [val]"):
                    logs = self.validation_step_on_sample(sample)
                    val_loss_sum += logs["loss_total"]
                    val_concept_sum += logs["loss_concept"]
                    val_identity_sum += logs["loss_identity"]
                    val_pair_sum += logs["loss_pair"]
                    val_count += 1
                val_avg = val_loss_sum / max(val_count, 1)
                val_concept_avg = val_concept_sum / max(val_count, 1)
                val_identity_avg = val_identity_sum / max(val_count, 1)
                val_pair_avg = val_pair_sum / max(val_count, 1)
            else:
                val_avg = float("nan")
                val_concept_avg = float("nan")
                val_identity_avg = float("nan")
                val_pair_avg = float("nan")

            self.history["val"].append(val_avg)

            print(f"Epoch {epoch}/{self.config.num_epochs} done in {time.time() - t0:.1f}s")
            print(f"  train loss:     {train_avg:.6f}")
            print(f"    concept:      {train_concept_avg:.6f}")
            print(f"    identity:     {train_identity_avg:.6f}")
            print(f"    pair:         {train_pair_avg:.6f}")
            print(f"  val   loss:     {val_avg:.6f}")
            print(f"    concept:      {val_concept_avg:.6f}")
            print(f"    identity:     {val_identity_avg:.6f}")
            print(f"    pair:         {val_pair_avg:.6f}")

            if (epoch % self.config.checkpoint_every) == 0:
                ckpt_path = out_dir / f"lora_epoch_{epoch:03d}.pt"
                torch.save(export_lora_state_dict(self.pipe.unet), ckpt_path)
                print("  saved:", ckpt_path)

            if len(val_samples) > 0 and val_avg < best_val:
                best_val = val_avg
                best_path = out_dir / "lora_best.pt"
                torch.save(export_lora_state_dict(self.pipe.unet), best_path)
                print("  new best:", best_path)

        return self.history

    @torch.no_grad()
    def run_slider_inference_on_source(self, source_path, strengths=(-2.0, -1.0, 0.0, 1.0, 2.0)):
        self.pipe.unet.eval()

        source = self.latent_cache[source_path]
        zT_src = source["zT"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)

        source_wav = source.get("wav")
        source_sr = source["sr"]
        if source_wav is None:
            source_wav, source_sr = self.get_audio_for_path(source_path)

        outputs = []
        for s in strengths:
            direction = +1 if s >= 0 else -1
            z0_edit = self.forward_edit_latent(zT_src, direction=direction, lora_multiplier=float(abs(s)))
            wav_edit = decode_latents_to_audio(self.pipe, z0_edit, wav_len=source["wav_len"])
            outputs.append({"strength": float(s), "wav": wav_edit, "sr": source_sr})

            del z0_edit
            gc.collect()
            if torch.cuda.is_available() and self.config.empty_cuda_cache_each_step:
                torch.cuda.empty_cache()

        del zT_src
        gc.collect()
        if torch.cuda.is_available() and self.config.empty_cuda_cache_each_step:
            torch.cuda.empty_cache()

        return {"source_wav": source_wav, "source_sr": source_sr, "outputs": outputs}
