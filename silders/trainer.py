from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import gc
import hashlib
import random
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .lora import freeze_non_lora_params, export_lora_state_dict, set_all_lora_multipliers
from .prompts import encode_prompt_fixed
from .diffusion import (
    predict_eps_cfg,
    predict_eps_for_inversion,
    build_alpha_cache,
    ddim_inverse_step_scheduler_aware,
)
from .utils import decode_latents_to_audio


@dataclass
class TrainerConfig:
    dataset_root: str = "lora_dataset"
    train_songs: Optional[List[str]] = None
    val_songs: Optional[List[str]] = None

    # - "prompt": prompt-based contrastive regime
    # - "prompt_free_paired": paired, prompt-free regime using source/target example contrast
    training_regime: str = "prompt"

    # These names are kept for compatibility with your launcher.
    prompt_target: str = ""
    prompt_neutral: str = ""
    prompt_piano: str = ""
    prompt_guitar: str = ""

    train_audio_length_s: float = 10.0
    train_cfg_scale: float = 3.5
    train_steps: int = 50
    train_max_new_tokens: int = 8

    # Loss controls
    identity_loss_weight: float = 0.0
    pair_loss_weight: float = 1.0
    separation_loss_weight: float = 1.0
    separation_margin: float = 0.0

    # NEW: prompt-free paired regime controls
    prompt_free_direction_batch_size: int = 4
    prompt_free_local_pair_weight: float = 0.5
    prompt_free_direction_scale_mode: str = "mean_norm"  # {"mean_norm", "current_norm", "unit"}

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

    enable_gradient_checkpointing: bool = True
    enable_attention_slicing: bool = False
    empty_cuda_cache_each_step: bool = True

    # Prompt-free x0-prediction regime controls
    prompt_free_x0_target_weight: float = 1.0
    prompt_free_x0_direction_weight: float = 1.0

    # Null-text optimization controls
    use_null_text_optimization: bool = False
    nti_cache_dir_name: Optional[str] = None
    nti_inner_steps: int = 5
    nti_lr: float = 1e-2
    nti_prompt: str = ""
    nti_optimize: str = "gu"  # {"pu", "gu", "both"}
    nti_overwrite: bool = False

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

        self.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(self.pipe.device, dtype=torch.float32)

        self.prompt_bundle = {}
        self._cache_prompt_bundles()

        # Empty branch used by the prompt-free regime
        self.empty_branch = self._encode_branch("")

        # Set by train()/validation so prompt-free paired can sample extra references
        self._current_sample_pool: List[dict] = []
        self._current_stage: str = "train"
        
    def _ensure_unet_on_device(self, device):
        self.pipe.unet.to(device=device, dtype=torch.float32)
        self.pipe.unet.eval()
        
    def _resolve_default_prompts(self):
        target = self.config.prompt_target.strip() if self.config.prompt_target else ""
        neutral = self.config.prompt_neutral.strip() if self.config.prompt_neutral else ""
        positive = self.config.prompt_piano.strip() if self.config.prompt_piano else ""
        negative = self.config.prompt_guitar.strip() if self.config.prompt_guitar else ""

        if target == "":
            target = "a version of this sound that preserves its rhythm, timing, and overall structure"
        if neutral == "":
            neutral = "this sound preserving its rhythm, timing, and overall structure"
        if positive == "":
            positive = "a version of this sound with the positive attribute, preserving its rhythm, timing, and overall structure"
        if negative == "":
            negative = "a version of this sound with the negative attribute, preserving its rhythm, timing, and overall structure"

        return target, neutral, positive, negative

    def _encode_branch(self, prompt: str):
        return encode_prompt_fixed(
            self.pipe,
            prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )

    def _encode_prompt_triplet(
        self,
        target_prompt: str,
        positive_prompt: str,
        unconditional_prompt: str,
        neutral_prompt: str,
    ):
        return {
            "target": self._encode_branch(target_prompt),
            "positive": self._encode_branch(positive_prompt),
            "unconditional": self._encode_branch(unconditional_prompt),
            "neutral": self._encode_branch(neutral_prompt),
        }

    def _cache_prompt_bundles(self):
        target, neutral, positive, negative = self._resolve_default_prompts()

        self.prompt_bundle[+1] = self._encode_prompt_triplet(
            target_prompt=target,
            positive_prompt=positive,
            unconditional_prompt=negative,
            neutral_prompt=neutral,
        )

        self.prompt_bundle[-1] = self._encode_prompt_triplet(
            target_prompt=target,
            positive_prompt=negative,
            unconditional_prompt=positive,
            neutral_prompt=neutral,
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

    def _collect_audio_paths_from_entries(self, entries):
        paths = set()
        for item in entries:
            for key in ("piano", "guitar", "source_path", "target_path", "path"):
                value = item.get(key, None)
                if value is not None:
                    paths.add(str(value))
        return sorted(paths)

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

        all_paths = self._collect_audio_paths_from_entries(pairs)
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
        return self.prompt_bundle[direction]["target"]

    def _get_prompt_triplet(self, direction: int):
        return self.prompt_bundle[direction]

    def _compute_eps_from_branch(
        self,
        x,
        timestep,
        branch,
        lora_multiplier: float,
        cfg_scale: float,
        null_override: Optional[dict] = None,
        timestep_index: Optional[int] = None,
    ):
        pt, am, gt, pu, amu, gu = branch

        if null_override is not None:
            if timestep_index is None:
                raise ValueError("timestep_index must be provided when using null_override.")

            nti_item = null_override["per_step"][int(timestep_index)]

            if "pu" in nti_item and nti_item["pu"] is not None:
                pu = nti_item["pu"].to(device=self.pipe.device, dtype=torch.float32)

            if "gu" in nti_item and nti_item["gu"] is not None:
                gu = nti_item["gu"].to(device=self.pipe.device, dtype=torch.float32)

        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)
        self._ensure_unet_on_device(x.device)
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

    def _sample_timestep_index(self):
        if len(self.timesteps) <= 2:
            return 0
        return random.randint(1, len(self.timesteps) - 2)

    def _sample_timestep(self):
        return self.timesteps[self._sample_timestep_index()]

    def _add_noise(self, z0, timestep, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        x_t = self.pipe.scheduler.add_noise(z0, noise, timestep)
        return x_t, noise

    def _predict_x0_from_eps(self, x_t, eps, timestep):
        t_idx = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        alpha_bar = self.alphas_cumprod[t_idx].to(dtype=x_t.dtype, device=x_t.device).view(1, 1, 1, 1)
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar)

        pred_type = getattr(self.pipe.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            x0 = (x_t - sqrt_one_minus_ab * eps) / (sqrt_ab + 1e-8)
        elif pred_type == "v_prediction":
            x0 = sqrt_ab * x_t - sqrt_one_minus_ab * eps
        elif pred_type == "sample":
            x0 = eps
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        return x0

    def _forward_edit_with_multiplier(self, zT, lora_multiplier: float, direction: int):
        if self.config.training_regime == "prompt_free_paired":
            target_branch = self.empty_branch
        else:
            target_branch = self._get_prompt_triplet(direction)["target"]

        x = zT
        iterator = range(len(self.timesteps) - 1)
        if self.config.use_inner_tqdm:
            iterator = tqdm(iterator, desc="Forward edit", leave=False)

        for i in iterator:
            t = self.timesteps[i]
            eps = self._compute_eps_from_branch(
                x,
                t,
                target_branch,
                lora_multiplier=lora_multiplier,
                cfg_scale=self.config.train_cfg_scale,
            )
            x = self.pipe.scheduler.step(eps, t, x, **self.extra_step_kwargs).prev_sample
        return x

    # ------------------------------------------------------------------
    # Regime 1: current prompt-based contrastive loss
    # ------------------------------------------------------------------
    def _prompt_concept_slider_losses(self, z0_src, direction: int, null_override: Optional[dict] = None):
        prompt_triplet = self._get_prompt_triplet(direction)

        timestep_index = self._sample_timestep_index()
        timestep = self.timesteps[timestep_index]

        x_t, noise = self._add_noise(z0_src, timestep)

        eps_target_base = self._compute_eps_from_branch(
            x_t,
            timestep,
            prompt_triplet["target"],
            lora_multiplier=0.0,
            cfg_scale=self.config.train_cfg_scale,
            null_override=null_override,
            timestep_index=timestep_index,
        )
        eps_neutral_base = self._compute_eps_from_branch(
            x_t,
            timestep,
            prompt_triplet["neutral"],
            lora_multiplier=0.0,
            cfg_scale=self.config.train_cfg_scale,
            null_override=null_override,
            timestep_index=timestep_index,
        )
        eps_positive_base = self._compute_eps_from_branch(
            x_t,
            timestep,
            prompt_triplet["positive"],
            lora_multiplier=0.0,
            cfg_scale=self.config.train_cfg_scale,
            null_override=null_override,
            timestep_index=timestep_index,
        )
        eps_uncond_base = self._compute_eps_from_branch(
            x_t,
            timestep,
            prompt_triplet["unconditional"],
            lora_multiplier=0.0,
            cfg_scale=self.config.train_cfg_scale,
            null_override=null_override,
            timestep_index=timestep_index,
        )

        eps_edit = self._compute_eps_from_branch(
            x_t,
            timestep,
            prompt_triplet["target"],
            lora_multiplier=float(direction),
            cfg_scale=self.config.train_cfg_scale,
            null_override=null_override,
            timestep_index=timestep_index,
        )

        direction_vec = eps_positive_base - eps_uncond_base
        concept_target = eps_neutral_base + self.config.train_cfg_scale * direction_vec

        loss_concept_mse = F.mse_loss(eps_edit, concept_target)

        edit_delta = eps_edit - eps_target_base
        flat_edit = edit_delta.flatten(1)
        flat_dir = direction_vec.flatten(1)
        cos_sim = F.cosine_similarity(flat_edit, flat_dir, dim=1).mean()
        loss_direction = 1.0 - cos_sim

        if self.config.separation_margin > 0.0:
            edit_norm = flat_edit.norm(dim=1)
            dir_norm = flat_dir.norm(dim=1).detach()
            loss_margin = torch.relu(self.config.separation_margin * dir_norm - edit_norm).mean()
        else:
            loss_margin = torch.zeros((), device=x_t.device, dtype=x_t.dtype)

        loss_concept = loss_concept_mse + loss_direction + loss_margin
        loss_pair = F.mse_loss(eps_edit, noise)

        x0_pred = self._predict_x0_from_eps(x_t, eps_edit, timestep)
        loss_identity = F.mse_loss(x0_pred, z0_src)

        loss_total = (
            self.config.separation_loss_weight * loss_concept
            + self.config.identity_loss_weight * loss_identity
            + self.config.pair_loss_weight * loss_pair
        )

        return loss_total, {
            "loss_total": float(loss_total.detach().item()),
            "loss_concept": float(loss_concept.detach().item()),
            "loss_identity": float(loss_identity.detach().item()),
            "loss_pair": float(loss_pair.detach().item()),
        }

    # ------------------------------------------------------------------
    # Regime 2: prompt-free paired loss in predicted-x0 space
    # ------------------------------------------------------------------
    def _paired_prompt_free_losses(self, z0_src, z0_tgt, direction: int, sample: Optional[dict] = None):
        """
        Prompt-free paired objective.

        Instead of matching epsilon-space source->target differences directly,
        this predicts the clean latent x0 from a noised source latent and
        encourages that predicted x0 to move toward the paired target latent.

        This is still single-timestep training, so it is much cheaper than
        full 50-step reconstruction, but it gives a clearer target than
        epsilon-difference regression.
        """
        del sample  # kept for compatibility with previous version

        timestep = self._sample_timestep()

        # Noise only the source latent. The LoRA model must predict an edited x0.
        x_t_src, noise = self._add_noise(z0_src, timestep)

        # LoRA-edited denoiser output from source noisy latent
        eps_edit = self._compute_eps_from_branch(
            x_t_src,
            timestep,
            self.empty_branch,
            lora_multiplier=float(direction),
            cfg_scale=self.config.train_cfg_scale,
        )

        # Convert edited epsilon prediction to predicted clean latent
        x0_edit = self._predict_x0_from_eps(x_t_src, eps_edit, timestep)

        # --------------------------------------------------------------
        # 1) Direct paired target loss
        # --------------------------------------------------------------
        # Minimal success case: predicted edited clean latent should move
        # toward the paired target clean latent.
        loss_x0_target = F.mse_loss(x0_edit, z0_tgt)

        # --------------------------------------------------------------
        # 2) Directional latent-space alignment
        # --------------------------------------------------------------
        # Encourage the edit vector to point in the same direction as
        # source -> target, without requiring perfect reconstruction only.
        edit_delta = x0_edit - z0_src
        target_delta = z0_tgt - z0_src

        flat_edit = edit_delta.flatten(1)
        flat_target = target_delta.flatten(1)

        edit_norm = flat_edit.norm(dim=1, keepdim=True).clamp_min(1e-8)
        target_norm = flat_target.norm(dim=1, keepdim=True).clamp_min(1e-8)

        flat_edit_unit = flat_edit / edit_norm
        flat_target_unit = flat_target / target_norm

        cos_sim = F.cosine_similarity(flat_edit_unit, flat_target_unit, dim=1).mean()
        loss_x0_direction = 1.0 - cos_sim

        # Optional margin: force the edit to have non-trivial magnitude.
        # Here margin is relative to target_delta norm.
        if self.config.separation_margin > 0.0:
            loss_margin = torch.relu(
                self.config.separation_margin * target_norm.squeeze(1).detach()
                - edit_norm.squeeze(1)
            ).mean()
        else:
            loss_margin = torch.zeros((), device=x_t_src.device, dtype=x_t_src.dtype)

        loss_concept = (
            self.config.prompt_free_x0_target_weight * loss_x0_target
            + self.config.prompt_free_x0_direction_weight * loss_x0_direction
            + loss_margin
        )

        # --------------------------------------------------------------
        # 3) Optional denoising consistency
        # --------------------------------------------------------------
        # Keep this small, e.g. pair_loss_weight=0.0, 0.05, or 0.1.
        # Too much of it encourages "stay a normal denoiser / do nothing".
        loss_pair = F.mse_loss(eps_edit, noise)

        # --------------------------------------------------------------
        # 4) Optional identity preservation
        # --------------------------------------------------------------
        # For this diagnostic experiment, I would keep identity_loss_weight=0.
        # Otherwise the loss fights against the source->target movement.
        loss_identity = F.mse_loss(x0_edit, z0_src)

        loss_total = (
            self.config.separation_loss_weight * loss_concept
            + self.config.identity_loss_weight * loss_identity
            + self.config.pair_loss_weight * loss_pair
        )

        return loss_total, {
            "loss_total": float(loss_total.detach().item()),
            "loss_concept": float(loss_concept.detach().item()),
            "loss_identity": float(loss_identity.detach().item()),
            "loss_pair": float(loss_pair.detach().item()),
        }

    def _sample_reference_pairs(self, direction: int, exclude_sample: Optional[dict], sample_pool: List[dict], k: int):
        if k <= 0 or len(sample_pool) == 0:
            return []

        candidates = []
        exclude_source = None if exclude_sample is None else exclude_sample.get("source_path", None)

        for s in sample_pool:
            if int(s["direction"]) != int(direction):
                continue
            if s.get("target_path", None) is None:
                continue
            if exclude_source is not None and s.get("source_path", None) == exclude_source:
                continue
            candidates.append(s)

        if len(candidates) == 0:
            return []

        if len(candidates) <= k:
            return random.sample(candidates, len(candidates))

        return random.sample(candidates, k)

    def _compute_concept_slider_loss(
        self,
        zT_src,
        z0_src,
        z0_tgt,
        direction: int,
        mode: str = "random",
        sample: Optional[dict] = None,
        null_override: Optional[dict] = None,
    ):
        del zT_src, mode

        if self.config.training_regime == "prompt":
            return self._prompt_concept_slider_losses(
                z0_src=z0_src,
                direction=direction,
                null_override=null_override,
            )

        if self.config.training_regime == "prompt_free_paired":
            if z0_tgt is None:
                raise ValueError("prompt_free_paired regime requires paired target samples (z0_tgt is None).")
            return self._paired_prompt_free_losses(
                z0_src=z0_src,
                z0_tgt=z0_tgt,
                direction=direction,
                sample=sample,
            )

        raise ValueError(f"Unknown training_regime: {self.config.training_regime}")

    def forward_edit_latent(self, zT, direction: int, lora_multiplier: float):
        return self._forward_edit_with_multiplier(zT, lora_multiplier=lora_multiplier, direction=direction)

    def training_step_on_sample(self, sample):
        source = self.latent_cache[sample["source_path"]]
        zT_src = source["zT"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        z0_src = source["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        direction = int(sample["direction"])

        z0_tgt = None
        if sample.get("target_path", None) is not None:
            target = self.latent_cache[sample["target_path"]]
            z0_tgt = target["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)

        null_override = None
        if self.config.use_null_text_optimization:
            null_override = source.get("nti", None)
            if null_override is None:
                raise RuntimeError(
                    f"NTI requested but missing for source_path={sample['source_path']}. "
                    "Run trainer.precompute_null_text_cache(...) before training."
                )

        return self._compute_concept_slider_loss(
            zT_src=zT_src,
            z0_src=z0_src,
            z0_tgt=z0_tgt,
            direction=direction,
            mode="random",
            sample=sample,
            null_override=null_override,
        )

    @torch.no_grad()
    def validation_step_on_sample(self, sample):
        self.pipe.unet.eval()

        source = self.latent_cache[sample["source_path"]]
        zT_src = source["zT"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        z0_src = source["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)
        direction = int(sample["direction"])

        z0_tgt = None
        if sample.get("target_path", None) is not None:
            target = self.latent_cache[sample["target_path"]]
            z0_tgt = target["z0"].to(self.pipe.device, dtype=torch.float32, non_blocking=True)

        null_override = None
        if self.config.use_null_text_optimization:
            null_override = source.get("nti", None)
            if null_override is None:
                raise RuntimeError(
                    f"NTI requested but missing for source_path={sample['source_path']}. "
                    "Run trainer.precompute_null_text_cache(...) before validation."
                )

        loss, logs = self._compute_concept_slider_loss(
            zT_src=zT_src,
            z0_src=z0_src,
            z0_tgt=z0_tgt,
            direction=direction,
            mode="random",
            sample=sample,
            null_override=null_override,
        )

        del zT_src, z0_src, loss
        if z0_tgt is not None:
            del z0_tgt
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

            self._current_sample_pool = train_samples
            self._current_stage = "train"

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
                        pair=f"{logs['loss_pair']:.6f}",
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
                self._current_sample_pool = val_samples
                self._current_stage = "val"

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
            z0_edit = self.forward_edit_latent(zT_src, direction=direction, lora_multiplier=float(s))
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

    def get_default_nti_cache_dir(self) -> Path:
        dir_name = self.config.nti_cache_dir_name or (
            f"nti_cache_steps{self.config.train_steps}"
            f"_inner{self.config.nti_inner_steps}"
            f"_lr{self.config.nti_lr:g}"
            f"_opt_{self.config.nti_optimize}"
        )
        return Path(self.config.out_dir) / dir_name

    def get_nti_cache_file(self, audio_path: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self.make_latent_cache_key(audio_path)
        stem = Path(audio_path).stem
        return cache_dir / f"{stem}_{key}_nti.pt"

    def load_nti_cache_item(self, cache_file: Path):
        payload = torch.load(cache_file, map_location="cpu")
        return payload

    def save_nti_cache_item(self, cache_file: Path, audio_path: str, per_step: List[dict], losses: List[float]):
        payload = {
            "audio_path": audio_path,
            "train_steps": int(self.config.train_steps),
            "nti_inner_steps": int(self.config.nti_inner_steps),
            "nti_lr": float(self.config.nti_lr),
            "nti_prompt": str(self.config.nti_prompt),
            "nti_optimize": str(self.config.nti_optimize),
            "per_step": per_step,
            "losses": losses,
        }
        torch.save(payload, cache_file)

    @torch.no_grad()
    def _compute_ddim_inversion_pivots(self, z0, invert_prompt: str = ""):
        """
        Returns inversion pivots in forward-denoising order:

            pivots[0]  = z_T
            pivots[-1] = z_0

        This is the trajectory NTI will try to reconstruct step by step.
        """
        self.pipe.unet.to(self.pipe.device, dtype=torch.float32)
        self.pipe.unet.eval()
        
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = encode_prompt_fixed(
            self.pipe,
            invert_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )

        inv_timesteps = self.timesteps.flip(0)
        alphas_cumprod = build_alpha_cache(self.pipe)

        x = z0.detach().to(self.pipe.device, dtype=torch.float32)
        inv_pivots = [x.detach().clone()]

        for i in tqdm(range(len(inv_timesteps) - 1), desc="NTI | DDIM inversion pivots", leave=False):
            t = inv_timesteps[i]
            t_next = inv_timesteps[i + 1]

            eps = predict_eps_for_inversion(
                self.pipe,
                x,
                t,
                do_cfg=False,
                cfg_scale=1.0,
                prompt_embeds_uncond=pu_inv,
                prompt_embeds_text=pt_inv,
                attn_uncond=au_inv,
                attn_text=at_inv,
                gen_uncond=gu_inv,
                gen_text=gt_inv,
            )

            x = ddim_inverse_step_scheduler_aware(
                self.pipe,
                alphas_cumprod,
                x,
                eps,
                t,
                t_next,
            )
            inv_pivots.append(x.detach().clone())

        return list(reversed(inv_pivots))

    def optimize_null_text_for_cached_item(self, audio_path: str):
        """
        Per-instance NTI.

        Optimizes unconditional prompt tensors per denoising step so that
        the forward trajectory reconstructs the DDIM inversion pivots.
        """
        if audio_path not in self.latent_cache:
            raise KeyError(f"{audio_path} not found in latent_cache. Run precompute_latent_cache first.")

        item = self.latent_cache[audio_path]
        z0 = item["z0"].to(self.pipe.device, dtype=torch.float32)

        device = self.editor.device if hasattr(self.editor, "device") else self.pipe.device
        self._ensure_unet_on_device(device)
        set_all_lora_multipliers(self.pipe.unet, 0.0)

        branch = encode_prompt_fixed(
            self.pipe,
            self.config.nti_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )
        pt, am, gt, pu_base, amu, gu_base = branch

        pivots = self._compute_ddim_inversion_pivots(z0, invert_prompt=self.config.nti_prompt)
        extra_step_kwargs = self.extra_step_kwargs

        per_step = []
        losses_out = []

        x = pivots[0].detach().clone()
        pt = pt.to(x.device)
        am = am.to(x.device) if am is not None else None
        gt = gt.to(x.device)
        amu = amu.to(x.device) if amu is not None else None
        device = x.device
        self._ensure_unet_on_device(device)
        step_iter = tqdm(
            range(len(self.timesteps) - 1),
            desc=f"NTI | optimizing nulls | {Path(audio_path).name}",
            leave=True,
        )
        print("DEBUG x device:", x.device)
        print("DEBUG unet device:", next(self.pipe.unet.parameters()).device)
        print("DEBUG time embedding device:", self.pipe.unet.time_embedding.linear_1.weight.device)
        for i in step_iter:
            t = self.timesteps[i]
            target_prev = pivots[i + 1].detach()

            if self.config.nti_optimize in ("pu", "both"):
                pu_opt = pu_base.detach().clone().requires_grad_(True)
            else:
                pu_opt = pu_base.detach()

            if self.config.nti_optimize in ("gu", "both"):
                gu_opt = gu_base.detach().clone().requires_grad_(True)
            else:
                gu_opt = gu_base.detach()

            params = []
            if isinstance(pu_opt, torch.Tensor) and pu_opt.requires_grad:
                params.append(pu_opt)
            if isinstance(gu_opt, torch.Tensor) and gu_opt.requires_grad:
                params.append(gu_opt)

            if len(params) == 0:
                raise ValueError("nti_optimize must be one of {'pu', 'gu', 'both'}.")

            opt = torch.optim.Adam(params, lr=self.config.nti_lr)

            last_loss = None

            for _ in range(self.config.nti_inner_steps):
                opt.zero_grad(set_to_none=True)
                self._ensure_unet_on_device(x.device)
                eps = predict_eps_cfg(
                    self.pipe,
                    x,
                    t,
                    pu_opt,
                    pt,
                    amu,
                    am,
                    gu_opt,
                    gt,
                    self.config.train_cfg_scale,
                    force_cpu_unet=False,
                    cpu_grad_mode=False,
                )

                x_prev = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample
                loss = F.mse_loss(x_prev, target_prev)
                loss.backward()
                opt.step()

                last_loss = float(loss.detach().item())

            with torch.no_grad():
                self._ensure_unet_on_device(x.device)
                eps = predict_eps_cfg(
                    self.pipe,
                    x,
                    t,
                    pu_opt.detach(),
                    pt,
                    amu,
                    am,
                    gu_opt.detach(),
                    gt,
                    self.config.train_cfg_scale,
                    force_cpu_unet=False,
                    cpu_grad_mode=False,
                )
                x = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample.detach()

            per_step.append({
                "pu": pu_opt.detach().cpu() if self.config.nti_optimize in ("pu", "both") else None,
                "gu": gu_opt.detach().cpu() if self.config.nti_optimize in ("gu", "both") else None,
            })
            losses_out.append(last_loss)

            step_iter.set_postfix(loss=f"{last_loss:.6f}")

            del opt, params, pu_opt, gu_opt, eps, x_prev, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "audio_path": audio_path,
            "per_step": per_step,
            "losses": losses_out,
        }

    def precompute_null_text_cache(
        self,
        samples,
        cache_dir: Optional[str | Path] = None,
        overwrite: Optional[bool] = None,
    ):
        """
        Computes or loads per-instance NTI caches for every source_path used
        during LoRA training/validation.

        Must be called after precompute_latent_cache().
        """
        overwrite = self.config.nti_overwrite if overwrite is None else overwrite
        cache_dir = Path(cache_dir) if cache_dir is not None else self.get_default_nti_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_paths = sorted({str(s["source_path"]) for s in samples if s.get("source_path", None) is not None})

        print(f"Unique source files for NTI: {len(all_paths)}")
        print(f"NTI cache dir: {cache_dir}")
        print(f"NTI config: inner_steps={self.config.nti_inner_steps}, lr={self.config.nti_lr}, optimize={self.config.nti_optimize}")

        for idx, path in enumerate(all_paths, start=1):
            cache_file = self.get_nti_cache_file(path, cache_dir)

            print("\n" + "=" * 80)
            print(f"NTI item {idx}/{len(all_paths)}")
            print(f"path: {path}")

            if cache_file.exists() and not overwrite:
                payload = self.load_nti_cache_item(cache_file)
                self.latent_cache[path]["nti"] = payload
                print(f"Loaded cached NTI: {cache_file}")
                continue

            result = self.optimize_null_text_for_cached_item(path)
            self.save_nti_cache_item(
                cache_file=cache_file,
                audio_path=path,
                per_step=result["per_step"],
                losses=result["losses"],
            )

            self.latent_cache[path]["nti"] = self.load_nti_cache_item(cache_file)

            print(f"Saved NTI: {cache_file}")
            if len(result["losses"]) > 0:
                print(f"Final NTI step loss: {result['losses'][-1]:.6f}")
                print(f"Mean NTI step loss:  {sum(result['losses']) / len(result['losses']):.6f}")

            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nNTI cache ready.")