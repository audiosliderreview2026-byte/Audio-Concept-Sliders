from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import random
import time

import torch
import soundfile as sf
from tqdm.auto import tqdm

from .utils import load_mono_audio, decode_latents_to_audio
from .frontend import waveform_to_logmel_exact_audioldm
from .prompts import encode_prompt_fixed
from .diffusion import build_alpha_cache, ddim_inverse_step_scheduler_aware, predict_eps_cfg, predict_eps_for_inversion
from .lora import freeze_non_lora_params, export_lora_state_dict, set_all_lora_multipliers, load_lora_state_dict


@dataclass
class TrainerConfig:
    dataset_root: str = "lora_dataset"
    train_songs: Optional[List[str]] = None
    val_songs: Optional[List[str]] = None
    prompt_piano: str = "solo piano instrumental music"
    prompt_guitar: str = "solo acoustic guitar instrumental music"
    train_audio_length_s: float = 10.0
    train_cfg_scale: float = 3.5
    train_steps: int = 200
    train_max_new_tokens: int = 8
    num_epochs: int = 10
    lr_lora: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    max_train_pairs: Optional[int] = None
    max_val_pairs: Optional[int] = None
    out_dir: str = "lora_slider_training"
    checkpoint_every: int = 1
    seed: int = 42


class LoRASliderTrainer:
    def __init__(self, editor, config: TrainerConfig):
        self.editor = editor
        self.pipe = editor.pipe
        self.config = config
        self.latent_cache: Dict[str, dict] = {}
        self.history = {"train": [], "val": []}
        self._resume_checkpoint_path: Optional[str] = None
        self._resume_state: Optional[dict] = None
        self._resume_epoch: int = 0
        self._resume_best_val: float = float("inf")

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.pipe.scheduler.set_timesteps(config.train_steps, device=self.pipe.device)
        self.timesteps = self.pipe.scheduler.timesteps
        self.extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

        self.pt_piano, self.am_piano, self.gt_piano, self.pu_piano, self.amu_piano, self.gu_piano = encode_prompt_fixed(
            self.pipe, config.prompt_piano, negative_prompt="", max_new_tokens=config.train_max_new_tokens
        )
        self.pt_guitar, self.am_guitar, self.gt_guitar, self.pu_guitar, self.amu_guitar, self.gu_guitar = encode_prompt_fixed(
            self.pipe, config.prompt_guitar, negative_prompt="", max_new_tokens=config.train_max_new_tokens
        )
    from pathlib import Path
    import hashlib
    import torch
    
    
    def make_latent_cache_key(audio_path: str) -> str:
        return hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    
    
    def get_latent_cache_file(cache_dir, audio_path: str) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = make_latent_cache_key(audio_path)
        stem = Path(audio_path).stem
        return cache_dir / f"{stem}_{key}.pt"
    
    
    def save_latent_cache_item(cache_file, audio_path, sr, wav, z0, zT):
        payload = {
            "audio_path": audio_path,
            "sr": int(sr),
            "wav_len": int(len(wav)),
            "z0": z0.detach().cpu(),
            "zT": zT.detach().cpu(),
        }
        torch.save(payload, cache_file)
    
    
    def load_latent_cache_item(cache_file, device):
        payload = torch.load(cache_file, map_location="cpu")
        return {
            "wav": None,  # optional, keep None if you don't need waveform now
            "sr": payload["sr"],
            "wav_len": payload["wav_len"],
            "z0": payload["z0"].to(device=device, dtype=torch.float32),
            "zT": payload["zT"].to(device=device, dtype=torch.float32),
        }
    def build_optimizer(self):
        freeze_non_lora_params(self.pipe.unet)
        self.pipe.unet.to(self.pipe.device, dtype=torch.float32)
        self.pipe.unet.train()
        params = [p for p in self.pipe.unet.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.lr_lora, weight_decay=self.config.weight_decay)

    @torch.no_grad()
    def encode_wav_to_z0(self, wav, sr):
        _, mel_for_vae, _ = waveform_to_logmel_exact_audioldm(
            wav, sr, self.pipe, duration_sec=self.config.train_audio_length_s
        )
        mel_tensor = torch.from_numpy(mel_for_vae).to(device=self.pipe.device, dtype=torch.float32)
        posterior = self.pipe.vae.encode(mel_tensor).latent_dist
        z0 = posterior.sample()
        z0 = z0 * self.pipe.vae.config.scaling_factor
        return z0.detach()

    @torch.no_grad()
    def invert_z0_to_zT(self, z0, invert_prompt: str = ""):
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = encode_prompt_fixed(
            self.pipe,
            invert_prompt,
            negative_prompt="",
            max_new_tokens=self.config.train_max_new_tokens,
        )
        inv_timesteps = self.pipe.scheduler.timesteps.flip(0)
        alphas_cumprod = build_alpha_cache(self.pipe)
        x = z0.clone()
        for i in range(len(inv_timesteps) - 1):
            t = inv_timesteps[i]
            t_next = inv_timesteps[i + 1]
            model_out = predict_eps_for_inversion(
                self.pipe, x, t, do_cfg=False, cfg_scale=1.0,
                prompt_embeds_uncond=pu_inv, prompt_embeds_text=pt_inv,
                attn_uncond=au_inv, attn_text=at_inv,
                gen_uncond=gu_inv, gen_text=gt_inv,
            )
            x = ddim_inverse_step_scheduler_aware(self.pipe, alphas_cumprod, x, model_out, t, t_next)
        return x.detach()

    def precompute_latent_cache(pairs, pipe, invert_prompt="", cache_dir="latent_cache", store_wav=False):
        pipe.scheduler.set_timesteps(TRAIN_STEPS, device=pipe.device)
    
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = encode_prompt_fixed(
            pipe,
            invert_prompt,
            negative_prompt="",
            max_new_tokens=TRAIN_MAX_NEW_TOKENS,
        )
    
        all_paths = sorted({
            item["piano"] for item in pairs
        } | {
            item["guitar"] for item in pairs
        })
    
        print(f"Unique wav files to cache: {len(all_paths)}")
    
        for path in tqdm(all_paths, desc="Precomputing / loading latents"):
            if path in latent_cache:
                continue
    
            cache_file = get_latent_cache_file(cache_dir, path)
    
            if cache_file.exists():
                item = load_latent_cache_item(cache_file, device=pipe.device)
    
                if store_wav:
                    wav, sr = load_audio_for_training(path, pipe, target_audio_length_s=TRAIN_AUDIO_LENGTH_S)
                    item["wav"] = wav
                    item["sr"] = sr
    
                latent_cache[path] = item
                continue
    
            wav, sr = load_audio_for_training(path, pipe, target_audio_length_s=TRAIN_AUDIO_LENGTH_S)
            z0 = encode_wav_to_z0(pipe, wav, sr, duration_sec=TRAIN_AUDIO_LENGTH_S)
            zT = invert_z0_to_zT(
                pipe,
                z0,
                pt_inv, pu_inv,
                at_inv, au_inv,
                gt_inv, gu_inv,
            )
    
            save_latent_cache_item(cache_file, path, sr, wav, z0, zT)
    
            latent_cache[path] = {
                "wav": wav if store_wav else None,
                "sr": sr,
                "wav_len": len(wav),
                "z0": z0.detach(),
                "zT": zT.detach(),
            }
    
        print("Latent cache ready.")
        print(f"Cached items in memory: {len(latent_cache)}")

    def get_endpoint_prompt_tensors(self, direction: int):
        if direction == +1:
            return self.pt_piano, self.am_piano, self.gt_piano, self.pu_piano, self.amu_piano, self.gu_piano
        if direction == -1:
            return self.pt_guitar, self.am_guitar, self.gt_guitar, self.pu_guitar, self.amu_guitar, self.gu_guitar
        raise ValueError("direction must be +1 or -1")

    def forward_edit_latent(self, zT, direction: int, lora_multiplier: float):
        pt_edit, am_edit, gt_edit, pu_edit, amu_edit, gu_edit = self.get_endpoint_prompt_tensors(direction)
        set_all_lora_multipliers(self.pipe.unet, lora_multiplier)
        x = zT
        from tqdm.auto import tqdm

        for i in tqdm(range(len(timesteps) - 1), desc="Forward edit", leave=False):
            t = self.timesteps[i]
            eps = predict_eps_cfg(
                self.pipe, x, t, pu_edit, pt_edit, amu_edit, am_edit, gu_edit, gt_edit,
                self.config.train_cfg_scale, force_cpu_unet=False, cpu_grad_mode=False
            )
            x = self.pipe.scheduler.step(eps, t, x, **self.extra_step_kwargs).prev_sample
        return x

    def training_step_on_sample(self, sample):
        source = self.latent_cache[sample["source_path"]]
        target = self.latent_cache[sample["target_path"]]
        zT_src = source["zT"].detach().to(self.pipe.device, dtype=torch.float32)
        z0_tgt = target["z0"].detach().to(self.pipe.device, dtype=torch.float32)
        direction = int(sample["direction"])
        z0_edit = self.forward_edit_latent(zT_src, direction=direction, lora_multiplier=float(direction))
        loss = torch.nn.functional.mse_loss(z0_edit, z0_tgt)
        return loss, {"loss_total": float(loss.detach().item())}

    @torch.no_grad()
    def validation_step_on_sample(self, sample):
        self.pipe.unet.eval()
        source = self.latent_cache[sample["source_path"]]
        target = self.latent_cache[sample["target_path"]]
        zT_src = source["zT"].detach().to(self.pipe.device, dtype=torch.float32)
        z0_tgt = target["z0"].detach().to(self.pipe.device, dtype=torch.float32)
        direction = int(sample["direction"])
        z0_edit = self.forward_edit_latent(zT_src, direction=direction, lora_multiplier=float(direction))
        loss = torch.nn.functional.mse_loss(z0_edit, z0_tgt)
        self.pipe.unet.train()
        return {"loss_total": float(loss.item())}

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
            train_count = 0
            t0 = time.time()

            pbar = tqdm(train_samples, desc=f"Epoch {epoch}/{self.config.num_epochs} [train]")
            for sample in pbar:
                optimizer.zero_grad(set_to_none=True)
                loss, logs = self.training_step_on_sample(sample)
                loss.backward()

                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.pipe.unet.parameters() if p.requires_grad],
                        self.config.grad_clip_norm,
                    )

                optimizer.step()
                train_loss_sum += logs["loss_total"]
                train_count += 1
                pbar.set_postfix(loss=f"{logs['loss_total']:.6f}", avg=f"{train_loss_sum / train_count:.6f}")

            train_avg = train_loss_sum / max(train_count, 1)
            self.history["train"].append(train_avg)

            if len(val_samples) > 0:
                val_loss_sum = 0.0
                val_count = 0
                for sample in tqdm(val_samples, desc=f"Epoch {epoch}/{self.config.num_epochs} [val]"):
                    logs = self.validation_step_on_sample(sample)
                    val_loss_sum += logs["loss_total"]
                    val_count += 1
                val_avg = val_loss_sum / max(val_count, 1)
            else:
                val_avg = float("nan")
            self.history["val"].append(val_avg)

            print(f"Epoch {epoch}/{self.config.num_epochs} done in {time.time()-t0:.1f}s")
            print(f"  train loss: {train_avg:.6f}")
            print(f"  val   loss: {val_avg:.6f}")

            if (epoch % self.config.checkpoint_every) == 0:
                ckpt_path = out_dir / f"lora_epoch_{epoch:03d}.pt"
                torch.save(export_lora_state_dict(self.pipe.unet), ckpt_path)

                full_state_path = out_dir / f"training_state_epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val": best_val,
                        "history": self.history,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lora_state_dict": export_lora_state_dict(self.pipe.unet),
                    },
                    full_state_path,
                )
                print("  saved:", ckpt_path)
                print("  saved training state:", full_state_path)

            if len(val_samples) > 0 and val_avg < best_val:
                best_val = val_avg
                best_path = out_dir / "lora_best.pt"
                torch.save(export_lora_state_dict(self.pipe.unet), best_path)

                best_state_path = out_dir / "training_state_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val": best_val,
                        "history": self.history,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lora_state_dict": export_lora_state_dict(self.pipe.unet),
                    },
                    best_state_path,
                )
                print("  new best:", best_path)
                print("  new best training state:", best_state_path)

        return self.history

    @torch.no_grad()
    def run_slider_inference_on_source(self, source_path, strengths=(-2.0, -1.0, 0.0, 1.0, 2.0)):
        self.pipe.unet.eval()
        source = self.latent_cache[source_path]
        zT_src = source["zT"].detach().to(self.pipe.device, dtype=torch.float32)
        outputs = []
        for s in strengths:
            direction = +1 if s >= 0 else -1
            z0_edit = self.forward_edit_latent(zT_src, direction=direction, lora_multiplier=float(s))
            wav_edit = decode_latents_to_audio(self.pipe, z0_edit, wav_len=len(source["wav"]))
            outputs.append({"strength": float(s), "wav": wav_edit, "sr": source["sr"]})
        return {"source_wav": source["wav"], "source_sr": source["sr"], "outputs": outputs}
