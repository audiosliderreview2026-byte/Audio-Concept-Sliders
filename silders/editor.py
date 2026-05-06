from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from diffusers import AudioLDM2Pipeline
from diffusers.models.attention_processor import AttnProcessor
from tqdm.auto import tqdm

from .frontend import waveform_to_logmel_exact_audioldm
from .prompts import encode_prompt_fixed
from .diffusion import (
    build_alpha_cache,
    ddim_inverse_step_scheduler_aware,
    hard_disable_checkpointing,
    freeze_module_params,
    predict_eps_cfg,
    predict_eps_for_inversion,
)
from .utils import decode_latents_to_audio, load_mono_audio
from .lora import LoRAConfig, inject_lora_into_unet, freeze_non_lora_params


@dataclass
class EditorConfig:
    model_path: str
    device: Optional[str] = None
    torch_dtype: torch.dtype = torch.float32
    target_audio_length_s: float = 10.0
    steps: int = 50
    max_new_tokens: int = 8
    cpu_unet_opt: bool = True
    cpu_unet_edit: bool = False
    shared_iters: int = 10
    shared_lr: float = 5e-3
    perstep_iters: int = 3
    perstep_lr: float = 1e-2
    planb_optimize: str = "gu"

    def resolve_device(self):
        if self.device is not None:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class AudioLDM2Editor:
    def __init__(self, config: EditorConfig):
        self.config = config
        self.device = config.resolve_device()
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=config.torch_dtype,
        ).to(self.device)

        try:
            self.pipe.disable_attention_slicing()
        except Exception:
            pass

        self.pipe.unet.set_attn_processor(AttnProcessor())
        self.pipe.unet.to(dtype=torch.float32)
        self.pipe.text_encoder.to(dtype=torch.float32)
        self.pipe.text_encoder_2.to(dtype=torch.float32)
        self.pipe.projection_model.to(dtype=torch.float32)
        self.pipe.language_model.to(dtype=torch.float32)

        self.prompt_cache: Dict[tuple, tuple] = {}

    def setup_scheduler(self, steps: Optional[int] = None):
        steps = self.config.steps if steps is None else steps
        self.pipe.scheduler.set_timesteps(steps, device=self.pipe.device)
        timesteps = self.pipe.scheduler.timesteps
        inv_timesteps = timesteps.flip(0)
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
        return timesteps, inv_timesteps, extra_step_kwargs

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        key = (prompt, negative_prompt, self.config.max_new_tokens)
        if key not in self.prompt_cache:
            self.prompt_cache[key] = encode_prompt_fixed(
                self.pipe,
                prompt,
                negative_prompt=negative_prompt,
                max_new_tokens=self.config.max_new_tokens,
            )
        return self.prompt_cache[key]

    def load_audio(self, audio_path: str):
        target_sr = int(self.pipe.vocoder.config.sampling_rate)
        wav, sr = load_mono_audio(
            audio_path,
            target_sr=target_sr,
            duration_sec=self.config.target_audio_length_s,
        )
    
        peak = max(abs(wav.min()), abs(wav.max()))
        if peak > 1.0:
            wav = wav / peak
    
        wav = wav.clip(-1.0, 1.0)
        return wav, sr
    def edit(
        self,
        audio_path: str,
        invert_prompt: str = "",
        edit_prompt: str = "",
        plan: str = "A",
        cfg_scale: float = 3.5,
    ):
        """
        High-level convenience wrapper for prompt-based editing.
    
        Returns a dict containing:
          - input wav / sr
          - z0, zT, z0_hat, z0_edit
          - decoded wavs:
              orig, vae_roundtrip, inv_roundtrip, zT_decode, edited
          - prompts / config info
        """
        # ---------------------------------
        # Load + encode input audio
        # ---------------------------------
        wav, sr = self.load_audio(audio_path)
        z0 = self.encode_audio_to_z0(wav, sr)
    
        # ---------------------------------
        # Inversion
        # ---------------------------------
        zT = self.invert(z0, invert_prompt=invert_prompt)
    
        # ---------------------------------
        # Sanity forward reconstruction
        # ---------------------------------
        z0_hat = self.sanity_forward(zT, invert_prompt=invert_prompt)
    
        # ---------------------------------
        # Encode prompts
        # ---------------------------------
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = self.encode_prompt(invert_prompt, "")
        pt_edit, am_edit, gt_edit, pu_edit, amu_edit, gu_edit = self.encode_prompt(edit_prompt, "")
    
        # ---------------------------------
        # Run selected regime
        # ---------------------------------
        z0_edit = self.run_plan_for_cfg(
            plan_name=plan,
            cfg_scale=cfg_scale,
            z0=z0,
            zT=zT,
            prompt_embeds_text=pt_inv,
            prompt_embeds_uncond=pu_inv,
            attn_text=at_inv,
            attn_uncond=au_inv,
            gen_text=gt_inv,
            gen_uncond=gu_inv,
            pt_edit=pt_edit,
            am_edit=am_edit,
            gt_edit=gt_edit,
            pu_edit=pu_edit,
            amu_edit=amu_edit,
            gu_edit=gu_edit,
        )
    
        # ---------------------------------
        # Decode useful audio outputs
        # ---------------------------------
        wav_vae = self.decode_latents(z0, wav_len=len(wav))
        wav_inv = self.decode_latents(z0_hat, wav_len=len(wav))
        wav_noise = self.decode_latents(zT, wav_len=len(wav))
        wav_edit = self.decode_latents(z0_edit, wav_len=len(wav))
    
        # ---------------------------------
        # Reconstruction metrics
        # ---------------------------------
        rmse = (z0_hat - z0).pow(2).mean().sqrt().item()
        rel = rmse / (z0.pow(2).mean().sqrt().item() + 1e-8)
    
        return {
            "audio_path": audio_path,
            "sr": sr,
            "wav_len": len(wav),
            "invert_prompt": invert_prompt,
            "edit_prompt": edit_prompt,
            "plan": plan,
            "cfg_scale": cfg_scale,
            "z0": z0.detach(),
            "zT": zT.detach(),
            "z0_hat": z0_hat.detach(),
            "z0_edit": z0_edit.detach(),
            "wav_orig": wav,
            "wav_vae": wav_vae,
            "wav_inv": wav_inv,
            "wav_noise": wav_noise,
            "wav_edit": wav_edit,
            "rmse_inv": rmse,
            "rel_rmse_inv": rel,
        }
    def encode_audio_to_z0(self, wav, sr: int):
        _, mel_for_vae, _ = waveform_to_logmel_exact_audioldm(
            wav, sr, self.pipe, duration_sec=self.config.target_audio_length_s
        )
        mel_tensor = torch.from_numpy(mel_for_vae).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            posterior = self.pipe.vae.encode(mel_tensor).latent_dist
            z0 = posterior.sample()
            z0 = z0 * self.pipe.vae.config.scaling_factor
        return z0

    def invert(self, z0, invert_prompt: str = ""):
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = self.encode_prompt(invert_prompt, "")
        _, inv_timesteps, _ = self.setup_scheduler()
        alphas_cumprod = build_alpha_cache(self.pipe)

        x = z0.clone()
        for i in tqdm(range(len(inv_timesteps) - 1), desc="DDIM inversion", leave=False):
            t = inv_timesteps[i]
            t_next = inv_timesteps[i + 1]
            model_out = predict_eps_for_inversion(
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
            x = ddim_inverse_step_scheduler_aware(self.pipe, alphas_cumprod, x, model_out, t, t_next)
        return x

    def encode_and_invert_audio(self, audio_path: str, invert_prompt: str = ""):
        wav, sr = self.load_audio(audio_path)
        z0 = self.encode_audio_to_z0(wav, sr)
        zT = self.invert(z0, invert_prompt=invert_prompt)
        return {"wav": wav, "sr": sr, "wav_len": len(wav), "z0": z0.detach(), "zT": zT.detach()}

    def sanity_forward(self, zT, invert_prompt: str = ""):
        pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = self.encode_prompt(invert_prompt, "")
        timesteps, _, extra_step_kwargs = self.setup_scheduler()
        with torch.no_grad():
            x = zT.clone()
            for t in timesteps:
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
                x = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample
        return x

    def decode_latents(self, latents, wav_len: Optional[int] = None):
        return decode_latents_to_audio(self.pipe, latents, wav_len=wav_len)

    def inject_lora(self, lora_config: LoRAConfig, verbose: bool = True):
        names = inject_lora_into_unet(self.pipe.unet, lora_config, verbose=verbose)
        self.pipe.unet.to(self.pipe.device, dtype=torch.float32)
        freeze_non_lora_params(self.pipe.unet)
        return names

    def run_plan_for_cfg(
        self,
        plan_name: str,
        cfg_scale: float,
        z0,
        zT,
        prompt_embeds_text,
        prompt_embeds_uncond,
        attn_text,
        attn_uncond,
        gen_text,
        gen_uncond,
        pt_edit,
        am_edit,
        gt_edit,
        pu_edit,
        amu_edit,
        gu_edit,
    ):
        timesteps, _, extra_step_kwargs = self.setup_scheduler()

        print(f"PLAN {plan_name} | CFG {cfg_scale}")
        self.pipe.unet.to(self.pipe.device, dtype=torch.float32)
        self.pipe.unet.eval()

        pu_shared = None
        gu_shared = None
        pu_steps = None
        gu_steps = None

        self.pipe.unet = hard_disable_checkpointing(self.pipe.unet)
        freeze_module_params(self.pipe.unet)
        self.pipe.unet.eval()

        cfg = self.config

        if plan_name == "B":
            if cfg.planb_optimize in ("both", "pu"):
                pu_shared = prompt_embeds_uncond.clone().detach().requires_grad_(True)
            else:
                pu_shared = prompt_embeds_uncond.detach()
            if cfg.planb_optimize in ("both", "gu"):
                gu_shared = gen_uncond.clone().detach().requires_grad_(True)
            else:
                gu_shared = gen_uncond.detach()

            params = []
            if isinstance(pu_shared, torch.Tensor) and pu_shared.requires_grad:
                params.append(pu_shared)
            if isinstance(gu_shared, torch.Tensor) and gu_shared.requires_grad:
                params.append(gu_shared)
            opt = torch.optim.Adam(params, lr=cfg.shared_lr)

            for _ in range(cfg.shared_iters):
                opt.zero_grad(set_to_none=True)
                x = zT.clone()
                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    eps = predict_eps_cfg(
                        self.pipe, x, t, pu_shared, prompt_embeds_text, attn_uncond, attn_text,
                        gu_shared, gen_text, cfg_scale, force_cpu_unet=False, cpu_grad_mode=False
                    )
                    x = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample
                loss = torch.nn.functional.mse_loss(x, z0)
                loss.backward()
                opt.step()

            pu_shared = pu_shared.detach()
            gu_shared = gu_shared.detach()

        if plan_name == "C":
            traj_fwd = [zT.clone()]
            with torch.no_grad():
                x = zT.clone()
                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    eps = predict_eps_for_inversion(
                        self.pipe,
                        x,
                        t,
                        do_cfg=False,
                        cfg_scale=1.0,
                        prompt_embeds_uncond=prompt_embeds_uncond,
                        prompt_embeds_text=prompt_embeds_text,
                        attn_uncond=attn_uncond,
                        attn_text=attn_text,
                        gen_uncond=gen_uncond,
                        gen_text=gen_text,
                    )
                    x = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample
                    traj_fwd.append(x.clone())

            pu_steps, gu_steps = [], []
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                x_t = traj_fwd[i]
                x_target = traj_fwd[i + 1]

                if cfg.planb_optimize in ("both", "pu"):
                    pu = prompt_embeds_uncond.clone().detach().requires_grad_(True)
                else:
                    pu = prompt_embeds_uncond.detach()

                if cfg.planb_optimize in ("both", "gu"):
                    gu = gen_uncond.clone().detach().requires_grad_(True)
                else:
                    gu = gen_uncond.detach()

                params = []
                if isinstance(pu, torch.Tensor) and pu.requires_grad:
                    params.append(pu)
                if isinstance(gu, torch.Tensor) and gu.requires_grad:
                    params.append(gu)
                opt = torch.optim.Adam(params, lr=cfg.perstep_lr)

                for _ in range(cfg.perstep_iters):
                    opt.zero_grad(set_to_none=True)
                    eps = predict_eps_cfg(
                        self.pipe, x_t, t, pu, prompt_embeds_text, attn_uncond, attn_text,
                        gu, gen_text, cfg_scale, force_cpu_unet=False, cpu_grad_mode=False
                    )
                    x_prev = self.pipe.scheduler.step(eps, t, x_t, **extra_step_kwargs).prev_sample
                    loss = torch.nn.functional.mse_loss(x_prev, x_target)
                    loss.backward()
                    opt.step()

                pu_steps.append(pu.detach())
                gu_steps.append(gu.detach())

        with torch.no_grad():
            x = zT.clone()
            for i in tqdm(range(len(timesteps) - 1), desc=f"Forward edit | {plan_name} | CFG={cfg_scale}", leave=False):
                t = timesteps[i]
                if plan_name == "A":
                    pu_i, amu_i, gu_i = pu_edit, amu_edit, gu_edit
                elif plan_name == "B":
                    pu_i, amu_i, gu_i = pu_shared, amu_edit, gu_shared
                else:
                    pu_i, amu_i, gu_i = pu_steps[i], amu_edit, gu_steps[i]

                eps = predict_eps_cfg(
                    self.pipe,
                    x,
                    t,
                    pu_i,
                    pt_edit,
                    amu_i,
                    am_edit,
                    gu_i,
                    gt_edit,
                    cfg_scale,
                    force_cpu_unet=cfg.cpu_unet_edit,
                    cpu_grad_mode=False,
                )
                x = self.pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample

        return x
