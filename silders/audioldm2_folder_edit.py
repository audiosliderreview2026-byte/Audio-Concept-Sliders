#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from diffusers import AudioLDM2Pipeline
from diffusers.models.attention_processor import AttnProcessor
from transformers import VitsModel


# =========================
# Basic helpers
# =========================

def peak_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    return x if m < eps else x / m


def sanitize_audio(x: np.ndarray, name: str = "audio") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n_nan = np.isnan(x).sum()
    n_inf = np.isinf(x).sum()
    if n_nan or n_inf:
        print(f"[WARN] {name}: NaN={n_nan} Inf={n_inf} -> replacing with 0")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mx = np.max(np.abs(x)) if x.size else 0.0
    if mx > 1e3:
        print(f"[WARN] {name}: huge peak {mx:.2e} -> soft clipping")
        x = np.tanh(x / mx)
    return x


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = sanitize_audio(wav, name=str(path))
    wav = peak_normalize(wav)
    sf.write(str(path), wav, sr)


@torch.no_grad()
def decode_latents_to_audio(pipe: AudioLDM2Pipeline, latents: torch.Tensor, wav_len: int | None = None) -> np.ndarray:
    mel = pipe.vae.decode((1.0 / pipe.vae.config.scaling_factor) * latents).sample
    wav_out = pipe.mel_spectrogram_to_waveform(mel).cpu().numpy()[0]
    if wav_len is not None:
        wav_out = wav_out[:wav_len]
    return wav_out


# =========================
# Exact AudioLDM frontend
# =========================
def get_exact_audioldm_frontend():
    from audioldm.audio.stft import TacotronSTFT

    sr = 16000
    n_fft = 1024
    hop_length = 160
    win_length = 1024
    n_mels = 64
    mel_fmin = 0
    mel_fmax = 8000

    stft = TacotronSTFT(
        filter_length=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mels,
        sampling_rate=sr,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )
    return stft, sr, n_mels, n_fft, hop_length, win_length


def get_target_mel_height(pipe: AudioLDM2Pipeline, audio_length_s: float) -> int:
    upsample_rates = pipe.vocoder.config.upsample_rates
    vocoder_upsample_factor = np.prod(upsample_rates) / pipe.vocoder.config.sampling_rate

    height = int(audio_length_s / vocoder_upsample_factor)
    if height % pipe.vae_scale_factor != 0:
        height = int(np.ceil(height / pipe.vae_scale_factor)) * pipe.vae_scale_factor
    return height


def load_and_prepare_waveform(audio_path: Path, target_sr: int, target_audio_length_s: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    target_len = int(target_audio_length_s * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    wav = np.clip(wav, -1.0, 1.0)
    return wav, sr


def waveform_to_logmel_exact_audioldm(wav, sr, pipe, duration_sec=10.0):
    stft, target_sr, n_mels, n_fft, hop_length, win_length = get_exact_audioldm_frontend()
    target_height = get_target_mel_height(pipe, duration_sec)

    wav = np.asarray(wav, dtype=np.float32)

    # Mono safety
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Remove NaN/Inf
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample first
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Fix tiny overshoots introduced by decoding/resampling
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1.0:
        # conservative fix: only scale when truly needed
        wav = wav / (peak + 1e-8)

    # hard clip for numerical safety
    wav = np.clip(wav, -1.0, 1.0)

    target_len = int(duration_sec * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    # clip again after pad/crop just to be safe
    wav = np.clip(wav, -1.0, 1.0)

    wav_t = torch.from_numpy(wav).unsqueeze(0).float()

    with torch.no_grad():
        mel_out = stft.mel_spectrogram(wav_t)

    mel = mel_out[0] if isinstance(mel_out, tuple) else mel_out
    mel_np = mel.squeeze(0).cpu().numpy()

    frames = mel_np.shape[1]
    if frames < target_height:
        pad_val = mel_np.min()
        mel_np = np.pad(mel_np, ((0, 0), (0, target_height - frames)), constant_values=pad_val)
    else:
        mel_np = mel_np[:, :target_height]

    mel_for_vae = mel_np.T[None, None, :, :].astype(np.float32)
    return mel_np, mel_for_vae, target_height


# =========================
# Prompt encoding utilities
# =========================

@torch.no_grad()
def encode_prompt_fixed(
    pipe: AudioLDM2Pipeline,
    prompt: str,
    negative_prompt: str = "",
    max_new_tokens: int = 8,
):
    device = pipe.device
    batch_size = 1
    num_waveforms_per_prompt = 1

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    is_vits_text_encoder = isinstance(pipe.text_encoder_2, VitsModel)
    text_encoders = (
        [pipe.text_encoder, pipe.text_encoder_2.text_encoder]
        if is_vits_text_encoder
        else [pipe.text_encoder, pipe.text_encoder_2]
    )

    def unwrap_text_features(x):
        if torch.is_tensor(x):
            return x
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if isinstance(x, (tuple, list)):
            return x[0]
        if hasattr(x, "to_tuple"):
            return x.to_tuple()[0]
        raise TypeError(f"Unsupported text feature output type: {type(x)}")

    def _encode_one(text_a: str, text_b: str):
        prompt_embeds_list = []
        attention_mask_list = []

        for tokenizer, text_encoder, text_in in zip(tokenizers, text_encoders, [text_a, text_b]):
            text_inputs = tokenizer(
                text_in,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attn = text_inputs.attention_mask.to(device)

            if text_encoder.config.model_type == "clap":
                pe = text_encoder.get_text_features(input_ids, attention_mask=attn)
                pe = unwrap_text_features(pe)
                pe = pe[:, None, :]
                attn = attn.new_ones((batch_size, 1))
            elif is_vits_text_encoder:
                pe = text_encoder(
                    input_ids,
                    attention_mask=attn,
                    padding_mask=attn.unsqueeze(-1),
                )[0]
            else:
                pe = text_encoder(input_ids, attention_mask=attn)[0]

            prompt_embeds_list.append(pe)
            attention_mask_list.append(attn)

        proj = pipe.projection_model(
            hidden_states=prompt_embeds_list[0],
            hidden_states_1=prompt_embeds_list[1],
            attention_mask=attention_mask_list[0],
            attention_mask_1=attention_mask_list[1],
        )

        projected = proj.hidden_states
        projected_attn = proj.attention_mask

        generated = pipe.generate_language_model(
            projected,
            attention_mask=projected_attn,
            max_new_tokens=max_new_tokens,
        )

        prompt_embeds = prompt_embeds_list[1].to(dtype=pipe.text_encoder_2.dtype, device=device)
        attention_mask = attention_mask_list[1].to(device=device)
        generated = generated.to(dtype=pipe.language_model.dtype, device=device)

        bs, sl, hs = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1).view(bs * num_waveforms_per_prompt, sl, hs)
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt).view(bs * num_waveforms_per_prompt, sl)

        bs, sl, hs = generated.shape
        generated = generated.repeat(1, num_waveforms_per_prompt, 1).view(bs * num_waveforms_per_prompt, sl, hs)

        return prompt_embeds, attention_mask, generated

    pe_c, am_c, ge_c = _encode_one(prompt, prompt)
    pe_u, am_u, ge_u = _encode_one(negative_prompt, negative_prompt)
    return pe_c, am_c, ge_c, pe_u, am_u, ge_u


# =========================
# Diffusion / inversion utils
# =========================

def build_alpha_cache(pipe: AudioLDM2Pipeline) -> torch.Tensor:
    return pipe.scheduler.alphas_cumprod.to(device=pipe.device, dtype=torch.float32)


def get_alpha_bar_cached(alphas_cumprod: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return alphas_cumprod[int(t.item())]


def eps_x0_from_model_output(
    pipe: AudioLDM2Pipeline,
    alphas_cumprod: torch.Tensor,
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar_t = get_alpha_bar_cached(alphas_cumprod, t).to(dtype=x_t.dtype).view(1, 1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
    pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")

    if pred_type == "epsilon":
        eps = model_output
        x0 = (x_t - sqrt_one_minus_ab * eps) / sqrt_ab
    elif pred_type == "v_prediction":
        v = model_output
        x0 = sqrt_ab * x_t - sqrt_one_minus_ab * v
        eps = sqrt_one_minus_ab * x_t + sqrt_ab * v
    elif pred_type == "sample":
        x0 = model_output
        eps = (x_t - sqrt_ab * x0) / (sqrt_one_minus_ab + 1e-8)
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

    return eps, x0


def ddim_inverse_step_scheduler_aware(
    pipe: AudioLDM2Pipeline,
    alphas_cumprod: torch.Tensor,
    x_t: torch.Tensor,
    model_output: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
) -> torch.Tensor:
    eps, x0 = eps_x0_from_model_output(pipe, alphas_cumprod, model_output, x_t, t)
    alpha_bar_next = get_alpha_bar_cached(alphas_cumprod, t_next).to(dtype=x_t.dtype).view(1, 1, 1, 1)
    x_next = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * eps
    return x_next


def hard_disable_checkpointing(unet):
    try:
        unet.disable_gradient_checkpointing()
    except Exception:
        pass

    if hasattr(unet, "gradient_checkpointing"):
        unet.gradient_checkpointing = False

    for m in unet.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False
        if hasattr(m, "use_gradient_checkpointing"):
            m.use_gradient_checkpointing = False
    return unet


def freeze_module_params(m) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def unet_forward_cpu(pipe, x_in, t_unet_int, ge, pe, am):
    dev = x_in.device
    dtype = x_in.dtype

    pipe.unet.to("cpu", dtype=torch.float32)
    pipe.unet.eval()

    x_cpu = x_in.detach().to("cpu", dtype=torch.float32)
    ge_cpu = ge.detach().to("cpu", dtype=torch.float32)
    pe_cpu = pe.detach().to("cpu", dtype=torch.float32)
    am_cpu = am.detach().to("cpu")
    t_cpu = torch.tensor([t_unet_int], device="cpu", dtype=torch.long)

    out = pipe.unet(
        x_cpu,
        t_cpu,
        encoder_hidden_states=ge_cpu,
        encoder_hidden_states_1=pe_cpu,
        encoder_attention_mask_1=am_cpu,
        return_dict=False,
    )[0]
    return out.to(dev, dtype=dtype)


def unet_forward_cpu_grad(pipe, x_in, t_unet_int, ge, pe, am):
    pipe.unet.to("cpu", dtype=torch.float32)
    pipe.unet.train()

    t_cpu = torch.tensor([t_unet_int], device="cpu", dtype=torch.long)
    out = pipe.unet(
        x_in,
        t_cpu,
        encoder_hidden_states=ge,
        encoder_hidden_states_1=pe,
        encoder_attention_mask_1=am,
        return_dict=False,
    )[0]
    return out


def predict_eps_cfg(
    pipe,
    x,
    t,
    pu,
    pt,
    amu,
    amt,
    gu,
    gt,
    cfg,
    force_cpu_unet=False,
    cpu_grad_mode=False,
):
    t_unet = int(t.item()) if torch.is_tensor(t) else int(t)
    t_sched = t.to(x.device) if torch.is_tensor(t) else torch.tensor(t, device=x.device, dtype=torch.long)

    x_in = torch.cat([x, x], dim=0)
    x_in = pipe.scheduler.scale_model_input(x_in, t_sched)

    pe = torch.cat([pu, pt], dim=0)
    am = torch.cat([amu, amt], dim=0)
    ge = torch.cat([gu, gt], dim=0)

    if force_cpu_unet:
        if cpu_grad_mode:
            eps = unet_forward_cpu_grad(pipe, x_in, t_unet, ge, pe, am)
        else:
            eps = unet_forward_cpu(pipe, x_in, t_unet, ge, pe, am)
    else:
        eps = pipe.unet(
            x_in,
            t_unet,
            encoder_hidden_states=ge,
            encoder_hidden_states_1=pe,
            encoder_attention_mask_1=am,
            return_dict=False,
        )[0]

    eps_u, eps_c = eps.chunk(2, dim=0)
    return eps_u + cfg * (eps_c - eps_u)


def predict_eps_for_inversion(
    pipe,
    x,
    t,
    do_cfg,
    cfg_scale,
    prompt_embeds_uncond,
    prompt_embeds_text,
    attn_uncond,
    attn_text,
    gen_uncond,
    gen_text,
):
    if do_cfg:
        return predict_eps_cfg(
            pipe,
            x,
            t,
            prompt_embeds_uncond,
            prompt_embeds_text,
            attn_uncond,
            attn_text,
            gen_uncond,
            gen_text,
            cfg_scale,
            force_cpu_unet=False,
            cpu_grad_mode=False,
        )

    t_unet = int(t.item()) if torch.is_tensor(t) else int(t)
    t_sched = t.to(x.device) if torch.is_tensor(t) else torch.tensor(t, device=x.device, dtype=torch.long)
    x_in = pipe.scheduler.scale_model_input(x, t_sched)
    model_out = unet_forward_cpu(
        pipe,
        x_in,
        t_unet,
        gen_text,
        prompt_embeds_text,
        attn_text,
    )
    return model_out


# =========================
# Editing regimes
# =========================

def run_plan_for_cfg(
    pipe,
    plan_name,
    cfg_scale,
    z0,
    zT,
    timesteps,
    extra_step_kwargs,
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
    shared_iters=10,
    shared_lr=5e-3,
    perstep_iters=3,
    perstep_lr=1e-2,
    planb_optimize="gu",
    cpu_unet_opt=True,
    cpu_unet_edit=False,
):
    print("========================================")
    print(f"PLAN {plan_name} | CFG {cfg_scale}")
    print("========================================")

    pipe.unet.to(pipe.device, dtype=torch.float32)
    pipe.unet.eval()
    pu_shared = None
    gu_shared = None
    pu_steps = None
    gu_steps = None

    pipe.unet = hard_disable_checkpointing(pipe.unet)
    freeze_module_params(pipe.unet)
    pipe.unet.eval()

    if plan_name == "B":
        print(
            f"[PLAN B] Shared null opt iters={shared_iters} "
            f"lr={shared_lr} optimize={planb_optimize} "
            f"cpu_unet_opt={cpu_unet_opt}"
        )

        if cpu_unet_opt:
            pipe.unet.to("cpu", dtype=torch.float32)
            pipe.unet.train()

            zT_cpu = zT.detach().to("cpu", dtype=torch.float32)
            z0_cpu = z0.detach().to("cpu", dtype=torch.float32)

            pt_cpu = prompt_embeds_text.detach().to("cpu", dtype=torch.float32)
            am_text_cpu = attn_text.detach().to("cpu")
            gt_cpu = gen_text.detach().to("cpu", dtype=torch.float32)

            if planb_optimize in ("both", "pu"):
                pu_shared = prompt_embeds_uncond.clone().detach().to("cpu", dtype=torch.float32).requires_grad_(True)
            else:
                pu_shared = prompt_embeds_uncond.detach().to("cpu", dtype=torch.float32)

            if planb_optimize in ("both", "gu"):
                gu_shared = gen_uncond.clone().detach().to("cpu", dtype=torch.float32).requires_grad_(True)
            else:
                gu_shared = gen_uncond.detach().to("cpu", dtype=torch.float32)

            am_uncond_cpu = attn_uncond.detach().to("cpu")

            params = []
            if isinstance(pu_shared, torch.Tensor) and pu_shared.requires_grad:
                params.append(pu_shared)
            if isinstance(gu_shared, torch.Tensor) and gu_shared.requires_grad:
                params.append(gu_shared)

            if not params:
                raise ValueError("Plan B selected but nothing is set to optimize. Use --planb_optimize gu|pu|both.")

            opt = torch.optim.Adam(params, lr=shared_lr)

            for it in tqdm(range(shared_iters), desc=f"Plan B opt | CFG={cfg_scale}"):
                opt.zero_grad(set_to_none=True)

                x_cpu = zT_cpu
                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    eps = predict_eps_cfg(
                        pipe, x_cpu, t,
                        pu_shared, pt_cpu,
                        am_uncond_cpu, am_text_cpu,
                        gu_shared, gt_cpu,
                        cfg_scale,
                        force_cpu_unet=True,
                        cpu_grad_mode=True,
                    )
                    x_cpu = pipe.scheduler.step(eps, t.to("cpu"), x_cpu, **extra_step_kwargs).prev_sample

                loss = torch.nn.functional.mse_loss(x_cpu, z0_cpu)
                loss.backward()
                opt.step()
                print(f"[Plan B][CFG={cfg_scale}] iter {it+1}/{shared_iters} loss={loss.item():.6f}")

            pu_shared = pu_shared.detach().to(pipe.device, dtype=torch.float32)
            gu_shared = gu_shared.detach().to(pipe.device, dtype=torch.float32)
            pipe.unet.to(pipe.device, dtype=torch.float32)

        else:
            if planb_optimize in ("both", "pu"):
                pu_shared = prompt_embeds_uncond.clone().detach().requires_grad_(True)
            else:
                pu_shared = prompt_embeds_uncond.detach()

            if planb_optimize in ("both", "gu"):
                gu_shared = gen_uncond.clone().detach().requires_grad_(True)
            else:
                gu_shared = gen_uncond.detach()

            params = []
            if isinstance(pu_shared, torch.Tensor) and pu_shared.requires_grad:
                params.append(pu_shared)
            if isinstance(gu_shared, torch.Tensor) and gu_shared.requires_grad:
                params.append(gu_shared)

            if not params:
                raise ValueError("Plan B selected but nothing is set to optimize. Use --planb_optimize gu|pu|both.")

            opt = torch.optim.Adam(params, lr=shared_lr)

            for it in tqdm(range(shared_iters), desc=f"Plan B opt (GPU) | CFG={cfg_scale}"):
                opt.zero_grad(set_to_none=True)
                x = zT.clone()
                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    eps = predict_eps_cfg(
                        pipe, x, t,
                        pu_shared, prompt_embeds_text,
                        attn_uncond, attn_text,
                        gu_shared, gen_text,
                        cfg_scale,
                        force_cpu_unet=False,
                        cpu_grad_mode=False,
                    )
                    x = pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample

                loss = torch.nn.functional.mse_loss(x, z0)
                loss.backward()
                opt.step()
                print(f"[Plan B][CFG={cfg_scale}] iter {it+1}/{shared_iters} loss={loss.item():.6f}")

            pu_shared = pu_shared.detach()
            gu_shared = gu_shared.detach()

    if plan_name == "C":
        print(
            f"[PLAN C] Per-step opt inner={perstep_iters} "
            f"lr={perstep_lr} optimize={planb_optimize} "
            f"cpu_unet_opt={cpu_unet_opt}"
        )

        print("[PLAN C] Building forward trajectory (CFG=1) for per-step targets...")
        traj_fwd = [zT.clone()]
        with torch.no_grad():
            x = zT.clone()
            for i in tqdm(range(len(timesteps) - 1), desc="Build traj_fwd"):
                t = timesteps[i]
                eps = predict_eps_for_inversion(
                    pipe, x, t,
                    do_cfg=False,
                    cfg_scale=1.0,
                    prompt_embeds_uncond=prompt_embeds_uncond,
                    prompt_embeds_text=prompt_embeds_text,
                    attn_uncond=attn_uncond,
                    attn_text=attn_text,
                    gen_uncond=gen_uncond,
                    gen_text=gen_text,
                )
                x = pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample
                traj_fwd.append(x.clone())

        pu_steps, gu_steps = [], []

        if cpu_unet_opt:
            pipe.unet.to("cpu", dtype=torch.float32)
            pipe.unet.train()

            prompt_text_cpu = prompt_embeds_text.detach().to("cpu", dtype=torch.float32)
            attn_text_cpu = attn_text.detach().to("cpu")
            gen_text_cpu = gen_text.detach().to("cpu", dtype=torch.float32)
            attn_uncond_cpu = attn_uncond.detach().to("cpu")

            for i in tqdm(range(len(timesteps) - 1), desc=f"Plan C opt | CFG={cfg_scale}"):
                t = timesteps[i]
                x_t = traj_fwd[i].detach().to("cpu", dtype=torch.float32)
                x_target = traj_fwd[i + 1].detach().to("cpu", dtype=torch.float32)

                if planb_optimize in ("both", "pu"):
                    pu = prompt_embeds_uncond.clone().detach().to("cpu", dtype=torch.float32).requires_grad_(True)
                else:
                    pu = prompt_embeds_uncond.detach().to("cpu", dtype=torch.float32)

                if planb_optimize in ("both", "gu"):
                    gu = gen_uncond.clone().detach().to("cpu", dtype=torch.float32).requires_grad_(True)
                else:
                    gu = gen_uncond.detach().to("cpu", dtype=torch.float32)

                params = []
                if isinstance(pu, torch.Tensor) and pu.requires_grad:
                    params.append(pu)
                if isinstance(gu, torch.Tensor) and gu.requires_grad:
                    params.append(gu)

                if not params:
                    raise ValueError("Plan C selected but nothing is set to optimize. Use --planb_optimize gu|pu|both.")

                opt = torch.optim.Adam(params, lr=perstep_lr)

                last_loss = None
                for _ in range(perstep_iters):
                    opt.zero_grad(set_to_none=True)

                    eps = predict_eps_cfg(
                        pipe, x_t, t,
                        pu, prompt_text_cpu,
                        attn_uncond_cpu, attn_text_cpu,
                        gu, gen_text_cpu,
                        cfg_scale,
                        force_cpu_unet=True,
                        cpu_grad_mode=True,
                    )

                    x_prev = pipe.scheduler.step(eps, t.to("cpu"), x_t, **extra_step_kwargs).prev_sample
                    loss = torch.nn.functional.mse_loss(x_prev, x_target)
                    loss.backward()
                    opt.step()
                    last_loss = loss.item()

                print(f"[Plan C][CFG={cfg_scale}] step {i+1}/{len(timesteps)-1} final_inner_loss={last_loss:.6f}")
                pu_steps.append(pu.detach().to(pipe.device, dtype=torch.float32))
                gu_steps.append(gu.detach().to(pipe.device, dtype=torch.float32))

            pipe.unet.to(pipe.device, dtype=torch.float32)

        else:
            for i in tqdm(range(len(timesteps) - 1), desc=f"Plan C opt (GPU) | CFG={cfg_scale}"):
                t = timesteps[i]
                x_t = traj_fwd[i]
                x_target = traj_fwd[i + 1]

                if planb_optimize in ("both", "pu"):
                    pu = prompt_embeds_uncond.clone().detach().requires_grad_(True)
                else:
                    pu = prompt_embeds_uncond.detach()

                if planb_optimize in ("both", "gu"):
                    gu = gen_uncond.clone().detach().requires_grad_(True)
                else:
                    gu = gen_uncond.detach()

                params = []
                if isinstance(pu, torch.Tensor) and pu.requires_grad:
                    params.append(pu)
                if isinstance(gu, torch.Tensor) and gu.requires_grad:
                    params.append(gu)

                if not params:
                    raise ValueError("Plan C selected but nothing is set to optimize. Use --planb_optimize gu|pu|both.")

                opt = torch.optim.Adam(params, lr=perstep_lr)

                last_loss = None
                for _ in range(perstep_iters):
                    opt.zero_grad(set_to_none=True)

                    eps = predict_eps_cfg(
                        pipe, x_t, t,
                        pu, prompt_embeds_text,
                        attn_uncond, attn_text,
                        gu, gen_text,
                        cfg_scale,
                        force_cpu_unet=False,
                        cpu_grad_mode=False,
                    )
                    x_prev = pipe.scheduler.step(eps, t, x_t, **extra_step_kwargs).prev_sample
                    loss = torch.nn.functional.mse_loss(x_prev, x_target)
                    loss.backward()
                    opt.step()
                    last_loss = loss.item()

                print(f"[Plan C][CFG={cfg_scale}] step {i+1}/{len(timesteps)-1} final_inner_loss={last_loss:.6f}")
                pu_steps.append(pu.detach())
                gu_steps.append(gu.detach())

    if not cpu_unet_edit:
        pipe.unet.to(pipe.device, dtype=torch.float32)
        pipe.unet.eval()

    print(f"Forward EDIT: zT -> z0_edit | PLAN={plan_name} | CFG={cfg_scale}")

    with torch.no_grad():
        x = zT.clone()
        for i in tqdm(range(len(timesteps) - 1), desc=f"Forward edit | {plan_name} | CFG={cfg_scale}"):
            t = timesteps[i]

            if plan_name == "A":
                pu_i, amu_i, gu_i = pu_edit, amu_edit, gu_edit
            elif plan_name == "B":
                pu_i, amu_i, gu_i = pu_shared, amu_edit, gu_shared
            else:
                pu_i, amu_i, gu_i = pu_steps[i], amu_edit, gu_steps[i]

            eps = predict_eps_cfg(
                pipe, x, t,
                pu_i, pt_edit,
                amu_i, am_edit,
                gu_i, gt_edit,
                cfg_scale,
                force_cpu_unet=cpu_unet_edit,
                cpu_grad_mode=False,
            )
            x = pipe.scheduler.step(eps, t, x, **extra_step_kwargs).prev_sample

    return x


# =========================
# Core processing
# =========================

def load_pipeline(model_path: str | Path, device: str) -> AudioLDM2Pipeline:
    pipe = AudioLDM2Pipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)

    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass

    pipe.unet.set_attn_processor(AttnProcessor())
    pipe.unet.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)
    pipe.text_encoder_2.to(dtype=torch.float32)
    pipe.projection_model.to(dtype=torch.float32)
    pipe.language_model.to(dtype=torch.float32)

    return pipe


def encode_audio_to_z0(pipe: AudioLDM2Pipeline, wav: np.ndarray, sr: int, target_audio_length_s: float) -> torch.Tensor:
    _, mel_for_vae, _ = waveform_to_logmel_exact_audioldm(
        wav,
        sr,
        pipe,
        duration_sec=target_audio_length_s,
    )
    mel_tensor = torch.from_numpy(mel_for_vae).to(device=pipe.device, dtype=torch.float32)

    with torch.no_grad():
        posterior = pipe.vae.encode(mel_tensor).latent_dist
        z0 = posterior.sample()
        z0 = z0 * pipe.vae.config.scaling_factor
    return z0


def invert_latent(
    pipe: AudioLDM2Pipeline,
    z0: torch.Tensor,
    steps: int,
    invert_prompt: str,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    inv_timesteps = timesteps.flip(0)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    alphas_cumprod = build_alpha_cache(pipe)

    pt_inv, at_inv, gt_inv, pu_inv, au_inv, gu_inv = encode_prompt_fixed(
        pipe,
        invert_prompt,
        negative_prompt="",
        max_new_tokens=max_new_tokens,
    )

    cfg_invert = 1.0
    do_cfg_invert = cfg_invert > 1.0

    x = z0.clone()
    for i in tqdm(range(len(inv_timesteps) - 1), desc="DDIM inversion", leave=False):
        t = inv_timesteps[i]
        t_next = inv_timesteps[i + 1]

        model_out = predict_eps_for_inversion(
            pipe,
            x,
            t,
            do_cfg=do_cfg_invert,
            cfg_scale=cfg_invert,
            prompt_embeds_uncond=pu_inv,
            prompt_embeds_text=pt_inv,
            attn_uncond=au_inv,
            attn_text=at_inv,
            gen_uncond=gu_inv,
            gen_text=gt_inv,
        )

        x = ddim_inverse_step_scheduler_aware(
            pipe,
            alphas_cumprod,
            x,
            model_out,
            t,
            t_next,
        )

    zT = x
    inv_ctx = {
        "timesteps": timesteps,
        "extra_step_kwargs": extra_step_kwargs,
        "prompt_embeds_text": pt_inv,
        "prompt_embeds_uncond": pu_inv,
        "attn_text": at_inv,
        "attn_uncond": au_inv,
        "gen_text": gt_inv,
        "gen_uncond": gu_inv,
    }
    return zT, timesteps, inv_ctx


def edit_one_file(pipe: AudioLDM2Pipeline, audio_path: Path, output_path: Path, args) -> None:
    target_sr = int(pipe.vocoder.config.sampling_rate)
    wav, sr = load_and_prepare_waveform(audio_path, target_sr, args.target_audio_length_s)

    print(f"\n[FILE] {audio_path}")
    z0 = encode_audio_to_z0(pipe, wav, sr, args.target_audio_length_s)
    zT, timesteps, inv_ctx = invert_latent(
        pipe,
        z0,
        steps=args.steps,
        invert_prompt=args.invert_prompt,
        max_new_tokens=args.max_new_tokens,
    )

    pt_edit, am_edit, gt_edit, pu_edit, amu_edit, gu_edit = encode_prompt_fixed(
        pipe,
        args.edit_prompt,
        negative_prompt="",
        max_new_tokens=args.max_new_tokens,
    )

    z0_edit = run_plan_for_cfg(
        pipe=pipe,
        plan_name=args.regime,
        cfg_scale=args.cfg_scale,
        z0=z0,
        zT=zT,
        timesteps=timesteps,
        extra_step_kwargs=inv_ctx["extra_step_kwargs"],
        prompt_embeds_text=inv_ctx["prompt_embeds_text"],
        prompt_embeds_uncond=inv_ctx["prompt_embeds_uncond"],
        attn_text=inv_ctx["attn_text"],
        attn_uncond=inv_ctx["attn_uncond"],
        gen_text=inv_ctx["gen_text"],
        gen_uncond=inv_ctx["gen_uncond"],
        pt_edit=pt_edit,
        am_edit=am_edit,
        gt_edit=gt_edit,
        pu_edit=pu_edit,
        amu_edit=amu_edit,
        gu_edit=gu_edit,
        shared_iters=args.shared_iters,
        shared_lr=args.shared_lr,
        perstep_iters=args.perstep_iters,
        perstep_lr=args.perstep_lr,
        planb_optimize=args.planb_optimize,
        cpu_unet_opt=args.cpu_unet_opt,
        cpu_unet_edit=args.cpu_unet_edit,
    )

    wav_edit = decode_latents_to_audio(pipe, z0_edit, wav_len=len(wav))
    save_wav(output_path, wav_edit, sr)
    print(f"[SAVED] {output_path}")


def find_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern_iter = input_dir.rglob("*.wav") if recursive else input_dir.glob("*.wav")
    return sorted([p for p in pattern_iter if p.is_file()])


def make_output_path(input_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    rel = input_path.relative_to(input_root)
    out_name = f"{rel.stem}{suffix}.wav"
    return output_root / rel.parent / out_name


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch latent inversion editing with AudioLDM2 over a folder of WAV files."
    )
    parser.add_argument("--input_folder", required=True, type=Path)
    parser.add_argument("--output_folder", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--edit_prompt", required=True, type=str)

    parser.add_argument("--regime", default="A", choices=["A", "B", "C"])
    parser.add_argument("--invert_prompt", default="", type=str)
    parser.add_argument("--cfg_scale", default=3.5, type=float)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--max_new_tokens", default=8, type=int)
    parser.add_argument("--target_audio_length_s", default=10.0, type=float)

    parser.add_argument("--shared_iters", default=10, type=int)
    parser.add_argument("--shared_lr", default=5e-3, type=float)
    parser.add_argument("--perstep_iters", default=3, type=int)
    parser.add_argument("--perstep_lr", default=1e-2, type=float)
    parser.add_argument("--planb_optimize", default="gu", choices=["both", "gu", "pu"])

    parser.add_argument("--cpu_unet_opt", action="store_true", default=False)
    parser.add_argument("--cpu_unet_edit", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--suffix", default="_edited", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_errors", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] regime={args.regime} | cfg={args.cfg_scale} | steps={args.steps}")

    audio_files = find_audio_files(args.input_folder, recursive=args.recursive)
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {args.input_folder}")

    print(f"[INFO] found {len(audio_files)} wav files")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model_path, device=device)

    num_done = 0
    num_skipped = 0
    num_failed = 0

    for audio_path in audio_files:
        output_path = make_output_path(audio_path, args.input_folder, args.output_folder, args.suffix)

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] already exists: {output_path}")
            num_skipped += 1
            continue

        try:
            edit_one_file(pipe, audio_path, output_path, args)
            num_done += 1
        except Exception as e:
            num_failed += 1
            print(f"[ERROR] {audio_path}: {type(e).__name__}: {e}")
            if not args.skip_errors:
                raise

    print("\n[SUMMARY]")
    print(f"done={num_done} skipped={num_skipped} failed={num_failed}")


if __name__ == "__main__":
    main()
