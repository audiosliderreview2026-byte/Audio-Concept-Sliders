from __future__ import annotations

import numpy as np
import torch
import librosa


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


def get_target_mel_height(pipe, audio_length_s: float):
    upsample_rates = pipe.vocoder.config.upsample_rates
    vocoder_upsample_factor = np.prod(upsample_rates) / pipe.vocoder.config.sampling_rate
    height = int(audio_length_s / vocoder_upsample_factor)
    if height % pipe.vae_scale_factor != 0:
        height = int(np.ceil(height / pipe.vae_scale_factor)) * pipe.vae_scale_factor
    return height


def waveform_to_logmel_exact_audioldm(wav, sr: int, pipe, duration_sec: float = 10.0):
    stft, target_sr, n_mels, _, _, _ = get_exact_audioldm_frontend()
    target_height = get_target_mel_height(pipe, duration_sec)

    wav = np.asarray(wav, dtype=np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    target_len = int(duration_sec * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

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
