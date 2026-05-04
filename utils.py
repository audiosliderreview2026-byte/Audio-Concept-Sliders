from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch


def peak_normalize(x, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    return x if m < eps else x / m


def sanitize_audio(x, name: str = "audio"):
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


def save_wav(path, wav, sr: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = sanitize_audio(wav, name=str(path))
    wav = peak_normalize(wav)
    sf.write(str(path), wav, sr)


def load_mono_audio(path, target_sr: Optional[int] = None, duration_sec: Optional[float] = None):
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if duration_sec is not None:
        target_len = int(duration_sec * sr)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

    return wav, sr


def decode_latents_to_audio(pipe, latents: torch.Tensor, wav_len: Optional[int] = None):
    with torch.no_grad():
        mel = pipe.vae.decode((1.0 / pipe.vae.config.scaling_factor) * latents).sample
        wav_out = pipe.mel_spectrogram_to_waveform(mel).cpu().numpy()[0]
    if wav_len is not None:
        wav_out = wav_out[:wav_len]
    return wav_out


def trim_leading_silence(wav, sr: int, top_db: float = 40, pre_margin: float = 0.2):
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    start = intervals[0][0]
    margin = int(pre_margin * sr)
    start = max(0, start - margin)
    return wav[start:]


def apply_offset(wav, offset_seconds: float, sr: int):
    offset_samples = int(offset_seconds * sr)
    if offset_samples > 0:
        wav = wav[offset_samples:]
    elif offset_samples < 0:
        wav = np.pad(wav, (abs(offset_samples), 0))
    return wav


def split_into_segments(wav, sr: int, segment_duration: float = 10.0):
    seg_len = int(segment_duration * sr)
    n_segments = len(wav) // seg_len
    return [wav[i * seg_len:(i + 1) * seg_len] for i in range(n_segments)]
