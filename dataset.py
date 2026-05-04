from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional

import soundfile as sf


from .utils import (
    apply_offset,
    load_mono_audio,
    peak_normalize,
    save_wav,
    split_into_segments,
    trim_leading_silence,
)


@dataclass
class SegmentConfig:
    target_sr: int = 16000
    segment_duration: float = 10.0
    trim_silence: bool = True
    silence_top_db: float = 40.0
    silence_pre_margin: float = 0.2
    offset_piano: float = 0.0
    offset_guitar: float = 0.0


def build_paired_dataset(
    piano_path,
    guitar_path,
    output_root="dataset",
    song_name="song",
    config: Optional[SegmentConfig] = None,
):
    config = SegmentConfig() if config is None else config

    piano_wav, sr = load_mono_audio(piano_path, target_sr=config.target_sr)
    guitar_wav, _ = load_mono_audio(guitar_path, target_sr=config.target_sr)

    if config.trim_silence:
        piano_wav = trim_leading_silence(
            piano_wav, sr, top_db=config.silence_top_db, pre_margin=config.silence_pre_margin
        )
        guitar_wav = trim_leading_silence(
            guitar_wav, sr, top_db=config.silence_top_db, pre_margin=config.silence_pre_margin
        )

    piano_wav = apply_offset(piano_wav, config.offset_piano, sr)
    guitar_wav = apply_offset(guitar_wav, config.offset_guitar, sr)

    min_len = min(len(piano_wav), len(guitar_wav))
    piano_wav = piano_wav[:min_len]
    guitar_wav = guitar_wav[:min_len]

    piano_segs = split_into_segments(piano_wav, sr, config.segment_duration)
    guitar_segs = split_into_segments(guitar_wav, sr, config.segment_duration)
    n = min(len(piano_segs), len(guitar_segs))

    base = Path(output_root) / song_name
    piano_dir = base / "piano"
    guitar_dir = base / "guitar"
    piano_dir.mkdir(parents=True, exist_ok=True)
    guitar_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        save_wav(piano_dir / f"seg_{i:04d}.wav", piano_segs[i], sr)
        save_wav(guitar_dir / f"seg_{i:04d}.wav", guitar_segs[i], sr)

    return {
        "song_name": song_name,
        "n_segments": n,
        "output_dir": str(base),
        "piano_dir": str(piano_dir),
        "guitar_dir": str(guitar_dir),
    }


def list_paired_segment_paths(dataset_root, song_name):
    base = Path(dataset_root) / song_name
    piano_dir = base / "piano"
    guitar_dir = base / "guitar"

    piano_files = sorted(piano_dir.glob("seg_*.wav"))
    guitar_files = sorted(guitar_dir.glob("seg_*.wav"))

    piano_map = {p.name: p for p in piano_files}
    guitar_map = {p.name: p for p in guitar_files}
    common_names = sorted(set(piano_map.keys()) & set(guitar_map.keys()))
    return [(piano_map[name], guitar_map[name]) for name in common_names]


def audition_paired_segments(dataset_root, song_name, indices=None, n_random: int = 3, seed: int = 0):
    pairs = list_paired_segment_paths(dataset_root, song_name)
    if len(pairs) == 0:
        print("No paired segments found.")
        return

    if indices is None:
        rng = random.Random(seed)
        all_idx = list(range(len(pairs)))
        rng.shuffle(all_idx)
        indices = sorted(all_idx[:min(n_random, len(pairs))])

    print(f"Found {len(pairs)} paired segments for song '{song_name}'.")
    print("Testing indices:", indices)

    for idx in indices:
        piano_path, guitar_path = pairs[idx]
        piano_wav, piano_sr = sf.read(str(piano_path), always_2d=False)
        guitar_wav, guitar_sr = sf.read(str(guitar_path), always_2d=False)
        if getattr(piano_wav, "ndim", 1) == 2:
            piano_wav = piano_wav.mean(axis=1)
        if getattr(guitar_wav, "ndim", 1) == 2:
            guitar_wav = guitar_wav.mean(axis=1)

        print("\n" + "=" * 70)
        print(f"Segment index: {idx}")
        print("Piano :", piano_path.name)
        print("Guitar:", guitar_path.name)
        print("\nPiano")
        display(Audio(peak_normalize(piano_wav), rate=piano_sr))
        print("Guitar")
        display(Audio(peak_normalize(guitar_wav), rate=guitar_sr))


def build_song_pairs(dataset_root: Path, song_name: str):
    song_dir = dataset_root / song_name
    piano_dir = song_dir / "piano"
    guitar_dir = song_dir / "guitar"

    piano_files = sorted(piano_dir.glob("seg_*.wav"))
    guitar_files = sorted(guitar_dir.glob("seg_*.wav"))

    piano_map = {p.name: p for p in piano_files}
    guitar_map = {p.name: p for p in guitar_files}

    common = sorted(set(piano_map.keys()) & set(guitar_map.keys()))
    pairs = []
    for name in common:
        pairs.append({
            "song": song_name,
            "segment": name,
            "piano": str(piano_map[name]),
            "guitar": str(guitar_map[name]),
        })
    return pairs


def build_dataset_pairs(dataset_root: Path, song_names, max_pairs=None):
    pairs = []
    for song in song_names:
        pairs.extend(build_song_pairs(dataset_root, song))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def make_bidirectional_samples(pairs):
    samples = []
    for item in pairs:
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["guitar"],
            "target_path": item["piano"],
            "direction": +1,
            "source_domain": "guitar",
            "target_domain": "piano",
        })
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["piano"],
            "target_path": item["guitar"],
            "direction": -1,
            "source_domain": "piano",
            "target_domain": "guitar",
        })
    return samples


def build_dataset_unpaired(dataset_root: Path, song_names, max_items_per_domain=None):
    piano_items = []
    guitar_items = []

    for song in song_names:
        song_dir = dataset_root / song
        piano_dir = song_dir / "piano"
        guitar_dir = song_dir / "guitar"

        piano_files = sorted(piano_dir.glob("seg_*.wav"))
        guitar_files = sorted(guitar_dir.glob("seg_*.wav"))

        for p in piano_files:
            piano_items.append({
                "song": song,
                "segment": p.name,
                "domain": "piano",
                "path": str(p),
            })

        for p in guitar_files:
            guitar_items.append({
                "song": song,
                "segment": p.name,
                "domain": "guitar",
                "path": str(p),
            })

    if max_items_per_domain is not None:
        piano_items = piano_items[:max_items_per_domain]
        guitar_items = guitar_items[:max_items_per_domain]

    return {
        "piano": piano_items,
        "guitar": guitar_items,
    }


def make_unpaired_directional_samples(unpaired_items: Dict[str, List[dict]]):
    samples = []

    for item in unpaired_items.get("guitar", []):
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["path"],
            "target_path": None,
            "direction": +1,
            "source_domain": "guitar",
            "target_domain": "piano",
            "paired": False,
        })

    for item in unpaired_items.get("piano", []):
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["path"],
            "target_path": None,
            "direction": -1,
            "source_domain": "piano",
            "target_domain": "guitar",
            "paired": False,
        })

    return samples


# ============================================================================
# NEW: surface dataset helpers
# ============================================================================

def _split_paths_train_val(paths, val_fraction: float = 0.1, seed: int = 0):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    if len(paths) == 0:
        return [], []

    n_val = int(round(len(paths) * val_fraction))
    if val_fraction > 0.0 and len(paths) > 1:
        n_val = max(1, n_val)

    n_val = min(n_val, len(paths))
    val_paths = sorted(paths[:n_val])
    train_paths = sorted(paths[n_val:])
    return train_paths, val_paths


def build_surface_dataset_unpaired(
    dataset_root: Path,
    split: str = "train",
    negative_domain: str = "leaves",
    positive_domain: str = "wooden_stairs_up",
    max_items_per_domain=None,
    val_fraction: float = 0.1,
    split_seed: int = 0,
):
    """
    Build an unpaired 2-domain dataset directly from:

        dataset_root/
          leaves/
            *.wav
          wooden_stairs_up/
            *.wav

    split must be "train" or "val".
    """

    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    neg_dir = Path(dataset_root) / negative_domain
    pos_dir = Path(dataset_root) / positive_domain

    neg_files = sorted(neg_dir.glob("*.wav"))
    pos_files = sorted(pos_dir.glob("*.wav"))

    neg_train, neg_val = _split_paths_train_val(neg_files, val_fraction=val_fraction, seed=split_seed)
    pos_train, pos_val = _split_paths_train_val(pos_files, val_fraction=val_fraction, seed=split_seed)

    neg_selected = neg_train if split == "train" else neg_val
    pos_selected = pos_train if split == "train" else pos_val

    if max_items_per_domain is not None:
        neg_selected = neg_selected[:max_items_per_domain]
        pos_selected = pos_selected[:max_items_per_domain]

    neg_items = []
    for p in neg_selected:
        neg_items.append({
            "song": "surface_dataset",
            "segment": p.name,
            "domain": negative_domain,
            "path": str(p),
        })

    pos_items = []
    for p in pos_selected:
        pos_items.append({
            "song": "surface_dataset",
            "segment": p.name,
            "domain": positive_domain,
            "path": str(p),
        })

    return {
        negative_domain: neg_items,
        positive_domain: pos_items,
    }


def make_surface_unpaired_directional_samples(
    unpaired_items: Dict[str, List[dict]],
    negative_domain: str = "leaves",
    positive_domain: str = "wooden_stairs_up",
):
    """
    Positive direction (+1): negative -> positive
        leaves -> wooden_stairs_up

    Negative direction (-1): positive -> negative
        wooden_stairs_up -> leaves
    """
    samples = []

    for item in unpaired_items.get(negative_domain, []):
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["path"],
            "target_path": None,
            "direction": +1,
            "source_domain": negative_domain,
            "target_domain": positive_domain,
            "paired": False,
        })

    for item in unpaired_items.get(positive_domain, []):
        samples.append({
            "song": item["song"],
            "segment": item["segment"],
            "source_path": item["path"],
            "target_path": None,
            "direction": -1,
            "source_domain": positive_domain,
            "target_domain": negative_domain,
            "paired": False,
        })

    return samples


# ============================================================================
# NEW: emotion paired dataset (angry vs calm)
# ============================================================================

def build_emotion_dataset_paired(
    dataset_root: Path,
    split: str = "train",
    negative_domain: str = "calm",
    positive_domain: str = "angry",
    max_pairs=None,
    val_fraction: float = 0.1,
    split_seed: int = 0,
):
    """
    Build a paired dataset from:

        dataset_root/
          calm/
            name1.wav
          angry/
            name1.wav

    Returns same structure as build_dataset_pairs().
    """

    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    neg_dir = dataset_root / negative_domain
    pos_dir = dataset_root / positive_domain

    neg_files = sorted(neg_dir.glob("*.wav"))
    pos_files = sorted(pos_dir.glob("*.wav"))

    neg_map = {p.name: p for p in neg_files}
    pos_map = {p.name: p for p in pos_files}

    common_names = sorted(set(neg_map.keys()) & set(pos_map.keys()))

    if len(common_names) == 0:
        raise RuntimeError("No matching filenames between domains.")

    # deterministic split
    rng = random.Random(split_seed)
    rng.shuffle(common_names)

    n_val = int(round(len(common_names) * val_fraction))
    if val_fraction > 0.0 and len(common_names) > 1:
        n_val = max(1, n_val)

    val_names = sorted(common_names[:n_val])
    train_names = sorted(common_names[n_val:])

    selected = train_names if split == "train" else val_names

    if max_pairs is not None:
        selected = selected[:max_pairs]

    pairs = []
    for name in selected:
        pairs.append({
            "song": "emotion_dataset",
            "segment": name,
            "piano": str(pos_map[name]),   # positive domain
            "guitar": str(neg_map[name]),  # negative domain
        })

    return pairs