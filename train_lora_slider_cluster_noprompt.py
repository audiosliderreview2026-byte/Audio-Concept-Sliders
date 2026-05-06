import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from audioldm2_slider import (
    EditorConfig,
    AudioLDM2Editor,
    LoRAConfig,
    TrainerConfig,
    LoRASliderTrainer,
    build_dataset_pairs,
    make_bidirectional_samples,
)
from audioldm2_slider.utils import peak_normalize

from audioldm2_slider.dataset import (
    build_dataset_unpaired,
    make_unpaired_directional_samples,
    build_surface_dataset_unpaired,
    make_surface_unpaired_directional_samples,
    build_emotion_dataset_paired,   # NEW
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AudioLDM2 LoRA slider.")

    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_root", required=True)

    # instrument mode still uses these
    p.add_argument("--train_songs", nargs="*", default=[])
    p.add_argument("--val_songs", nargs="*", default=[])

    # NEW: training regime
    p.add_argument(
        "--training_regime",
        choices=["prompt", "prompt_free_paired"],
        default="prompt",
        help="Use the prompt-based contrastive loss or a prompt-free paired contrastive loss.",
    )

    p.add_argument("--prompt_target", default="")
    p.add_argument("--prompt_neutral", default="")
    p.add_argument("--prompt_piano", default="")
    p.add_argument("--prompt_guitar", default="")

    p.add_argument("--train_steps", type=int, default=50)
    p.add_argument("--train_cfg_scale", type=float, default=3.5)
    p.add_argument("--beta_identity", "--identity_loss_weight", dest="beta_identity", type=float, default=0.0)
    p.add_argument("--pair_loss_weight", type=float, default=1.0)
    p.add_argument("--separation_loss_weight", type=float, default=1.0)
    p.add_argument("--separation_margin", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--max_train_pairs", type=int, default=8)
    p.add_argument("--max_val_pairs", type=int, default=4)
    p.add_argument("--out_dir", default="lora_slider_training")

    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--target_audio_length_s", type=float, default=10.0)
    p.add_argument("--invert_prompt", default="")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--store_wav", action="store_true")

    p.add_argument("--save_inference_examples", type=int, default=2)
    p.add_argument("--strengths", nargs="+", type=float, default=[-2.0, -1.0, 0.0, 1.0, 2.0])

    p.add_argument("--resume_from", type=str, default=None, help="Path to a LoRA checkpoint (.pt) to resume from")

    p.add_argument(
        "--data_regime",
        choices=["paired", "unpaired"],
        default="paired",
        help="Use aligned paired samples or ignore pairing and train from unpaired domain sets.",
    )

    # switch between the original instrument dataset and the new surface dataset
    p.add_argument(
        "--dataset_type",
        choices=["instrument", "surface", "emotion"],
        default="instrument",
        help="Use the original piano/guitar dataset structure or the surface dataset structure.",
    )

    # only used when dataset_type=surface
    p.add_argument("--surface_negative_domain", default="leaves")
    p.add_argument("--surface_positive_domain", default="wooden_stairs_up")
    p.add_argument("--surface_val_fraction", type=float, default=0.1)
    p.add_argument("--surface_split_seed", type=int, default=0)

    # only used when dataset_type=emotion
    p.add_argument("--emotion_negative_domain", default="calm")
    p.add_argument("--emotion_positive_domain", default="angry")
    p.add_argument("--emotion_val_fraction", type=float, default=0.1)
    p.add_argument("--emotion_split_seed", type=int, default=0)

    # Null-text optimization
    p.add_argument("--use_nti", action="store_true", help="Use per-sample null-text optimization during LoRA training.")
    p.add_argument("--nti_cache_dir", default=None)
    p.add_argument("--nti_inner_steps", type=int, default=5)
    p.add_argument("--nti_lr", type=float, default=1e-2)
    p.add_argument("--nti_prompt", default="")
    p.add_argument("--nti_optimize", choices=["pu", "gu", "both"], default="gu")
    p.add_argument("--nti_overwrite", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading editor...")
    editor_cfg = EditorConfig(
        model_path=args.model_path,
        steps=args.train_steps,
        max_new_tokens=args.max_new_tokens,
        target_audio_length_s=args.target_audio_length_s,
    )
    editor = AudioLDM2Editor(editor_cfg)

    print("Injecting LoRA...")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    editor.inject_lora(lora_cfg, verbose=True)

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading LoRA checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        missing, unexpected = editor.pipe.unet.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    print(f"Building dataset entries... regime={args.data_regime}, dataset_type={args.dataset_type}, training_regime={args.training_regime}")

    if args.training_regime == "prompt_free_paired" and args.data_regime != "paired":
        raise ValueError("--training_regime prompt_free_paired currently requires --data_regime paired.")

    if args.dataset_type == "instrument":
        if len(args.train_songs) == 0 and args.data_regime in {"paired", "unpaired"}:
            raise ValueError("--train_songs must be provided when --dataset_type instrument is used.")

        if args.data_regime == "paired":
            train_entries = build_dataset_pairs(dataset_root, args.train_songs, max_pairs=args.max_train_pairs)
            val_entries = build_dataset_pairs(dataset_root, args.val_songs, max_pairs=args.max_val_pairs)

            train_samples = make_bidirectional_samples(train_entries)
            val_samples = make_bidirectional_samples(val_entries)

            print("train pairs:", len(train_entries))
            print("val pairs:", len(val_entries))
            print("train samples:", len(train_samples))
            print("val samples:", len(val_samples))

        else:
            train_entries = build_dataset_unpaired(
                dataset_root,
                args.train_songs,
                max_items_per_domain=args.max_train_pairs,
            )
            val_entries = build_dataset_unpaired(
                dataset_root,
                args.val_songs,
                max_items_per_domain=args.max_val_pairs,
            )

            train_samples = make_unpaired_directional_samples(train_entries)
            val_samples = make_unpaired_directional_samples(val_entries)

            print("train piano items:", len(train_entries["piano"]))
            print("train guitar items:", len(train_entries["guitar"]))
            print("val piano items:", len(val_entries["piano"]))
            print("val guitar items:", len(val_entries["guitar"]))
            print("train samples:", len(train_samples))
            print("val samples:", len(val_samples))

    elif args.dataset_type == "surface":
        if args.data_regime != "unpaired":
            raise ValueError("--dataset_type surface requires unpaired mode.")
    
        if args.training_regime != "prompt":
            raise ValueError("--dataset_type surface supports only prompt regime.")
    
        train_entries = build_surface_dataset_unpaired(
            dataset_root,
            split="train",
            negative_domain=args.surface_negative_domain,
            positive_domain=args.surface_positive_domain,
            max_items_per_domain=args.max_train_pairs,
            val_fraction=args.surface_val_fraction,
            split_seed=args.surface_split_seed,
        )
        val_entries = build_surface_dataset_unpaired(
            dataset_root,
            split="val",
            negative_domain=args.surface_negative_domain,
            positive_domain=args.surface_positive_domain,
            max_items_per_domain=args.max_val_pairs,
            val_fraction=args.surface_val_fraction,
            split_seed=args.surface_split_seed,
        )
    
        train_samples = make_surface_unpaired_directional_samples(
            train_entries,
            negative_domain=args.surface_negative_domain,
            positive_domain=args.surface_positive_domain,
        )
        val_samples = make_surface_unpaired_directional_samples(
            val_entries,
            negative_domain=args.surface_negative_domain,
            positive_domain=args.surface_positive_domain,
        )
    
    elif args.dataset_type == "emotion":
            if args.data_regime != "paired":
                raise ValueError("--dataset_type emotion requires paired mode.")
        
            train_entries = build_emotion_dataset_paired(
                dataset_root,
                split="train",
                negative_domain=args.emotion_negative_domain,
                positive_domain=args.emotion_positive_domain,
                max_pairs=args.max_train_pairs,
                val_fraction=args.emotion_val_fraction,
                split_seed=args.emotion_split_seed,
            )
        
            val_entries = build_emotion_dataset_paired(
                dataset_root,
                split="val",
                negative_domain=args.emotion_negative_domain,
                positive_domain=args.emotion_positive_domain,
                max_pairs=args.max_val_pairs,
                val_fraction=args.emotion_val_fraction,
                split_seed=args.emotion_split_seed,
            )
        
            train_samples = make_bidirectional_samples(train_entries)
            val_samples = make_bidirectional_samples(val_entries)
        
            print("train pairs:", len(train_entries))
            print("val pairs:", len(val_entries))
            print("train samples:", len(train_samples))
            print("val samples:", len(val_samples))

    trainer_cfg = TrainerConfig(
        dataset_root=str(dataset_root),
        train_songs=args.train_songs,
        val_songs=args.val_songs,
        training_regime=args.training_regime,
        prompt_target=args.prompt_target,
        prompt_neutral=args.prompt_neutral,
        prompt_piano=args.prompt_piano,
        prompt_guitar=args.prompt_guitar,
        train_steps=args.train_steps,
        train_cfg_scale=args.train_cfg_scale,
        identity_loss_weight=args.beta_identity,
        pair_loss_weight=args.pair_loss_weight,
        separation_loss_weight=args.separation_loss_weight,
        separation_margin=args.separation_margin,
        num_epochs=args.num_epochs,
        lr_lora=args.lr_lora,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        out_dir=str(out_dir),
        use_null_text_optimization=args.use_nti,
        nti_cache_dir_name=args.nti_cache_dir,
        nti_inner_steps=args.nti_inner_steps,
        nti_lr=args.nti_lr,
        nti_prompt=args.nti_prompt,
        nti_optimize=args.nti_optimize,
        nti_overwrite=args.nti_overwrite,
    )

    trainer = LoRASliderTrainer(editor, trainer_cfg)

    print("Precomputing/loading latent cache...")
    all_entries_for_cache = train_samples + val_samples
    trainer.precompute_latent_cache(
        all_entries_for_cache,
        invert_prompt=args.invert_prompt,
        cache_dir=args.cache_dir,
        store_wav=args.store_wav,
    )
    if args.use_nti:
        print("\nStarting / loading per-sample null-text optimization cache...")
        print(f"NTI inner steps: {args.nti_inner_steps}")
        print(f"NTI lr:          {args.nti_lr}")
        print(f"NTI optimize:    {args.nti_optimize}")
        print(f"NTI prompt:      {repr(args.nti_prompt)}")

        trainer.precompute_null_text_cache(
            train_samples + val_samples,
            cache_dir=args.nti_cache_dir,
            overwrite=args.nti_overwrite,
        )

    print("Starting training...")
    history = trainer.train(train_samples, val_samples=val_samples)

    history_path = out_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()