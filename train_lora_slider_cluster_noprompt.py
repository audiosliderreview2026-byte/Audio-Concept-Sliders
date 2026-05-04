import argparse
import json
from pathlib import Path

import soundfile as sf
import torch  # <-- added

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AudioLDM2 LoRA slider on paired segments without semantic prompts.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--train_songs", nargs="+", required=True)
    p.add_argument("--val_songs", nargs="*", default=[])

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

    # ✅ NEW ARGUMENT
    p.add_argument("--resume_from", type=str, default=None, help="Path to a LoRA checkpoint (.pt) to resume from")

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

    # ✅ LOAD CHECKPOINT HERE (after LoRA injection)
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading LoRA checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        missing, unexpected = editor.pipe.unet.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    print("Building dataset pairs...")
    train_pairs = build_dataset_pairs(dataset_root, args.train_songs, max_pairs=args.max_train_pairs)
    val_pairs = build_dataset_pairs(dataset_root, args.val_songs, max_pairs=args.max_val_pairs)

    train_samples = make_bidirectional_samples(train_pairs)
    val_samples = make_bidirectional_samples(val_pairs)

    print("train pairs:", len(train_pairs))
    print("val pairs:", len(val_pairs))
    print("train samples:", len(train_samples))
    print("val samples:", len(val_samples))

    trainer_cfg = TrainerConfig(
        dataset_root=str(dataset_root),
        train_songs=args.train_songs,
        val_songs=args.val_songs,
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
    )

    trainer = LoRASliderTrainer(editor, trainer_cfg)

    print("Precomputing/loading latent cache...")
    all_pairs_for_cache = train_pairs + val_pairs
    trainer.precompute_latent_cache(
        all_pairs_for_cache,
        invert_prompt=args.invert_prompt,
        cache_dir=args.cache_dir,
        store_wav=args.store_wav,
    )

    print("Starting training...")
    history = trainer.train(train_samples, val_samples=val_samples)

    history_path = out_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()