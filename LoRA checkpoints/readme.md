
Available checkpoints:

Emotion slider with null-text inversion:
- 200 steps LoRA rank=8

Emotion slider without NTI:
- 200 denoising steps LoRA rank=8
- 50 denoising steps, LoRA rank=4

Surface slider between leaves and wooden stairs:
- 50 denoising steps, LoRA rank=4

Instrument slider between acoustic guitar and piano:
- 200 denoising steps LoRA rank=8
- 50 denoising steps, LoRA rank=4

Early empirical results have shown that training LoRA with 50 steps was sufficient to get significant results but increasing this number led to higher quality samples.

Increasing the rank of the LoRA from 4 to 8 didn't change much the effect of the slider.

Adding per-instance null-text optimizations steps led to slightly better identity preservation but the tradeoff was significant training time since we need to optimize the null-text embedding for each new sample on which we want to use the slider.
