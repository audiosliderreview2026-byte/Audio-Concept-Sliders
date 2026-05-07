# Towards Continuous Semantic Audio Editing with Concept Sliders

This repository contains interactive audio demos, code, and pretrained checkpoints for the paper:

> *Towards Continuous Semantic Audio Editing with Concept Sliders*  
> Submitted to NeurIPS 2026.

In accordance with double-blind review policies, this repository has been fully anonymized.

---

## Audio demos

Interactive audio demonstrations are available here:  

https://audiosliderreview2026-byte.github.io/Audio-Concept-Sliders

Reviewers are encouraged to listen to several examples across different slider strengths.

---

## LoRA checkpoints

Pretrained LoRA checkpoints are available in the `checkpoints/` folder.

Inference scripts are provided in the `evaluation/` folder. Running inference requires downloading the AudioLDM2 backbone separately.

Please ensure that the number of diffusion steps and the LoRA rank used during inference match those used during training of the selected checkpoint.

---

## Conda environment

A reproducible conda environment can be installed using:

```bash
conda env create -f environment.yml
