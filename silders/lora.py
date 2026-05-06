from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    target_keywords: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_keywords is None:
            self.target_keywords = ["to_q", "to_k", "to_v", "to_out.0"]


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")

        self.base = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.multiplier = 1.0

        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = self.base.in_features
        out_features = self.base.out_features

        dev = self.base.weight.device
        dt = self.base.weight.dtype

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, self.rank, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False, device=dev, dtype=dt)

        self.reset_parameters_for_training()

    def reset_parameters_for_training(self, std=1e-4):
        # Standard practical LoRA-style init:
        # A = small random, B = zero
        # => initial delta is zero, but gradients are non-zero
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=std)
        nn.init.zeros_(self.lora_B.weight)
    
    def reset_parameters_zero(self):
        # Useful for exact no-op sanity checks
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def reset_parameters_small_random(self, std: float = 1e-4):
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=std)
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=std)

    def set_multiplier(self, value: float):
        self.multiplier = float(value)

    def forward(self, x):
        if self.lora_A.weight.device != x.device or self.lora_A.weight.dtype != x.dtype:
            self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_B.to(device=x.device, dtype=x.dtype)
            self.base.to(device=x.device, dtype=x.dtype)
        base_out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.multiplier * self.scaling * delta


def _get_parent_module_and_child_name(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def _module_name_matches(name: str, keywords: List[str]) -> bool:
    return any(name.endswith(k) for k in keywords)


def find_lora_candidate_linears(model: nn.Module, target_keywords: List[str]) -> List[Tuple[str, nn.Linear]]:
    matches = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _module_name_matches(name, target_keywords):
            matches.append((name, module))
    return matches


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_keywords: Optional[List[str]] = None,
    verbose: bool = True,
):
    if target_keywords is None:
        target_keywords = ["to_q", "to_k", "to_v", "to_out.0"]

    candidates = find_lora_candidate_linears(model, target_keywords)
    injected_names = []

    for full_name, module in candidates:
        parent, child_name = _get_parent_module_and_child_name(model, full_name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        injected_names.append(full_name)

    if verbose:
        print(f"Injected LoRA into {len(injected_names)} linear layers.")
    return injected_names


def inject_lora_into_unet(unet: nn.Module, config: LoRAConfig, verbose: bool = True):
    return inject_lora_into_model(
        unet,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_keywords=config.target_keywords,
        verbose=verbose,
    )


def get_all_lora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_all_lora_multipliers(model: nn.Module, value: float):
    for m in get_all_lora_modules(model):
        m.set_multiplier(value)


def reset_all_lora_zero(model: nn.Module):
    for m in get_all_lora_modules(model):
        m.reset_parameters_zero()


def randomize_all_lora_small(model: nn.Module, std: float = 1e-4):
    for m in get_all_lora_modules(model):
        m.reset_parameters_small_random(std=std)


def freeze_non_lora_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in get_all_lora_modules(model):
        for p in m.lora_A.parameters():
            p.requires_grad_(True)
        for p in m.lora_B.parameters():
            p.requires_grad_(True)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def summarize_lora(model: nn.Module):
    loras = get_all_lora_modules(model)
    total, trainable = count_parameters(model)
    print(f"Number of LoRA modules: {len(loras)}")
    print(f"Total params:      {total:,}")
    print(f"Trainable params:  {trainable:,}")
    if total > 0:
        print(f"Trainable ratio:   {100 * trainable / total:.6f}%")


def export_lora_state_dict(model: nn.Module):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return lora_state


def load_lora_state_dict(model: nn.Module, state_dict: dict):
    module_map = dict(model.named_modules())
    for key, value in state_dict.items():
        if key.endswith(".lora_A.weight"):
            module_name = key[:-len(".lora_A.weight")]
            module_map[module_name].lora_A.weight.data.copy_(value.to(module_map[module_name].lora_A.weight.device))
        elif key.endswith(".lora_B.weight"):
            module_name = key[:-len(".lora_B.weight")]
            module_map[module_name].lora_B.weight.data.copy_(value.to(module_map[module_name].lora_B.weight.device))
