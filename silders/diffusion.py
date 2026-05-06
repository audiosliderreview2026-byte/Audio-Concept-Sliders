from __future__ import annotations

import torch


def build_alpha_cache(pipe):
    return pipe.scheduler.alphas_cumprod.to(device=pipe.device, dtype=torch.float32)


def get_alpha_bar_cached(alphas_cumprod, t):
    return alphas_cumprod[int(t.item())]


def eps_x0_from_model_output(pipe, alphas_cumprod, model_output, x_t, t):
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


def ddim_inverse_step_scheduler_aware(pipe, alphas_cumprod, x_t, model_output, t, t_next):
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


def freeze_module_params(m):
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
        encoder_attention_mask_1=None,
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
        encoder_attention_mask_1=None,
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
    force_cpu_unet: bool = False,
    cpu_grad_mode: bool = False,
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
            encoder_attention_mask_1=None,
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
    else:
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
