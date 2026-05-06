from __future__ import annotations

import torch
from transformers import VitsModel


@torch.no_grad()
def encode_prompt_fixed(pipe, prompt: str, negative_prompt: str = "", max_new_tokens: int = 8):
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
