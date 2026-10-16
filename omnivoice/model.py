import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen3_06BConfig
from .conditioning import combine_text


def _gumbel_sample(x, temperature, generator):
    u = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x / temperature - torch.log(-torch.log(u + 1e-10) + 1e-10)


def _unmask_schedule(total, num_steps, t_shift):
    """Tokens revealed per step, from upstream's shifted linear time schedule (computed on CPU like upstream)."""
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
    timesteps = (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).tolist()
    schedule, remaining = [], total
    for step in range(num_steps):
        k = remaining if step == num_steps - 1 else min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), remaining)
        schedule.append(k)
        remaining -= k
    return schedule


class OmniVoiceModel(nn.Module):
    """OmniVoice: a bidirectional Qwen3 over interleaved text and 8-codebook audio tokens.

    State-dict layout matches the reference checkpoint: `llm.*`, `audio_embeddings`,
    `audio_heads`, `codebook_layer_offsets`.
    """

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        llm = config.get("llm_config", {})
        llm_config = Qwen3_06BConfig(
            vocab_size=llm.get("vocab_size", 151676),
            hidden_size=llm.get("hidden_size", 1024),
            intermediate_size=llm.get("intermediate_size", 3072),
            num_hidden_layers=llm.get("num_hidden_layers", 28),
            num_attention_heads=llm.get("num_attention_heads", 16),
            num_key_value_heads=llm.get("num_key_value_heads", 8),
            rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
            rope_theta=llm.get("rope_parameters", {}).get("rope_theta", 1000000.0),
        )
        llm_config.head_dim = llm.get("head_dim", 128)
        self.hidden_size = llm_config.hidden_size
        self.num_audio_codebook = config.get("num_audio_codebook", 8)
        self.audio_vocab_size = config.get("audio_vocab_size", 1025)
        self.audio_mask_id = config.get("audio_mask_id", 1024)

        self.llm = Llama2_(llm_config, device=device, dtype=dtype, ops=operations)
        self.audio_embeddings = operations.Embedding(self.num_audio_codebook * self.audio_vocab_size, self.hidden_size, device=device, dtype=dtype)
        self.audio_heads = operations.Linear(self.hidden_size, self.num_audio_codebook * self.audio_vocab_size, bias=False, device=device, dtype=dtype)
        self.register_buffer("codebook_layer_offsets", torch.empty(self.num_audio_codebook, dtype=torch.long, device=device))

    def forward(self, input_ids, audio_mask, target_len):
        """input_ids: [1, codebooks, seq] (row 0 holds text ids), audio_mask: [1, seq] bool.

        Every position attends to every other (no mask), as in upstream inference.
        Returns logits [1, codebooks, target_len, audio_vocab] for the last target_len positions.
        """
        seq_len = input_ids.shape[2]
        text_embeds = self.llm.embed_tokens(input_ids[:, 0, :])
        offsets = comfy.model_management.cast_to(self.codebook_layer_offsets, device=input_ids.device)
        shifted_ids = input_ids * audio_mask.unsqueeze(1) + offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
        x = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

        freqs_cis = self.llm.compute_freqs_cis(torch.arange(seq_len, device=x.device).unsqueeze(0), x.device)
        attention = optimized_attention_for_device(x.device)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.llm.layers), x.device, {"prefetch_dynamic_vbars": True})
        for layer in self.llm.layers:
            def core():
                nonlocal x
                x = layer(x=x, freqs_cis=freqs_cis, optimized_attention=attention)[0]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, layer, x.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None, malloc_scope="block")

        logits = self.audio_heads(self.llm.norm(x[:, -target_len:]))
        return logits.view(1, target_len, self.num_audio_codebook, self.audio_vocab_size).permute(0, 2, 1, 3)

    def build_inference_inputs(self, text, num_target_tokens, ref_text=None, ref_audio_tokens=None,
                               lang=None, instruct=None, denoise=True, device=None):
        """Build [1, codebooks, seq] input ids and [1, seq] audio mask for inference."""
        codebooks = self.num_audio_codebook
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        style_text += "<|lang_start|>{}<|lang_end|>".format(lang if lang else "None")
        style_text += "<|instruct_start|>{}<|instruct_end|>".format(instruct if instruct else "None")
        style_tokens = self.text_tokenizer.encode(style_text).repeat(codebooks, 1).unsqueeze(0)

        wrapped_text = "<|text_start|>{}<|text_end|>".format(combine_text(text, ref_text))
        text_tokens = self.text_tokenizer.encode_tagged(wrapped_text).repeat(codebooks, 1).unsqueeze(0)

        parts = [style_tokens, text_tokens]
        num_audio = num_target_tokens
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).cpu())
            num_audio += ref_audio_tokens.shape[-1]
        parts.append(torch.full((1, codebooks, num_target_tokens), self.audio_mask_id, dtype=torch.long))
        input_ids = torch.cat(parts, dim=2)
        audio_mask = torch.zeros(1, input_ids.shape[2], dtype=torch.bool)
        audio_mask[0, -num_audio:] = True
        return {"input_ids": input_ids.to(device), "audio_mask": audio_mask.to(device)}

    def generate(self, input_ids, audio_mask, target_len, generator, num_steps=32, guidance_scale=2.0,
                 t_shift=0.1, layer_penalty=5.0, position_temp=5.0, class_temp=0.0, callback=None):
        """Iterative masked decoding with classifier-free guidance.

        Upstream batches the unconditional branch as a padded row whose padding only attends to
        itself; its target positions see exactly the target tokens, so it is run here as its own
        unpadded sequence, which gives the same logits without the padding work.
        Returns audio token ids [1, num_audio_codebook, target_len].
        """
        input_ids = input_ids.clone()
        tokens = input_ids[:, :, -target_len:].clone()
        uncond_mask = audio_mask[:, -target_len:]
        layer_ids = torch.arange(self.num_audio_codebook, device=input_ids.device).view(1, -1, 1)

        for step, k in enumerate(_unmask_schedule(target_len * self.num_audio_codebook, num_steps, t_shift)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            log_probs = F.log_softmax(self.forward(input_ids, audio_mask, target_len).float(), dim=-1)
            if guidance_scale != 0:
                u_log_probs = F.log_softmax(self.forward(tokens, uncond_mask, target_len).float(), dim=-1)
                log_probs = F.log_softmax(log_probs + guidance_scale * (log_probs - u_log_probs), dim=-1)
            log_probs[..., self.audio_mask_id] = -float("inf")

            if class_temp > 0.0:
                topk_vals, topk_inds = log_probs.topk(math.ceil(0.1 * self.audio_vocab_size), dim=-1)
                filtered = torch.full_like(log_probs, float("-inf")).scatter_(-1, topk_inds, topk_vals)
                pred_tokens = _gumbel_sample(filtered, class_temp, generator).argmax(dim=-1)
            else:
                pred_tokens = log_probs.argmax(dim=-1)

            scores = log_probs.max(dim=-1).values - layer_ids * layer_penalty
            if position_temp > 0.0:
                scores = _gumbel_sample(scores, position_temp, generator)
            scores.masked_fill_(tokens != self.audio_mask_id, -float("inf"))

            if k > 0:
                topk_idx = scores.flatten().topk(k).indices
                tokens.view(-1)[topk_idx] = pred_tokens.flatten()[topk_idx]
                input_ids[:, :, -target_len:] = tokens
            if callback is not None:
                callback(step + 1)
        return tokens
