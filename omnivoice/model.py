import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from .conditioning import combine_text


def _gumbel_sample(x, temperature):
    u = torch.rand_like(x)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return x / temperature + gumbel_noise


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return comfy.ops.cast_to_input(self.weight, x).to(dtype) * x.to(dtype)


def precompute_freqs_cis(position_ids, inv_freq, device=None):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    if device is not None:
        inv_freq_expanded = inv_freq_expanded.to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.unsqueeze(1), sin.unsqueeze(1)


def apply_rope(xq, xk, freqs_cis):
    cos, sin = freqs_cis
    cos = cos.to(dtype=xq.dtype, device=xq.device)
    sin = sin.to(dtype=xq.dtype, device=xq.device)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_ndim, k_ndim = xq.ndim, xk.ndim
    if q_ndim == 3:
        xq = xq.unsqueeze(0)
    if k_ndim == 3:
        xk = xk.unsqueeze(0)

    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)

    if q_ndim == 3:
        xq = xq.squeeze(0)
    if k_ndim == 3:
        xk = xk.squeeze(0)
    return xq, xk


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.inner_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.q_proj = operations.Linear(hidden_size, self.inner_size, bias=False, device=device, dtype=dtype)
        self.k_proj = operations.Linear(hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)
        self.v_proj = operations.Linear(hidden_size, self.kv_size, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(self.inner_size, hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        if freqs_cis is not None:
            xq, xk = apply_rope(xq, xk, freqs_cis)

        if self.num_heads != self.num_kv_heads:
            xk, xv = comfy.ops.repeat_kv_for_gqa(xk, xv, self.num_heads, 1)
        q, k, v = AttentionTensorContainer(xq), AttentionTensorContainer(xk), AttentionTensorContainer(xv)
        output = optimized_attention(q, k, v, self.num_heads, mask=attention_mask, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None, operations=None):
        super().__init__()
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, device=device, dtype=dtype, operations=operations)
        self.mlp = MLP(hidden_size, intermediate_size, device=device, dtype=dtype, operations=operations)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask=attention_mask, freqs_cis=freqs_cis)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class OmniVoiceLLM(nn.Module):
    def __init__(self, llm_config, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_size = llm_config.get("hidden_size", 1024)
        num_layers = llm_config.get("num_hidden_layers", 28)
        num_heads = llm_config.get("num_attention_heads", 16)
        num_kv_heads = llm_config.get("num_key_value_heads", 8)
        head_dim = llm_config.get("head_dim", 128)
        intermediate_size = llm_config.get("intermediate_size", 3072)
        rms_norm_eps = llm_config.get("rms_norm_eps", 1e-6)
        vocab_size = llm_config.get("vocab_size", 151676)

        self.embed_tokens = operations.Embedding(vocab_size, self.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.hidden_size, num_heads, num_kv_heads,
                head_dim, intermediate_size, rms_norm_eps,
                device=device, dtype=dtype, operations=operations,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(self.hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, freqs_cis=freqs_cis)
        return self.norm(x)


class OmniVoiceModel(nn.Module):
    """OmniVoice: Qwen3-based multimodal model for audio generation.

    State-dict layout matches the reference checkpoint exactly:
    `llm.*`, `audio_embeddings`, `audio_heads`, `codebook_layer_offsets`.
    """

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.operations = operations

        llm_config = config.get("llm_config", {})
        self.hidden_size = llm_config.get("hidden_size", 1024)

        self.num_audio_codebook = config.get("num_audio_codebook", 8)
        self.audio_vocab_size = config.get("audio_vocab_size", 1025)
        self.audio_mask_id = config.get("audio_mask_id", 1024)

        ops = operations
        self.llm = OmniVoiceLLM(llm_config, device=device, dtype=dtype, operations=ops)
        rope_theta = llm_config.get("rope_parameters", {}).get("rope_theta", 1000000)
        head_dim = llm_config.get("head_dim", 128)
        rope_inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("rope_inv_freq", rope_inv_freq.to(dtype), persistent=False)
        self.audio_embeddings = ops.Embedding(
            self.num_audio_codebook * self.audio_vocab_size, self.hidden_size, device=device, dtype=dtype
        )
        self.audio_heads = ops.Linear(
            self.hidden_size, self.num_audio_codebook * self.audio_vocab_size, bias=False, device=device, dtype=dtype
        )
        self.register_buffer("codebook_layer_offsets", torch.empty(self.num_audio_codebook, dtype=torch.long, device=device))

    def forward(self, input_ids, audio_mask, attention_mask=None):
        """Forward pass.

        input_ids: [batch, num_audio_codebook, seq] token ids; layer 0 holds
            text ids, all layers hold audio ids at audio positions.
        audio_mask: [batch, seq] bool, True at audio positions.
        attention_mask: optional broadcastable bool mask (True = attend).
        Returns {"logits": [batch, num_audio_codebook, seq, audio_vocab_size]}.
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[2]

        text_embeds = self.llm.embed_tokens(input_ids[:, 0, :])

        offsets = self.codebook_layer_offsets.to(device=input_ids.device).view(1, -1, 1)
        shifted_ids = input_ids * audio_mask.unsqueeze(1) + offsets
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        x = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(position_ids, self.rope_inv_freq, x.device)

        x = self.llm(x, attention_mask=attention_mask, freqs_cis=freqs_cis)
        logits = self.audio_heads(x).view(
            batch_size, seq_len, self.num_audio_codebook, self.audio_vocab_size
        ).permute(0, 2, 1, 3)

        return {"logits": logits}

    def build_inference_inputs(self, text, num_target_tokens, ref_text=None, ref_audio_tokens=None,
                                 lang=None, instruct=None, denoise=True, device=None):
        """Build [1, codebooks, seq] input ids and [1, seq] audio mask for inference."""
        codebooks = self.num_audio_codebook
        mask_id = self.audio_mask_id

        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        style_text += "<|lang_start|>{}<|lang_end|>".format(lang if lang else "None")
        style_text += "<|instruct_start|>{}<|instruct_end|>".format(instruct if instruct else "None")
        style_tokens = self.text_tokenizer.encode(style_text).repeat(codebooks, 1).unsqueeze(0)

        full_text = combine_text(text, ref_text)
        wrapped_text = "<|text_start|>{}<|text_end|>".format(full_text)
        text_tokens = self.text_tokenizer.encode_tagged(wrapped_text).repeat(codebooks, 1).unsqueeze(0)

        target_audio_tokens = torch.full(
            (1, codebooks, num_target_tokens), mask_id, dtype=torch.long,
        )

        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(style_tokens.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(1, cond_total_length, dtype=torch.bool)
        cond_audio_mask[0, cond_audio_start_idx:] = True

        if device is not None:
            cond_input_ids = cond_input_ids.to(device)
            cond_audio_mask = cond_audio_mask.to(device)
        return {"input_ids": cond_input_ids, "audio_mask": cond_audio_mask}

    def generate(self, cond_input_ids, cond_audio_mask, target_len, num_steps=32, guidance_scale=2.0,
                 t_shift=0.1, layer_penalty=5.0, position_temp=5.0, class_temp=0.0, callback=None):
        """Iterative masked decoding with classifier-free guidance.

        Returns audio token ids shaped [1, num_audio_codebook, target_len].
        """
        device = cond_input_ids.device
        codebooks = self.num_audio_codebook
        vocab_size = self.audio_vocab_size
        mask_id = self.audio_mask_id

        c_len = cond_input_ids.shape[2]
        u_len = target_len

        batch_input_ids = torch.full((2, codebooks, c_len), mask_id, dtype=torch.long, device=device)
        batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool, device=device)
        batch_attention_mask = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=device)

        batch_input_ids[0, :, :c_len] = cond_input_ids
        batch_audio_mask[0, :c_len] = cond_audio_mask
        batch_attention_mask[0, :, :c_len, :c_len] = True

        batch_input_ids[1, :, :u_len] = cond_input_ids[..., -u_len:]
        batch_audio_mask[1, :u_len] = cond_audio_mask[..., -u_len:]
        batch_attention_mask[1, :, :u_len, :u_len] = True
        if c_len > u_len:
            pad_diag = torch.arange(u_len, c_len, device=device)
            batch_attention_mask[1, :, pad_diag, pad_diag] = True

        tokens = torch.full((1, codebooks, target_len), mask_id, dtype=torch.long, device=device)

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

        total_mask = target_len * codebooks
        remaining = total_mask
        layer_ids = torch.arange(codebooks, device=device).view(1, -1, 1)

        for step in range(num_steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logits = self.forward(batch_input_ids, batch_audio_mask, batch_attention_mask)["logits"].float()

            c_logits = logits[0:1, :, c_len - target_len:c_len, :]
            u_logits = logits[1:2, :, :target_len, :]

            if guidance_scale != 0:
                c_log_probs = F.log_softmax(c_logits, dim=-1)
                u_log_probs = F.log_softmax(u_logits, dim=-1)
                log_probs = F.log_softmax(c_log_probs + guidance_scale * (c_log_probs - u_log_probs), dim=-1)
            else:
                log_probs = F.log_softmax(c_logits, dim=-1)

            log_probs[..., mask_id] = -float("inf")

            if class_temp > 0.0:
                k = math.ceil(0.1 * vocab_size)
                topk_vals, topk_inds = log_probs.topk(k, dim=-1)
                filtered = torch.full_like(log_probs, float("-inf"))
                filtered.scatter_(-1, topk_inds, topk_vals)
                pred_tokens = _gumbel_sample(filtered, class_temp).argmax(dim=-1)
            else:
                pred_tokens = log_probs.argmax(dim=-1)

            scores = log_probs.max(dim=-1).values
            scores = scores - layer_ids * layer_penalty

            if position_temp > 0.0:
                scores = _gumbel_sample(scores, position_temp)

            sample_tokens = tokens[0:1, :, :target_len]
            scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

            if step == num_steps - 1:
                k = remaining
            else:
                k = min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step]).item()), remaining)
            k = int(k)
            if k <= 0:
                continue

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

            tokens[0:1, :, :target_len] = sample_tokens
            batch_input_ids[0:1, :, c_len - target_len:c_len] = sample_tokens
            batch_input_ids[1:2, :, :target_len] = sample_tokens
            remaining -= k
            if callback is not None:
                callback(step + 1)

        return tokens

    @classmethod
    def from_state_dict(cls, sd, config, device=None, dtype=None, operations=None):
        return cls(config, device=device, dtype=dtype, operations=operations)
