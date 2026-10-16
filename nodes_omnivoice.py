import json

import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.storage
import torch

from .omnivoice import audio_utils
from .omnivoice.chunking import chunk_text_punctuation, cross_fade_chunks
from .omnivoice.conditioning import (
    ZH_RE,
    INSTRUCT_ACCENTS,
    INSTRUCT_AGE,
    INSTRUCT_DIALECTS,
    INSTRUCT_GENDER,
    INSTRUCT_PITCH,
    INSTRUCT_STYLE,
    add_punctuation,
    resolve_instruct,
    resolve_language,
)
from .omnivoice.duration import RuleDurationEstimator
from .omnivoice.model import OmniVoiceModel
from .omnivoice.text_tokenizer import QwenBpeTokenizer
from .omnivoice.tokenizer import OmniVoiceTokenizer


class OmniVoiceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "omnivoice_name": (folder_paths.get_filename_list("omnivoice"), {
                "tooltip": "Single-file OmniVoice checkpoint from huggingface.co/Aero-Ex/OmniVoice."}),
        }}

    RETURN_TYPES = ("OMNIVOICE", "OMNIVOICE_TOKENIZER")
    RETURN_NAMES = ("model", "audio_tokenizer")
    FUNCTION = "execute"
    CATEGORY = "OmniVoice"

    def execute(self, omnivoice_name):
        model_path = folder_paths.get_full_path_or_raise("omnivoice", omnivoice_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}

        if (metadata.get("omnivoice.format_version") != "1"
                or "llm.layers.0.self_attn.q_proj.weight" not in sd
                or "audio_tokenizer.quantizer.quantizers.0.codebook.embed" not in sd):
            raise RuntimeError(
                "ERROR: not a single-file OmniVoice checkpoint. "
                "Use omnivoice.safetensors or omnivoice-int8.safetensors from huggingface.co/Aero-Ex/OmniVoice."
            )

        config = json.loads(metadata["omnivoice.config"])
        codec_config = json.loads(metadata["omnivoice.audio_tokenizer.config"])
        num_quantizers = 0
        while "audio_tokenizer.quantizer.quantizers.{}.codebook.embed".format(num_quantizers) in sd:
            num_quantizers += 1
        codec_config["num_quantizers"] = num_quantizers
        hop_length = codec_config.get("acoustic_model_config", {}).get("hop_length", 960)
        if "semantic_downsample_factor" not in codec_config:
            codec_config["semantic_downsample_factor"] = int(
                hop_length / (codec_config.get("sample_rate", 24000) / codec_config.get("semantic_sample_rate", 16000))
                / codec_config.get("downsample_factor", 320)
            )

        codec_prefix = "audio_tokenizer."
        codec_sd = {k[len(codec_prefix):]: v for k, v in sd.items() if k.startswith(codec_prefix)}
        model_sd = {k: v for k, v in sd.items() if not k.startswith(codec_prefix)}

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.bfloat16, torch.float32])
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device, supported_dtypes=[torch.bfloat16, torch.float32])
        # INT8 checkpoints carry per-layer .comfy_quant keys, consumed by the mixed-precision ops.
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
            codec_ops = comfy.ops.mixed_precision_ops(quant, torch.float32)
        else:
            operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device)
            codec_ops = comfy.ops.disable_weight_init

        fast_disk = comfy.storage.state_dict_fast_disk(sd)
        model = OmniVoiceModel(config, device=offload_device, dtype=dtype, operations=operations)
        model_patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device, fast_disk=fast_disk)
        # Dynamic VRAM can use the checkpoint tensors in place when they are already in the compute dtype.
        model.load_state_dict(model_sd, assign=model_patcher.is_dynamic() and model_sd["llm.norm.weight"].dtype == dtype)
        model.text_tokenizer = QwenBpeTokenizer(json.loads(metadata["omnivoice.tokenizer"]))
        tokenizer = OmniVoiceTokenizer(codec_config, device=offload_device, dtype=torch.float32, operations=codec_ops)
        tokenizer.load_state_dict(codec_sd)
        tokenizer_patcher = comfy.model_patcher.CoreModelPatcher(tokenizer, load_device=load_device, offload_device=offload_device, fast_disk=fast_disk)
        return (model_patcher, tokenizer_patcher)


class OmniVoiceVoiceDesign:
    """Per-category dropdowns that build the instruct string for OmniVoiceGenerate."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gender": (["auto"] + INSTRUCT_GENDER, {"default": "auto"}),
            "age": (["auto"] + INSTRUCT_AGE, {"default": "auto"}),
            "pitch": (["auto"] + INSTRUCT_PITCH, {"default": "auto"}),
            "style": (["auto"] + INSTRUCT_STYLE, {"default": "auto"}),
            "accent": (["auto"] + INSTRUCT_ACCENTS, {"default": "auto"}),
            "dialect": (["auto"] + INSTRUCT_DIALECTS, {"default": "auto"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruct",)
    FUNCTION = "execute"
    CATEGORY = "OmniVoice"

    def execute(self, gender="auto", age="auto", pitch="auto", style="auto",
                accent="auto", dialect="auto"):
        picks = [p for p in (gender, age, pitch, style, accent, dialect)
                 if p and p != "auto"]
        return (", ".join(picks),)


class OmniVoiceGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("OMNIVOICE", {"tooltip": "The OmniVoice model to use for generation."}),
            "tokenizer": ("OMNIVOICE_TOKENIZER", {"tooltip": "The OmniVoice audio tokenizer to use."}),
            "prompt": ("STRING", {"multiline": True, "tooltip": "The text prompt to generate audio from."}),
            "num_steps": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1, "tooltip": "Iterative unmasking steps."}),
            "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. 0 skips the unconditional pass."}),
            "class_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Token sampling temperature. 0 is greedy."}),
            "chunk_duration": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 120.0, "step": 0.5, "tooltip": "Target chunk length in seconds for long outputs. 0 disables chunking."}),
            "chunk_threshold": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 3600.0, "step": 1.0, "tooltip": "Estimated output length in seconds above which chunking activates. 0 disables chunking."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for the sampling temperatures."}),
        }, "optional": {
            "position_temperature": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Unmask-order temperature. 0 is greedy."}),
            "ref_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning."}),
            "ref_text": ("STRING", {"multiline": True, "tooltip": "Transcription of the reference audio."}),
            "language": ("STRING", {"default": "", "tooltip": "Language name or code (e.g. English, en). Empty for auto."}),
            "instruct": ("STRING", {"default": "", "tooltip": "Voice style instruction (e.g. female, young adult). Empty for auto."}),
            "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Output duration in seconds. 0 estimates from text length."}),
            "postprocess_output": ("BOOLEAN", {"default": True, "tooltip": "Upstream post-processing: trim long pauses, match the reference loudness (or normalize), fade and pad the edges."}),
            "denoise": ("BOOLEAN", {"default": True, "tooltip": "With a reference voice, ask for clean studio output. Off keeps the reference's recording character too, which clones the voice slightly more closely but also copies its noise and room."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "OmniVoice"

    def execute(self, model, tokenizer, prompt, num_steps=32, guidance_scale=2.0, class_temperature=0.0,
                chunk_duration=15.0, chunk_threshold=30.0, seed=0, position_temperature=5.0,
                ref_audio=None, ref_text="", language="", instruct="", duration=0.0, postprocess_output=True, denoise=True):
        comfy.model_management.load_models_gpu([model, tokenizer])
        llm, codec = model.model, tokenizer.model
        device = comfy.model_management.get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        lang = resolve_language(language or None)
        instruct_resolved = resolve_instruct(instruct or None, use_zh=bool(prompt and ZH_RE.search(prompt)))

        ref_audio_tokens = None
        ref_rms = None
        if ref_audio is not None:
            if not ref_text:
                raise RuntimeError(
                    "OmniVoiceGenerate: ref_text is required when ref_audio is provided "
                    "(the native port has no ASR auto-transcription)."
                )
            ref_wav = comfy.audio.resample(ref_audio["waveform"][0].float().mean(0), ref_audio["sample_rate"], codec.sample_rate).cpu()
            ref_rms = float(ref_wav.square().mean().sqrt())
            if 0 < ref_rms < 0.1:
                ref_wav = ref_wav * 0.1 / ref_rms
            ref_wav = audio_utils.remove_silence(ref_wav, codec.sample_rate, mid_ms=200, lead_ms=100, trail_ms=200)
            if ref_wav.shape[-1] == 0:
                raise RuntimeError("OmniVoiceGenerate: reference audio is empty after silence removal.")
            ref_wav = ref_wav[:ref_wav.shape[-1] // (codec.pad * 2) * (codec.pad * 2)]
            ref_audio_tokens = codec.encode(ref_wav.to(device).view(1, 1, -1)).squeeze(0)
            ref_text = add_punctuation(ref_text)

        estimator = RuleDurationEstimator()

        def estimate(text, r_text, r_len):
            # Upstream rates the text against the reference, or "Nice to meet you." = 25 frames without one.
            if r_len is None or not r_text:
                r_text, r_len = "Nice to meet you.", 25
            return max(1, int(estimator.estimate_duration(text, r_text, r_len)))

        est_total = estimate(prompt, ref_text, ref_audio_tokens.shape[-1] if ref_audio_tokens is not None else None)
        num_target_tokens = max(1, round(duration * codec.frame_rate)) if duration > 0 else est_total

        chunks = [prompt]
        if chunk_threshold > 0 and chunk_duration > 0 and num_target_tokens > chunk_threshold * codec.frame_rate and prompt:
            text_chunk_len = int(chunk_duration * codec.frame_rate / (num_target_tokens / len(prompt)))
            if text_chunk_len > 0:
                split = chunk_text_punctuation(prompt, text_chunk_len, min_chunk_len=3)
                if len(split) > 1:
                    chunks = split

        pbar = comfy.utils.ProgressBar(num_steps * len(chunks))
        # Upstream _generate_chunked: per-chunk targets are re-estimated from text and scaled by
        # estimate / request when a duration is given. Without a user reference, chunk 0 becomes
        # the voice reference for the later chunks.
        speed = est_total / num_target_tokens if duration > 0 else 1.0
        wav_chunks = []
        chunk_ref_text, chunk_ref_tokens = (ref_text, ref_audio_tokens) if ref_audio_tokens is not None else (None, None)
        for ci, chunk_text in enumerate(chunks):
            if len(chunks) == 1:
                chunk_tokens = num_target_tokens
            else:
                chunk_tokens = estimate(chunk_text, chunk_ref_text, chunk_ref_tokens.shape[-1] if chunk_ref_tokens is not None else None)
                if speed != 1.0:
                    chunk_tokens = max(1, int(chunk_tokens / speed))
            cond = llm.build_inference_inputs(chunk_text, chunk_tokens, ref_text=chunk_ref_text, ref_audio_tokens=chunk_ref_tokens,
                                              lang=lang, instruct=instruct_resolved, denoise=denoise, device=device)
            done = ci * num_steps
            tokens = llm.generate(cond["input_ids"], cond["audio_mask"], chunk_tokens, generator,
                                  num_steps=num_steps, guidance_scale=guidance_scale,
                                  class_temp=class_temperature, position_temp=position_temperature,
                                  callback=lambda s: pbar.update_absolute(done + s))
            if ci == 0 and ref_audio_tokens is None and len(chunks) > 1:
                chunk_ref_text, chunk_ref_tokens = chunk_text, tokens[0]
            wav_chunks.append(codec.decode(tokens)[0].float().cpu())
        waveform = cross_fade_chunks(wav_chunks, codec.sample_rate)
        if postprocess_output:
            waveform = audio_utils.postprocess_output(waveform, codec.sample_rate, ref_rms)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": codec.sample_rate},)


NODE_CLASS_MAPPINGS = {
    "OmniVoiceLoader": OmniVoiceLoader,
    "OmniVoiceVoiceDesign": OmniVoiceVoiceDesign,
    "OmniVoiceGenerate": OmniVoiceGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceLoader": "Load OmniVoice Model",
    "OmniVoiceVoiceDesign": "OmniVoice Voice Design",
    "OmniVoiceGenerate": "OmniVoice Generate",
}
