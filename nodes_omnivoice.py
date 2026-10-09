import json

import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
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
                "tooltip": "Single-file OmniVoice checkpoint (see convert_omnivoice_checkpoint.py)."}),
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
                "Merge the model folder first: "
                "python convert_omnivoice_checkpoint.py <model_folder> <out.safetensors>"
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

        dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.bfloat16, torch.float32])
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, comfy.model_management.get_torch_device(), supported_dtypes=[torch.bfloat16, torch.float32])
        # Native quantized path (e.g. --dtype int8 checkpoints): detected from
        # per-layer .comfy_quant keys, consumed by mixed-precision ops.
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
            codec_ops = comfy.ops.mixed_precision_ops(quant, torch.float32)
        else:
            operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=comfy.model_management.get_torch_device())
            codec_ops = comfy.ops.disable_weight_init

        model = OmniVoiceModel(
            config,
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype,
            operations=operations,
        )
        model.load_state_dict(model_sd, assign=False)
        model.text_tokenizer = QwenBpeTokenizer(json.loads(metadata["omnivoice.tokenizer"]))

        tokenizer = OmniVoiceTokenizer(
            codec_config,
            device=comfy.model_management.unet_offload_device(),
            dtype=torch.float32,
            operations=codec_ops,
        )
        tokenizer.load_state_dict(codec_sd, assign=False)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        model_patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device)
        tokenizer_patcher = comfy.model_patcher.ModelPatcher(
            tokenizer, load_device=load_device, offload_device=offload_device)
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
            "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Classifier-free guidance scale."}),
            "class_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Token sampling temperature. 0 is greedy."}),
            "chunk_duration": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 120.0, "step": 0.5, "tooltip": "Target chunk length in seconds for long outputs. 0 disables chunking."}),
            "chunk_threshold": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 3600.0, "step": 1.0, "tooltip": "Estimated output length in seconds above which chunking activates. 0 disables chunking."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for sampling noise."}),
        }, "optional": {
            "ref_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning."}),
            "ref_text": ("STRING", {"multiline": True, "tooltip": "Transcription of the reference audio."}),
            "language": ("STRING", {"default": "", "tooltip": "Language name or code (e.g. English, en). Empty for auto."}),
            "instruct": ("STRING", {"default": "", "tooltip": "Voice style instruction (e.g. female, young adult). Empty for auto."}),
            "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1, "tooltip": "Output duration in seconds. 0 estimates from text length."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "OmniVoice"

    def execute(self, model, tokenizer, prompt, num_steps=32,
                guidance_scale=2.0, class_temperature=0.0,
                chunk_duration=15.0, chunk_threshold=30.0, seed=0,
                ref_audio=None, ref_text="", language="", instruct="", duration=0.0):
        # Loader wraps both models in ModelPatchers; unwrap for direct use.
        llm = model.model if isinstance(model, comfy.model_patcher.ModelPatcher) else model
        codec = tokenizer.model if isinstance(tokenizer, comfy.model_patcher.ModelPatcher) else tokenizer
        load_device = comfy.model_management.get_torch_device()

        # Dynamic VRAM: evicts other resident models as needed to fit this run.
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            comfy.model_management.load_models_gpu([model], force_full_load=True)
        if isinstance(tokenizer, comfy.model_patcher.ModelPatcher):
            comfy.model_management.load_models_gpu([tokenizer], force_full_load=True)

        try:
            if getattr(llm, "text_tokenizer", None) is None:
                raise RuntimeError("OmniVoiceGenerate: the model has no text tokenizer.")
            torch.manual_seed(seed)

            lang = resolve_language(language or None)
            use_zh = bool(prompt and ZH_RE.search(prompt))
            instruct_resolved = resolve_instruct(instruct or None, use_zh=use_zh)

            ref_audio_tokens = None
            if ref_audio is not None:
                if not ref_text:
                    raise RuntimeError(
                        "OmniVoiceGenerate: ref_text is required when ref_audio is provided "
                        "(the native port has no ASR auto-transcription)."
                    )
                waveform = ref_audio["waveform"]
                if waveform.shape[1] > 1:
                    waveform = waveform.mean(dim=1, keepdim=True)
                if ref_audio["sample_rate"] != codec.sample_rate:
                    waveform = comfy.audio.resample(waveform, ref_audio["sample_rate"], codec.sample_rate)
                ref_wav = waveform.to(load_device).reshape(-1)
                ref_rms = float(torch.sqrt(torch.mean(ref_wav ** 2)))
                if 0 < ref_rms < 0.1:
                    ref_wav = ref_wav * 0.1 / ref_rms
                ref_wav = audio_utils.remove_silence(ref_wav, codec.sample_rate, mid_ms=200, lead_ms=100, trail_ms=200)
                if ref_wav.shape[-1] == 0:
                    raise RuntimeError("OmniVoiceGenerate: reference audio is empty after silence removal.")
                hop_length = codec.pad * 2
                clip_size = int(ref_wav.shape[-1] % hop_length)
                if clip_size > 0:
                    ref_wav = ref_wav[..., :-clip_size]
                with torch.inference_mode():
                    ref_audio_tokens = codec.encode(ref_wav.reshape(1, 1, -1)).squeeze(0)
                ref_text = add_punctuation(ref_text)

            estimator = RuleDurationEstimator()
            if ref_audio_tokens is None or not ref_text:
                base_est_text, base_est_tokens = "Nice to meet you.", 25
            else:
                base_est_text, base_est_tokens = ref_text, ref_audio_tokens.size(-1)
            est_total = max(1, int(estimator.estimate_duration(
                prompt, base_est_text, base_est_tokens)))
            if duration > 0:
                num_target_tokens = max(1, round(duration * codec.frame_rate))
            else:
                # Mirrors upstream _estimate_target_tokens: rule-based estimate
                # against the reference rate, or "Nice to meet you." / 25 tokens
                # when cloning without a reference.
                num_target_tokens = est_total

            def estimate_chunk(chunk_text, r_text, r_len):
                if r_len is None or not r_text:
                    r_text, r_len = "Nice to meet you.", 25
                return max(1, int(estimator.estimate_duration(chunk_text, r_text, r_len)))

            chunks = None
            if (chunk_threshold > 0 and chunk_duration > 0
                    and num_target_tokens > chunk_threshold * codec.frame_rate
                    and len(prompt) > 0):
                avg_tokens_per_char = num_target_tokens / len(prompt)
                text_chunk_len = int(chunk_duration * codec.frame_rate / avg_tokens_per_char) \
                    if avg_tokens_per_char > 0 else 0
                if text_chunk_len > 0:
                    split = chunk_text_punctuation(prompt, text_chunk_len, min_chunk_len=3)
                    if len(split) > 1:
                        chunks = split

            def run_chunk(chunk_text, chunk_tokens, r_text, r_tokens):
                cond = llm.build_inference_inputs(
                    chunk_text, chunk_tokens,
                    ref_text=r_text,
                    ref_audio_tokens=r_tokens,
                    lang=lang,
                    instruct=instruct_resolved,
                    device=load_device,
                )
                with torch.inference_mode():
                    return llm.generate(
                        cond["input_ids"], cond["audio_mask"], chunk_tokens,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        class_temp=class_temperature,
                        callback=lambda s, b=run_chunk.done: pbar.update_absolute(b + s),
                    )

            if chunks is None:
                pbar = comfy.utils.ProgressBar(num_steps)
                run_chunk.done = 0
                with torch.inference_mode():
                    audio_tokens = run_chunk(prompt, num_target_tokens,
                                             ref_text or None, ref_audio_tokens)
                    waveform = codec.decode(audio_tokens).cpu()
            else:
                # Upstream _generate_chunked: per-chunk token targets re-estimated
                # from text; an explicit duration scales them proportionally via
                # speed = estimate / request. Without a user ref, chunk 0 output
                # becomes the voice ref for later chunks.
                pbar = comfy.utils.ProgressBar(num_steps * len(chunks))
                run_chunk.done = 0
                speed = est_total / num_target_tokens if duration > 0 and num_target_tokens > 0 else 1.0
                wav_chunks = []
                first_ref_tokens = None
                with torch.inference_mode():
                    for ci, chunk_text in enumerate(chunks):
                        if ref_audio_tokens is not None:
                            r_text, r_tokens = ref_text or None, ref_audio_tokens
                        elif ci == 0:
                            r_text, r_tokens = None, None
                        else:
                            r_text, r_tokens = chunks[0], first_ref_tokens
                        chunk_tokens = estimate_chunk(
                            chunk_text,
                            r_text if r_text else (ref_text if ref_audio_tokens is not None else None),
                            r_tokens.size(-1) if r_tokens is not None else None,
                        )
                        if speed != 1.0 and speed > 0:
                            chunk_tokens = max(1, int(chunk_tokens / speed))
                        tokens = run_chunk(chunk_text, chunk_tokens, r_text, r_tokens)
                        if ci == 0 and ref_audio_tokens is None:
                            first_ref_tokens = tokens[0].to(load_device)
                        wav_chunks.append(codec.decode(tokens).cpu()[0])
                        del tokens
                        run_chunk.done += num_steps
                waveform = cross_fade_chunks(wav_chunks, codec.sample_rate).unsqueeze(0)
        finally:
            comfy.model_management.soft_empty_cache()

        return ({"waveform": waveform, "sample_rate": codec.sample_rate},)


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
