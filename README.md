# ComfyUI-OmniVoice

A native, dependency-free port of the OmniVoice TTS model (Qwen3 LLM + HiggsAudioV2 codec) for ComfyUI. No extra pip packages — runs on stock ComfyUI (INT8 checkpoints use ComfyUI's built-in quantization stack).

## Nodes

- **Load OmniVoice Model** — loads a single-file checkpoint (`models/omnivoice/*.safetensors`). Auto-detects float and INT8-ConvRot checkpoints.
- **OmniVoice Voice Design** — per-category dropdowns (gender, age, pitch, style, accent, dialect) that build the voice-design string. Connect to Generate's `instruct`.
- **OmniVoice Generate** — text-to-speech. Optional voice clone (`ref_audio` + `ref_text`), auto duration estimation, chunked long-form generation (texts over ~30s are split into ~15s chunks and cross-faded, near-constant VRAM).

## Install

Only two ways:

1. **ComfyUI Manager** — Manager → Custom Nodes Manager → search `OmniVoice` → Install. Restart ComfyUI.
2. **Manual** — clone this repo into `ComfyUI/custom_nodes/`:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/Aero-Ex/ComfyUI-OmniVoice
   ```
   Restart ComfyUI. No installtion needed.

## Checkpoints

Download a ready single file from [Aero-Ex/OmniVoice](https://huggingface.co/Aero-Ex/OmniVoice) into `models/omnivoice/`:

- `omnivoice.safetensors` (3.3GB, full quality)
- `omnivoice-int8.safetensors` (1.1GB, INT8-ConvRot, ~86% top-1 agreement with full)


## Vocalization tags

Type these inline in the prompt, exact spelling in brackets — the model speaks them:

- `[laughter]`, `[sigh]`
- `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[confirmation-en]`
- `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`
- `[dissatisfaction-hnn]`

Example:

```
Wait, what [surprise-ah]? You finished it already [question-oh]? Ha [laughter]! I knew you could do it [confirmation-en]. Phew [sigh], now we can finally rest.
```

Keep normal punctuation spacing around tags (`! Ha`, `. Phew` — never `!Ha` or `.Phew`). Crushed punctuation creates rare token pieces that derail takes, especially on the INT8 checkpoint (verified: same prompt babbles 5/5 unspaced, speaks spaced). Anything else in brackets is read out as literal text.

## Usage

`Load OmniVoice Model` → `OmniVoice Generate`. For a designed voice, add `OmniVoice Voice Design` → `instruct`.

Notes:

- `ref_text` is required whenever `ref_audio` is connected (no ASR auto-transcription).
- No output post-processing is applied — use standard ComfyUI audio nodes for gain/fades/trim.
- Checkpoint picks: bf16 for tags + voice-design prompts (most robust); int8 for plain prompts (same quality, smaller). Same seed can give different takes across checkpoints — pin a keeper seed per setup.
