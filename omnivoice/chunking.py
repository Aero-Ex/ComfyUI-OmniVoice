import torch

# Text chunking + chunk joining for long-form generation. Mirrors the reference
# implementation (omnivoice/utils/text.py chunk_text_punctuation and
# omnivoice/utils/audio.py cross_fade_chunks, Apache-2.0, Copyright 2026
# Xiaomi Corp.). Chunking is stdlib-only; joining uses torch ops on the
# decoded waveforms (same fade math as upstream, no numpy roundtrip).

SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")

ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Fr.",
    "Hon.",
    "Pres.",
    "Gov.",
    "Capt.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Col.",
    "Maj.",
    "Lt.",
    "Cmdr.",
    "Sgt.",
    "Cpl.",
    "Co.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "Est.",
    "Dept.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Mt.",
    "Ft.",
    "No.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "i.e.",
    "e.g.",
    "vs.",
    "Vs.",
    "Etc.",
    "approx.",
    "fig.",
    "def.",
}


def chunk_text_punctuation(text, chunk_len, min_chunk_len=None):
    """Split text into chunks at punctuation, avoiding common abbreviations."""
    sentences = []
    current_sentence = []

    tokens_list = list(text)

    for token in tokens_list:
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS)
        ):
            sentences[-1].append(token)
        else:
            current_sentence.append(token)

            if token in SPLIT_PUNCTUATION:
                is_abbreviation = False

                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str:
                        last_word = temp_str.split()[-1]
                        if last_word in ABBREVIATIONS:
                            is_abbreviation = True

                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    if min_chunk_len is not None:
        first_chunk_short_flag = (
            len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        )
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short_flag:
                final_chunks[-1].extend(chunk)
            else:
                if len(chunk) >= min_chunk_len:
                    final_chunks.append(chunk)
                else:
                    if len(final_chunks) == 0:
                        final_chunks.append(chunk)
                    else:
                        final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    return [
        "".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()
    ]


def cross_fade_chunks(chunks, sample_rate, silence_duration=0.3):
    """Concatenate waveforms with silence gaps and cross-fades at boundaries.

    chunks: list of torch tensors, each (C, T). Returns a single (C, T) tensor
    on the first chunk's device/dtype. Same fade math as upstream.
    """
    if len(chunks) == 1:
        return chunks[0]

    device = chunks[0].device
    dtype = chunks[0].dtype
    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].clone()

    for chunk in chunks[1:]:
        other = chunk.to(device=device, dtype=dtype)
        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = torch.linspace(1, 0, fout_n, device=device, dtype=torch.float32).view(
                *([1] * (merged.dim() - 1)), fout_n
            ).to(dtype)
            merged[..., -fout_n:] = merged[..., -fout_n:] * w_out

        silence = torch.zeros(
            (*merged.shape[:-1], silence_n), device=device, dtype=dtype
        )
        fade_in = other.clone()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = torch.linspace(0, 1, fin_n, device=device, dtype=torch.float32).view(
                *([1] * (fade_in.dim() - 1)), fin_n
            ).to(dtype)
            fade_in[..., :fin_n] = fade_in[..., :fin_n] * w_in

        merged = torch.cat([merged, silence, fade_in], dim=-1)

    return merged
