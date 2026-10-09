import math

import torch

# Silence processing for reference-audio preparation. Mirrors the reference
# implementation (which uses pydub under the hood) using torch ops. All
# positions and ranges are milliseconds, exactly like pydub; conversion to
# samples happens only when slicing, with end-clamping and zero-padding
# past the buffer end.


def _to_int16(x):
    return (x * 32768.0).clamp(-32768, 32767).to(torch.int16)


def _from_int16(x):
    return x.to(torch.float64) / 32768.0


def _seg_len_ms(total, sample_rate):
    return round(total * 1000 / sample_rate)


def _parse_ms(val_ms, seg_len_ms, sample_rate):
    if val_ms < 0:
        val_ms = seg_len_ms - abs(val_ms)
    return int(val_ms * sample_rate / 1000)


def _slice_ms(x, sample_rate, start_ms, end_ms=None):
    """Slice samples like pydub's millisecond slicing (clamp + zero-pad)."""
    total = x.shape[0]
    seg_len_ms = _seg_len_ms(total, sample_rate)
    if end_ms is None:
        end_ms = seg_len_ms
    start_ms = min(start_ms, seg_len_ms)
    end_ms = min(end_ms, seg_len_ms)
    a = _parse_ms(start_ms, seg_len_ms, sample_rate)
    b = _parse_ms(end_ms, seg_len_ms, sample_rate)
    data = x[a:b]
    missing = (b - a) - data.shape[0]
    if missing > 0:
        data = torch.cat([data, torch.zeros(missing, dtype=data.dtype, device=data.device)])
    return data


def _dbfs(window):
    if window.numel() == 0:
        return float("-inf")
    rms = window.to(torch.float64).pow(2).mean().sqrt().item()
    if rms <= 0:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)


def detect_leading_silence(x, sample_rate, silence_threshold=-50.0, chunk_ms=10):
    """Return the millisecond index where leading silence ends."""
    total = x.shape[0]
    seg_len_ms = _seg_len_ms(total, sample_rate)
    trim_ms = 0
    while trim_ms < seg_len_ms:
        if not (_dbfs(_slice_ms(x, sample_rate, trim_ms, trim_ms + chunk_ms)) < silence_threshold):
            break
        trim_ms += chunk_ms
    return min(trim_ms, seg_len_ms)


def detect_silence(x, sample_rate, min_silence_ms=1000, silence_thresh=-16, seek_step_ms=1):
    """Return silent [[start, end]] millisecond ranges."""
    total = x.shape[0]
    seg_len_ms = _seg_len_ms(total, sample_rate)
    if seg_len_ms < min_silence_ms:
        return []

    last_slice_start = seg_len_ms - min_silence_ms
    slice_starts = list(range(0, last_slice_start + 1, seek_step_ms))
    if last_slice_start % seek_step_ms:
        slice_starts.append(last_slice_start)

    silence_starts = []
    for i in slice_starts:
        if _dbfs(_slice_ms(x, sample_rate, i, i + min_silence_ms)) <= silence_thresh:
            silence_starts.append(i)

    if not silence_starts:
        return []

    silent_ranges = []
    prev_i = silence_starts.pop(0)
    current_range_start = prev_i
    for silence_start_i in silence_starts:
        continuous = silence_start_i == prev_i + seek_step_ms
        silence_has_gap = silence_start_i > prev_i + min_silence_ms
        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start, prev_i + min_silence_ms])
            current_range_start = silence_start_i
        prev_i = silence_start_i
    silent_ranges.append([current_range_start, prev_i + min_silence_ms])
    return silent_ranges


def detect_nonsilent(x, sample_rate, min_silence_ms=1000, silence_thresh=-16, seek_step_ms=1):
    """Return non-silent [[start, end]] millisecond ranges."""
    seg_len_ms = _seg_len_ms(x.shape[0], sample_rate)
    silent_ranges = detect_silence(x, sample_rate, min_silence_ms, silence_thresh, seek_step_ms)

    if not silent_ranges:
        return [[0, seg_len_ms]]

    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == seg_len_ms:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != seg_len_ms:
        nonsilent_ranges.append([prev_end_i, seg_len_ms])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(x, sample_rate, min_silence_ms=1000, silence_thresh=-16, keep_silence_ms=100, seek_step_ms=1):
    """Split samples on silent sections, keeping some silence around cuts."""
    output_ranges = [
        [start - keep_silence_ms, end + keep_silence_ms]
        for start, end in detect_nonsilent(x, sample_rate, min_silence_ms, silence_thresh, seek_step_ms)
    ]

    for idx in range(len(output_ranges) - 1):
        last_end = output_ranges[idx][1]
        next_start = output_ranges[idx + 1][0]
        if next_start < last_end:
            mid = (last_end + next_start) // 2
            output_ranges[idx][1] = mid
            output_ranges[idx + 1][0] = mid

    return [_slice_ms(x, sample_rate, max(start, 0), end) for start, end in output_ranges]


def remove_silence_edges(x, sample_rate, lead_ms=100, trail_ms=300, silence_threshold=-50.0):
    """Trim edge silences, keeping lead_ms / trail_ms of audio."""
    start_ms = detect_leading_silence(x, sample_rate, silence_threshold)
    start_ms = max(0, start_ms - lead_ms)
    x = _slice_ms(x, sample_rate, start_ms)

    flipped = torch.flip(x, dims=[0])
    start_ms = detect_leading_silence(flipped, sample_rate, silence_threshold)
    start_ms = max(0, start_ms - trail_ms)
    return torch.flip(_slice_ms(flipped, sample_rate, start_ms), dims=[0])


def remove_silence(x, sample_rate, mid_ms=300, lead_ms=100, trail_ms=300):
    """Remove middle silences longer than mid_ms and trim edges."""
    wave = _to_int16(x.reshape(-1))

    if mid_ms > 0:
        segments = split_on_silence(
            wave, sample_rate,
            min_silence_ms=mid_ms, silence_thresh=-50,
            keep_silence_ms=mid_ms, seek_step_ms=10,
        )
        wave = torch.cat(segments, dim=0) if segments else wave[:0]

    wave = remove_silence_edges(wave, sample_rate, lead_ms, trail_ms, -50)
    return _from_int16(wave).to(x.dtype).reshape(x.shape[:-1] + (-1,))
