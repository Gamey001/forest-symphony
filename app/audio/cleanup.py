"""Cleans an outdoor recording: removes wind rumble, steady hiss and stretches of dead air."""

import numpy as np
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt

from app.audio.loader import AudioClip
from app.errors import AudioTooQuietError

FRAME_S = 0.02
PADDING_S = 0.1  # keep a little audio around each sound so chirps aren't clipped
MIN_ACTIVE_S = 0.3
ABSOLUTE_FLOOR_DB = -60.0  # anything quieter than this is effectively silence


def remove_wind(clip: AudioClip, cutoff_hz: float) -> AudioClip:
    """High-pass filter: wind and handling noise are low-pitched, birdsong is not."""
    sos = butter(4, cutoff_hz, btype="highpass", fs=clip.sample_rate, output="sos")
    return AudioClip(sosfiltfilt(sos, clip.samples).astype(np.float32), clip.sample_rate)


def reduce_hiss(clip: AudioClip) -> AudioClip:
    """Subtract the steady background noise (stream, insects, mic hiss)."""
    cleaned = nr.reduce_noise(
        y=clip.samples, sr=clip.sample_rate, stationary=True, prop_decrease=0.8
    )
    return AudioClip(cleaned.astype(np.float32), clip.sample_rate)


def _frame_levels_db(samples: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = len(samples) // frame_len
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(rms + 1e-10)


def trim_silence(clip: AudioClip, threshold_db: float) -> AudioClip:
    """Keep only the loud parts. `threshold_db` is relative to the loudest moment."""
    frame_len = int(FRAME_S * clip.sample_rate)
    levels = _frame_levels_db(clip.samples, frame_len)
    if levels.size == 0 or levels.max() < ABSOLUTE_FLOOR_DB:
        raise AudioTooQuietError("We couldn't hear anything. Try holding the phone closer to the bird.")

    active = levels > levels.max() + threshold_db
    pad = int(PADDING_S / FRAME_S)
    # Widen each active frame by `pad` frames on both sides.
    active = np.convolve(active, np.ones(2 * pad + 1), mode="same") > 0

    mask = np.repeat(active, frame_len)
    trimmed = clip.samples[: len(mask)][mask]
    if len(trimmed) < MIN_ACTIVE_S * clip.sample_rate:
        raise AudioTooQuietError("The recording is mostly silence. Try a longer, closer recording.")
    return AudioClip(trimmed, clip.sample_rate)


def clean_audio(clip: AudioClip, highpass_hz: float, silence_threshold_db: float) -> AudioClip:
    return trim_silence(reduce_hiss(remove_wind(clip, highpass_hz)), silence_threshold_db)
