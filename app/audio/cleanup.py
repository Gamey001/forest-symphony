"""Cleans an outdoor recording: removes wind rumble, steady hiss and stretches of dead air."""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import butter, istft, sosfiltfilt, stft

from app.audio.loader import AudioClip
from app.errors import AudioTooQuietError

FRAME_S = 0.02
PADDING_S = 0.1  # keep a little audio around each sound so chirps aren't clipped
MIN_ACTIVE_S = 0.3
ABSOLUTE_FLOOR_DB = -60.0  # anything quieter than this is effectively silence
NOISE_PERCENTILE = 30  # a band's background level = how loud it is 30% of the time
GATE_FACTOR = 2.0  # sounds must be twice (+6 dB) the background level to pass
NOISE_GAIN = 0.2  # background is turned down to 20%, not muted, so it sounds natural


def remove_wind(clip: AudioClip, cutoff_hz: float) -> AudioClip:
    """High-pass filter: wind and handling noise are low-pitched, birdsong is not."""
    sos = butter(4, cutoff_hz, btype="highpass", fs=clip.sample_rate, output="sos")
    return AudioClip(sosfiltfilt(sos, clip.samples).astype(np.float32), clip.sample_rate)


def _frames(samples: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = len(samples) // frame_len
    return samples[: n_frames * frame_len].reshape(n_frames, frame_len)


def _frame_levels_db(samples: np.ndarray, frame_len: int) -> np.ndarray:
    rms = np.sqrt(np.mean(_frames(samples, frame_len) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10)


def reduce_hiss(clip: AudioClip) -> AudioClip:
    """Turn down steady background noise (stream, insects, mic hiss) with a spectral gate.

    For every pitch band we learn its usual background level (a low percentile over time), then
    keep only moments clearly louder than that. Calls that come and go, like owl hoots, survive;
    sounds that never stop are treated as background.
    """
    nperseg = 1024
    _, _, spec = stft(clip.samples, fs=clip.sample_rate, nperseg=nperseg)
    magnitude = np.abs(spec)
    noise_floor = np.percentile(magnitude, NOISE_PERCENTILE, axis=1, keepdims=True)

    keep = (magnitude > noise_floor * GATE_FACTOR).astype(float)
    # Smooth the on/off decision so the gate doesn't produce warbling artefacts.
    keep = uniform_filter(keep, size=(3, 5))
    gain = NOISE_GAIN + (1 - NOISE_GAIN) * keep

    _, cleaned = istft(spec * gain, fs=clip.sample_rate, nperseg=nperseg)
    return AudioClip(cleaned[: len(clip.samples)].astype(np.float32), clip.sample_rate)


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
