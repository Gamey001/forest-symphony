"""Follows the bird's pitch over time by finding the loudest frequency in each short slice.

Most birdsong is close to a pure whistle, so "the loudest frequency" is a reliable, fast and
explainable pitch estimate. Slices that are quiet or noise-like (no clear peak) are skipped.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from app.audio.loader import AudioClip

WINDOW_S = 0.04
HOP_S = 0.01
MIN_HZ = 250.0
MAX_HZ = 10_000.0
LOUDNESS_RANGE_DB = 30.0  # slices more than 30 dB below the loudest are ignored
MIN_PEAKINESS_DB = 12.0  # the peak must stand this far above the typical level (tonal, not noise)
FUNDAMENTAL_RANGE_DB = 20.0  # a lower note within 20 dB of the loudest overtone counts as real


@dataclass(frozen=True)
class PitchContour:
    times_s: np.ndarray  # time of each tonal slice
    hz: np.ndarray  # its pitch
    hop_s: float

    @property
    def is_empty(self) -> bool:
        return self.hz.size == 0


def track_pitch(clip: AudioClip) -> PitchContour:
    sr = clip.sample_rate
    nperseg = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)
    freqs, times, spec = stft(clip.samples, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    power = np.abs(spec) ** 2

    band = (freqs >= MIN_HZ) & (freqs <= min(MAX_HZ, sr / 2 - 1))
    band_freqs, band_power = freqs[band], power[band]
    if band_power.size == 0:
        return PitchContour(np.array([]), np.array([]), HOP_S)

    peak_idx = np.argmax(band_power, axis=0)
    cols = np.arange(band_power.shape[1])
    peak_power = band_power[peak_idx, cols]
    peak_db = 10 * np.log10(peak_power + 1e-12)
    median_db = 10 * np.log10(np.median(band_power, axis=0) + 1e-12)

    tonal = (peak_db > peak_db.max() - LOUDNESS_RANGE_DB) & (peak_db - median_db > MIN_PEAKINESS_DB)

    # Parabolic interpolation refines the peak between FFT bins for a more accurate pitch.
    idx = np.clip(peak_idx, 1, len(band_freqs) - 2)
    left = np.log(band_power[idx - 1, cols] + 1e-12)
    mid = np.log(band_power[idx, cols] + 1e-12)
    right = np.log(band_power[idx + 1, cols] + 1e-12)
    denom = left - 2 * mid + right
    offset = np.where(np.abs(denom) > 1e-12, 0.5 * (left - right) / denom, 0.0)
    bin_hz = freqs[1] - freqs[0]
    hz = band_freqs[idx] + np.clip(offset, -0.5, 0.5) * bin_hz
    hz = _prefer_fundamental(hz, peak_power, power, freqs, bin_hz)

    return PitchContour(times_s=times[tonal], hz=hz[tonal], hop_s=HOP_S)


def _prefer_fundamental(
    hz: np.ndarray, peak_power: np.ndarray, power: np.ndarray, freqs: np.ndarray, bin_hz: float
) -> np.ndarray:
    """Many calls (owls especially) have overtones louder than the note itself.

    If there is real energy at 1/3 or 1/2 of the loudest frequency, that lower frequency is the
    note we actually hear, so we use it instead.
    """
    result = hz.copy()
    min_power = peak_power * 10 ** (-FUNDAMENTAL_RANGE_DB / 10)
    cols = np.arange(len(hz))
    for divisor in (2, 3):  # 3 runs last so the lowest valid candidate wins
        candidate = hz / divisor
        center = np.round(candidate / bin_hz).astype(int)
        # Look a couple of bins either side: overtones are rarely perfect multiples.
        nearby = np.stack(
            [power[np.clip(center + d, 0, len(freqs) - 1), cols] for d in range(-2, 3)]
        ).max(axis=0)
        accept = (candidate >= MIN_HZ) & (nearby >= min_power)
        result = np.where(accept, candidate, result)
    return result
