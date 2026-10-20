"""Reads uploaded audio bytes into a mono float array."""

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.errors import InvalidAudioError


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(data: bytes, max_seconds: float) -> AudioClip:
    """Decode WAV/FLAC/OGG/MP3 bytes, mix to mono and keep at most `max_seconds`."""
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise InvalidAudioError(
            "We couldn't read that file. Please use WAV, FLAC, OGG or MP3 audio."
        ) from exc

    mono = samples.mean(axis=1)
    mono = mono[: int(max_seconds * sample_rate)]
    if len(mono) < sample_rate // 2:
        raise InvalidAudioError("That clip is shorter than half a second. Please record a bit longer.")
    return AudioClip(samples=mono, sample_rate=sample_rate)
