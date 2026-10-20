"""Turns a melody into a WAV file using plain sine-wave maths (no AI involved)."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from app.music import REST, note_to_hz
from app.schemas import Melody

# A soft, flute-like tone: the main pitch plus two quiet overtones.
HARMONICS = ((1, 1.0), (2, 0.25), (3, 0.08))
ATTACK_S = 0.012
RELEASE_S = 0.08
PEAK_LEVEL = 0.89  # about -1 dBFS, leaves headroom so playback never clips


def _envelope(n_samples: int, sample_rate: int) -> np.ndarray:
    """Fade each note in and out so notes don't start or stop with an audible click."""
    env = np.ones(n_samples)
    attack = min(int(ATTACK_S * sample_rate), n_samples // 2)
    release = min(int(RELEASE_S * sample_rate), n_samples // 2)
    if attack:
        env[:attack] = np.linspace(0, 1, attack)
    if release:
        env[-release:] = np.linspace(1, 0, release) ** 2
    return env


def render_tone(hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    n_samples = int(round(duration_s * sample_rate))
    t = np.arange(n_samples) / sample_rate
    wave = sum(gain * np.sin(2 * np.pi * hz * k * t) for k, gain in HARMONICS)
    return wave * _envelope(n_samples, sample_rate)


def render_melody(melody: Melody, sample_rate: int) -> np.ndarray:
    seconds_per_beat = 60.0 / melody.tempo_bpm
    parts = []
    for note in melody.notes:
        duration_s = note.beats * seconds_per_beat
        if note.pitch == REST:
            parts.append(np.zeros(int(round(duration_s * sample_rate))))
        else:
            parts.append(render_tone(note_to_hz(note.pitch), duration_s, sample_rate))
    # Half a second of tail so the last note isn't cut off by players.
    parts.append(np.zeros(sample_rate // 2))
    audio = np.concatenate(parts)

    peak = np.max(np.abs(audio))
    return audio / peak * PEAK_LEVEL if peak > 0 else audio


def write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, (audio * 32767).astype(np.int16))
    return path


def synthesize_to_wav(melody: Melody, path: Path, sample_rate: int) -> Path:
    return write_wav(render_melody(melody, sample_rate), sample_rate, path)
