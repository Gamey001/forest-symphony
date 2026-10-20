import io

import numpy as np
import pytest
import soundfile as sf

from app.audio.cleanup import clean_audio, remove_wind, trim_silence
from app.audio.loader import AudioClip, load_audio
from app.errors import AudioTooQuietError, InvalidAudioError

SR = 32_000


def _tone(hz: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _band_energy(samples: np.ndarray, lo: float, hi: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / SR)
    return spectrum[(freqs >= lo) & (freqs < hi)].sum()


def test_highpass_removes_wind_but_keeps_birdsong():
    wind, chirp = _tone(60, 1.0), _tone(3000, 1.0)
    filtered = remove_wind(AudioClip(wind + chirp, SR), cutoff_hz=250).samples
    assert _band_energy(filtered, 40, 80) < 0.001 * _band_energy(wind, 40, 80)
    assert _band_energy(filtered, 2900, 3100) > 0.9 * _band_energy(chirp, 2900, 3100)


def test_trim_silence_drops_dead_air():
    samples = np.concatenate([np.zeros(SR * 2), _tone(3000, 0.5), np.zeros(SR * 2)])
    trimmed = trim_silence(AudioClip(samples.astype(np.float32), SR), threshold_db=-35)
    assert 0.5 <= trimmed.duration_s < 1.0


def test_silent_recording_is_rejected():
    with pytest.raises(AudioTooQuietError):
        trim_silence(AudioClip(np.zeros(SR * 2, dtype=np.float32), SR), threshold_db=-35)


def test_full_cleanup_returns_audio_for_a_chirp():
    rng = np.random.default_rng(0)
    noisy = np.concatenate([np.zeros(SR), _tone(2500, 1.0), np.zeros(SR)])
    noisy = noisy + 0.01 * rng.standard_normal(len(noisy))
    cleaned = clean_audio(AudioClip(noisy.astype(np.float32), SR), 250, -35)
    assert cleaned.duration_s < 3.0


def test_loader_mixes_to_mono_and_caps_length():
    stereo = np.stack([_tone(440, 3.0), _tone(440, 3.0)], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, SR, format="WAV")
    clip = load_audio(buf.getvalue(), max_seconds=2.0)
    assert clip.samples.ndim == 1
    assert clip.duration_s == pytest.approx(2.0)


def test_loader_rejects_non_audio():
    with pytest.raises(InvalidAudioError):
        load_audio(b"definitely not audio", max_seconds=10)
