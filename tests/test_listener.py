import numpy as np
import pytest

from app.audio.loader import AudioClip
from app.errors import NoBirdDetectedError
from app.listener.pitch import track_pitch
from app.listener.species import pick_bird

SR = 48_000


def _harmonic_tone(hz: float, seconds: float, gains: tuple[float, ...]) -> AudioClip:
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(g * np.sin(2 * np.pi * hz * (k + 1) * t) for k, g in enumerate(gains))
    return AudioClip((0.3 * wave).astype(np.float32), SR)


def test_pick_bird_returns_most_confident_bird():
    detection = pick_bird(
        {"Turdus migratorius_American Robin": 0.8, "Strix aluco_Tawny Owl": 0.6}, 0.5
    )
    assert detection.common_name == "American Robin"
    assert detection.scientific_name == "Turdus migratorius"


def test_pick_bird_ignores_non_bird_labels():
    detection = pick_bird({"Human vocal_Human vocal": 0.95, "Strix aluco_Tawny Owl": 0.7}, 0.5)
    assert detection.common_name == "Tawny Owl"


def test_low_confidence_bird_is_rejected():
    with pytest.raises(NoBirdDetectedError) as err:
        pick_bird({"Dryobates minor_Lesser Spotted Woodpecker": 0.45}, 0.5)
    assert err.value.heard is None


def test_rejection_names_what_was_heard_instead():
    with pytest.raises(NoBirdDetectedError) as err:
        pick_bird({"Engine_Engine": 0.8, "Corvus corax_Common Raven": 0.2}, 0.5)
    assert err.value.heard == "engine"
    assert "engine" in err.value.message


def test_nothing_detected_is_rejected():
    with pytest.raises(NoBirdDetectedError):
        pick_bird({}, 0.5)


def test_pitch_tracker_follows_a_pure_whistle():
    contour = track_pitch(_harmonic_tone(3000, 1.0, gains=(1.0,)))
    assert contour.hz.size > 50
    assert np.median(contour.hz) == pytest.approx(3000, rel=0.01)


def test_pitch_tracker_prefers_fundamental_over_louder_overtone():
    # Owl-like: the 2nd and 3rd overtones are louder than the 1 kHz note itself.
    contour = track_pitch(_harmonic_tone(1000, 1.0, gains=(0.3, 1.0, 0.6)))
    assert np.median(contour.hz) == pytest.approx(1000, rel=0.02)


def test_pitch_tracker_ignores_white_noise():
    noise = np.random.default_rng(1).standard_normal(SR).astype(np.float32) * 0.1
    assert track_pitch(AudioClip(noise, SR)).hz.size < 5
