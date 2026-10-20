import numpy as np
import pytest
from scipy.io import wavfile

from app.music import midi_to_note, note_to_hz, note_to_midi, snap_beats
from app.schemas import Melody, NoteEvent
from app.synth.synthesizer import render_melody, render_tone, synthesize_to_wav

SR = 44_100


@pytest.mark.parametrize(
    ("note", "midi"), [("C4", 60), ("A4", 69), ("F#5", 78), ("Bb3", 58), ("C7", 96)]
)
def test_note_round_trip(note, midi):
    assert note_to_midi(note) == midi
    assert note_to_midi(midi_to_note(midi)) == midi


@pytest.mark.parametrize("bad", ["H4", "C", "C#", "rest", "C44", ""])
def test_invalid_note_names_are_rejected(bad):
    with pytest.raises(ValueError):
        note_to_midi(bad)


def test_a4_is_440hz():
    assert note_to_hz("A4") == pytest.approx(440.0)


def test_snap_beats_picks_nearest_allowed_value():
    assert snap_beats(0.3) == 0.25
    assert snap_beats(1.2) == 1.0
    assert snap_beats(9) == 4.0


def test_rendered_tone_has_expected_pitch():
    tone = render_tone(note_to_hz("A4"), 1.0, SR)
    spectrum = np.abs(np.fft.rfft(tone))
    peak_hz = np.argmax(spectrum) * SR / len(tone)
    assert peak_hz == pytest.approx(440, abs=2)


def test_melody_length_matches_tempo_and_beats():
    melody = Melody(
        notes=[NoteEvent(pitch="C5", beats=1), NoteEvent(pitch="rest", beats=1)], tempo_bpm=120
    )
    audio = render_melody(melody, SR)
    assert len(audio) == pytest.approx(SR * (1.0 + 0.5), abs=2)
    assert np.max(np.abs(audio)) <= 1.0


def test_synthesize_writes_playable_wav(tmp_path):
    melody = Melody(notes=[NoteEvent(pitch="E5", beats=0.5)], tempo_bpm=90)
    path = synthesize_to_wav(melody, tmp_path / "out.wav", SR)
    rate, data = wavfile.read(path)
    assert rate == SR
    assert data.dtype == np.int16
    assert data.size > 0
