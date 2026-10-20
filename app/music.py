"""Small music-theory helpers: converting between note names, MIDI numbers and Hz."""

import re

import numpy as np

REST = "rest"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
_NOTE_RE = re.compile(r"^([A-G])([#b]?)(-?\d)$")

# Playable range for the synthesizer: C3 (131 Hz) to C7 (2093 Hz).
MIN_MIDI = 48
MAX_MIDI = 96

# Rhythm values the melody may use, in beats (sixteenth note up to whole note).
ALLOWED_BEATS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69 + 12 * np.log2(np.asarray(hz) / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def note_to_midi(note: str) -> int:
    """Parse "C4", "F#5" or "Bb3". Raises ValueError for anything else."""
    match = _NOTE_RE.match(note.strip())
    if not match:
        raise ValueError(f"'{note}' is not a note name like C4, F#5 or Bb3")
    letter, accidental, octave = match.groups()
    name = _FLATS.get(letter + accidental, letter + accidental)
    return NOTE_NAMES.index(name) + (int(octave) + 1) * 12


def note_to_hz(note: str) -> float:
    return midi_to_hz(note_to_midi(note))


def snap_beats(beats: float) -> float:
    return min(ALLOWED_BEATS, key=lambda allowed: abs(allowed - beats))
