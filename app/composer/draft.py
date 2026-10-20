"""Deterministic transcription: pitch contour -> a rough but playable melody.

This is the safety net. It always works without AI, and it is also the starting point the AI
composer refines.
"""

from dataclasses import dataclass

import numpy as np

from app.errors import NoMelodyFoundError
from app.listener.pitch import PitchContour
from app.music import (
    MAX_MIDI,
    MIN_MIDI,
    REST,
    detect_key,
    hz_to_midi,
    key_name,
    midi_to_note,
    snap_beats,
    snap_to_key,
)
from app.schemas import Melody, NoteEvent

TARGET_CENTER_MIDI = 72  # C5, a bright but comfortable register
SPLIT_SEMITONES = 1.5  # a pitch jump larger than this starts a new note
MAX_GAP_HOPS = 2  # a break longer than this many slices ends a note
MIN_NOTE_HOPS = 3  # notes shorter than ~30 ms are clicks, not notes
MAX_NOTES = 24
MAX_REST_BEATS = 2.0
DRAFT_TEMPO_BPM = 96
MIN_NOTES = 3


@dataclass(frozen=True)
class SungNote:
    start_s: float
    end_s: float
    midi: float


@dataclass(frozen=True)
class Draft:
    melody: Melody
    octave_shift: int  # how many octaves the bird was moved down (negative) or up
    pitch_range_hz: tuple[float, float]
    tonic: int  # home note as a pitch class, 0 = C
    mode: str  # "major" or "minor"

    @property
    def key(self) -> str:
        return key_name(self.tonic, self.mode)


def segment_notes(contour: PitchContour) -> list[SungNote]:
    """Group consecutive pitch readings into notes."""
    midis = hz_to_midi(contour.hz)
    notes: list[SungNote] = []
    current: list[int] = []

    def close() -> None:
        if len(current) >= MIN_NOTE_HOPS:
            notes.append(
                SungNote(
                    start_s=float(contour.times_s[current[0]]),
                    end_s=float(contour.times_s[current[-1]] + contour.hop_s),
                    midi=float(np.median(midis[current])),
                )
            )

    for i in range(len(midis)):
        if current:
            gap = contour.times_s[i] - contour.times_s[current[-1]]
            jump = abs(midis[i] - np.median(midis[current]))
            if gap > MAX_GAP_HOPS * contour.hop_s + 1e-6 or jump > SPLIT_SEMITONES:
                close()
                current = []
        current.append(i)
    close()
    return notes


def _fold_into_range(midi: int) -> int:
    while midi < MIN_MIDI:
        midi += 12
    while midi > MAX_MIDI:
        midi -= 12
    return midi


def build_draft(contour: PitchContour, title: str) -> Draft:
    sung = segment_notes(contour)[:MAX_NOTES]
    if len(sung) < MIN_NOTES:
        raise NoMelodyFoundError(
            "We heard the bird but couldn't pick out clear notes. Try a recording with a "
            "clearer, closer song."
        )

    # Shift by whole octaves so the tune keeps its shape but lands in a pleasant register.
    octave_shift = int(round((TARGET_CENTER_MIDI - np.median([n.midi for n in sung])) / 12))

    # Birds sing fast. Scale time so the bird's typical note becomes an eighth note (0.5 beat).
    seconds_per_beat = float(np.median([n.end_s - n.start_s for n in sung])) / 0.5

    pitched = [
        (
            _fold_into_range(int(round(note.midi)) + 12 * octave_shift),
            snap_beats((note.end_s - note.start_s) / seconds_per_beat),
        )
        for note in sung
    ]
    # Birds don't sing in keys; nudging stray notes into the best-fitting key makes it musical.
    tonic, mode = detect_key(pitched)

    events: list[NoteEvent] = []
    for i, (note, (midi, beats)) in enumerate(zip(sung, pitched)):
        if i > 0:
            gap_beats = (note.start_s - sung[i - 1].end_s) / seconds_per_beat
            if gap_beats >= 0.375:
                events.append(NoteEvent(pitch=REST, beats=snap_beats(min(gap_beats, MAX_REST_BEATS))))
        midi = _fold_into_range(snap_to_key(midi, tonic, mode))
        events.append(NoteEvent(pitch=midi_to_note(midi), beats=beats))

    return Draft(
        melody=Melody(notes=events, tempo_bpm=DRAFT_TEMPO_BPM, title=title),
        octave_shift=octave_shift,
        pitch_range_hz=(round(float(contour.hz.min()), 1), round(float(contour.hz.max()), 1)),
        tonic=tonic,
        mode=mode,
    )
