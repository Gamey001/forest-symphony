"""The `generate_melody` tool: its schema for the AI, strict argument checks, and execution."""

import ast
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.music import (
    ALLOWED_BEATS,
    MAX_MIDI,
    MIN_MIDI,
    REST,
    midi_to_note,
    note_to_midi,
    snap_beats,
)
from app.schemas import Melody, NoteEvent
from app.synth.synthesizer import synthesize_to_wav

TOOL_NAME = "generate_melody"
MIN_EVENTS = 4
MAX_EVENTS = 48
MIN_PITCHED = 3
MAX_SPAN_SEMITONES = 24  # wild octave leaps sound broken, not musical

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Render the final melody to a playable audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Note names from C3 to C7 such as "C5", "F#4", or "rest".',
                },
                "beats": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": f"Length of each note in beats, one per note. Allowed: {list(ALLOWED_BEATS)}.",
                },
                "tempo_bpm": {"type": "integer", "description": "Tempo, 60 to 140 beats per minute."},
                "title": {"type": "string", "description": "A short, evocative title (max 6 words)."},
            },
            "required": ["notes", "beats", "tempo_bpm", "title"],
        },
    },
}


class ToolArgumentError(ValueError):
    """Raised with a message the AI can read and correct on its next attempt."""


def _as_list(value: Any, name: str) -> list:
    """Small models often send lists as text, e.g. "['C5', 'E5']". Accept that, safely."""
    if isinstance(value, str):
        for parse in (json.loads, ast.literal_eval):  # literal_eval only reads data, never runs code
            try:
                value = parse(value)
                break
            except (ValueError, SyntaxError):
                continue
    if not isinstance(value, list):
        raise ToolArgumentError(f"'{name}' must be a list")
    return value


def _normalize_pitch(raw: Any) -> str:
    text = str(raw).strip()
    if text.lower() == REST:
        return REST
    try:
        midi = note_to_midi(text)
    except ValueError as exc:
        raise ToolArgumentError(str(exc)) from exc
    if not MIN_MIDI <= midi <= MAX_MIDI:
        raise ToolArgumentError(f"'{text}' is outside the playable range C3 to C7")
    return midi_to_note(midi)


def parse_tool_arguments(args: dict[str, Any]) -> Melody:
    """Check the AI's arguments and turn them into a Melody, or explain what is wrong."""
    notes, beats = _as_list(args.get("notes"), "notes"), _as_list(args.get("beats"), "beats")
    if len(notes) != len(beats):
        raise ToolArgumentError(f"'notes' has {len(notes)} items but 'beats' has {len(beats)}")
    if not MIN_EVENTS <= len(notes) <= MAX_EVENTS:
        raise ToolArgumentError(f"use between {MIN_EVENTS} and {MAX_EVENTS} notes, not {len(notes)}")

    events = []
    for pitch, beat in zip(notes, beats):
        try:
            beat_value = float(beat)
        except (TypeError, ValueError) as exc:
            raise ToolArgumentError(f"beat value {beat!r} is not a number") from exc
        if beat_value <= 0:
            raise ToolArgumentError("beat values must be positive")
        events.append(NoteEvent(pitch=_normalize_pitch(pitch), beats=snap_beats(beat_value)))

    pitched = [note_to_midi(e.pitch) for e in events if e.pitch != REST]
    if len(pitched) < MIN_PITCHED:
        raise ToolArgumentError(f"the melody needs at least {MIN_PITCHED} real notes")
    if max(pitched) - min(pitched) > MAX_SPAN_SEMITONES:
        raise ToolArgumentError(
            f"notes span {midi_to_note(min(pitched))} to {midi_to_note(max(pitched))}; "
            "keep the whole melody within two octaves, close to the draft's register"
        )

    try:
        return Melody(
            notes=events,
            tempo_bpm=int(args.get("tempo_bpm", 0)),
            title=str(args.get("title") or "Forest Symphony")[:60],
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise ToolArgumentError("tempo_bpm must be a whole number between 40 and 200") from exc


def run_generate_melody(melody: Melody, output_path: Path, sample_rate: int) -> Path:
    return synthesize_to_wav(melody, output_path, sample_rate)
