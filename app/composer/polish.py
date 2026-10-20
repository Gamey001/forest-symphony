"""Final, deterministic touches applied to every melody, whether the AI or the draft wrote it."""

from app.composer.draft import Draft
from app.music import MAX_MIDI, MIN_MIDI, REST, midi_to_note, note_to_midi, snap_to_key
from app.schemas import Melody, NoteEvent

FINAL_NOTE_BEATS = 2.0


def polish(melody: Melody, tonic: int, mode: str) -> Melody:
    """Keep every note in the key and make sure the tune comes to rest on the home note."""
    notes = [
        n
        if n.pitch == REST
        else NoteEvent(
            pitch=midi_to_note(snap_to_key(note_to_midi(n.pitch), tonic, mode)), beats=n.beats
        )
        for n in melody.notes
    ]
    while notes and notes[-1].pitch == REST:
        notes.pop()

    last = note_to_midi(notes[-1].pitch)
    if last % 12 == tonic:
        notes[-1] = NoteEvent(pitch=notes[-1].pitch, beats=max(notes[-1].beats, FINAL_NOTE_BEATS))
    else:
        # The home note closest to where the tune ended, so the resolution feels natural.
        homes = [last + d for d in range(-6, 7) if (last + d) % 12 == tonic]
        home = min(homes, key=lambda m: abs(m - last))
        home += 12 if home < MIN_MIDI else -12 if home > MAX_MIDI else 0
        notes.append(NoteEvent(pitch=midi_to_note(home), beats=FINAL_NOTE_BEATS))
    return melody.model_copy(update={"notes": notes})


def describe(draft: Draft, composed: Melody, source: str, fallback_reason: str | None) -> str:
    """A factual note for the user, measured from the melodies rather than claimed by the AI.

    `composed` is the melody before polishing, so only the AI's own edits are counted.
    """
    note_count = sum(n.pitch != REST for n in composed.notes)
    summary = f"{draft.key}, {composed.tempo_bpm} BPM, {note_count} notes."
    if source == "fallback":
        return f"{summary} A direct transcription of the bird's song ({fallback_reason})."

    before = [n.pitch for n in draft.melody.notes if n.pitch != REST]
    after = [n.pitch for n in composed.notes if n.pitch != REST]
    changed = sum(a != b for a, b in zip(before, after)) + abs(len(before) - len(after))
    rhythm_changed = [n.beats for n in draft.melody.notes] != [n.beats for n in composed.notes]
    if changed:
        return f"{summary} The AI composer reshaped {changed} of the bird's {len(before)} notes."
    if rhythm_changed:
        return f"{summary} The AI composer kept the bird's notes and reshaped the rhythm."
    return f"{summary} The AI composer kept the bird's melody exactly as sung."
