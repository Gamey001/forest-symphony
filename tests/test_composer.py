from types import SimpleNamespace

import numpy as np
import pytest

from app.composer.agent import OllamaComposer
from app.composer.draft import build_draft, segment_notes
from app.composer.tool import ToolArgumentError, parse_tool_arguments
from app.errors import NoMelodyFoundError
from app.listener.pitch import PitchContour
from app.music import REST, note_to_midi
from app.schemas import SpeciesDetection

HOP = 0.01
ROBIN = SpeciesDetection(scientific_name="Turdus migratorius", common_name="American Robin", confidence=0.8)


def _contour(phrases: list[tuple[float, float, float]]) -> PitchContour:
    """Build a contour from (start_s, duration_s, hz) phrases."""
    times, hz = [], []
    for start, duration, freq in phrases:
        steps = np.arange(start, start + duration, HOP)
        times.extend(steps)
        hz.extend([freq] * len(steps))
    return PitchContour(times_s=np.array(times), hz=np.array(hz), hop_s=HOP)


BIRD_PHRASES = [(0.0, 0.1, 2637), (0.15, 0.1, 3136), (0.3, 0.1, 2637), (0.6, 0.2, 2093)]


def test_segment_notes_splits_on_pitch_jumps_and_gaps():
    notes = segment_notes(_contour(BIRD_PHRASES))
    assert len(notes) == 4
    assert notes[0].midi == pytest.approx(100, abs=0.1)  # E7


def test_draft_moves_bird_into_a_comfortable_register():
    draft = build_draft(_contour(BIRD_PHRASES), "Test")
    pitched = [n.pitch for n in draft.melody.notes if n.pitch != REST]
    assert pitched == ["E5", "G5", "E5", "C5"]
    assert draft.octave_shift == -2
    assert any(n.pitch == REST for n in draft.melody.notes)  # the longer pause becomes a rest


def test_draft_needs_enough_notes():
    with pytest.raises(NoMelodyFoundError):
        build_draft(_contour([(0.0, 0.1, 3000)]), "Test")


VALID_ARGS = {
    "notes": ["C5", "E5", "G5", "rest", "C5"],
    "beats": [0.5, 0.5, 1, 0.5, 2],
    "tempo_bpm": 90,
    "title": "Dawn",
}


def test_valid_tool_arguments_become_a_melody():
    melody = parse_tool_arguments(VALID_ARGS)
    assert [n.pitch for n in melody.notes] == ["C5", "E5", "G5", "rest", "C5"]
    assert melody.tempo_bpm == 90


def test_lists_sent_as_text_are_accepted():
    melody = parse_tool_arguments(
        VALID_ARGS | {"notes": "['C5', 'E5', 'G5', 'rest', 'C5']", "beats": "[0.5, 0.5, 1, 0.5, 2]"}
    )
    assert len(melody.notes) == 5


def test_flats_are_normalized_and_odd_beats_snapped():
    melody = parse_tool_arguments(VALID_ARGS | {"notes": ["Bb4", "E5", "G5", "rest", "C5"], "beats": [0.6, 0.5, 1, 0.5, 2]})
    assert melody.notes[0].pitch == "A#4"
    assert melody.notes[0].beats == 0.5


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"beats": [1, 1]}, "'beats' has 2"),
        ({"notes": ["C5", "E5", "H9", "rest", "C5"]}, "not a note name"),
        ({"notes": ["C1", "E5", "G5", "rest", "C5"]}, "outside the playable range"),
        ({"tempo_bpm": 999}, "tempo_bpm"),
        ({"notes": ["rest", "rest", "rest", "rest", "C5"]}, "at least 3"),
    ],
)
def test_bad_tool_arguments_are_explained(override, fragment):
    with pytest.raises(ToolArgumentError, match=fragment):
        parse_tool_arguments(VALID_ARGS | override)


def _response(arguments: dict | None):
    calls = None
    if arguments is not None:
        calls = [SimpleNamespace(function=SimpleNamespace(name="generate_melody", arguments=arguments))]
    return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs | {"messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def draft():
    return build_draft(_contour(BIRD_PHRASES), "Test")


def test_composer_uses_valid_ai_melody(draft):
    composition = OllamaComposer(FakeClient([_response(VALID_ARGS)]), "m").compose(draft, ROBIN)
    assert composition.source == "ai"
    assert composition.melody.title == "Dawn"


def test_composer_retries_with_error_feedback(draft):
    client = FakeClient([_response(VALID_ARGS | {"beats": [1]}), _response(VALID_ARGS)])
    composition = OllamaComposer(client, "m", max_attempts=2).compose(draft, ROBIN)
    assert composition.source == "ai"
    feedback = client.calls[1]["messages"][-1]
    assert feedback["role"] == "tool" and "Error" in feedback["content"]


def test_composer_falls_back_when_ollama_is_offline(draft):
    composition = OllamaComposer(FakeClient([ConnectionError("refused")]), "m").compose(draft, ROBIN)
    assert composition.source == "fallback"
    assert composition.melody == draft.melody


def test_composer_falls_back_after_repeated_bad_answers(draft):
    client = FakeClient([_response(None), _response({"notes": "nope"})])
    composition = OllamaComposer(client, "m", max_attempts=2).compose(draft, ROBIN)
    assert composition.source == "fallback"


def test_draft_notes_stay_in_playable_range(draft):
    for n in draft.melody.notes:
        if n.pitch != REST:
            assert 48 <= note_to_midi(n.pitch) <= 96


def test_detect_key_and_snap_stray_notes():
    from app.music import detect_key, key_name, snap_to_key

    c_major_tune = [(60, 1.0), (64, 1.0), (67, 1.0), (72, 2.0), (61, 0.25)]  # C# is the stray
    tonic, mode = detect_key(c_major_tune)
    assert key_name(tonic, mode) == "C major"
    assert snap_to_key(61, tonic, mode) == 60  # C# -> C
    assert snap_to_key(64, tonic, mode) == 64


def test_draft_reports_its_key(draft):
    assert draft.key == "C major"


def test_melody_spanning_more_than_two_octaves_is_rejected():
    with pytest.raises(ToolArgumentError, match="two octaves"):
        parse_tool_arguments(VALID_ARGS | {"notes": ["C3", "E5", "G5", "rest", "C6"]})


def test_polish_ends_on_home_note_and_keeps_notes_in_key():
    from app.composer.polish import polish
    from app.schemas import Melody, NoteEvent

    melody = Melody(
        notes=[NoteEvent(pitch=p, beats=0.5) for p in ["C5", "C#5", "G5"]] + [NoteEvent(pitch="rest", beats=1)],
        tempo_bpm=96,
    )
    polished = polish(melody, tonic=0, mode="major")
    assert [n.pitch for n in polished.notes] == ["C5", "C5", "G5", "C6"]
    assert polished.notes[-1].beats == 2.0


def test_description_is_measured_not_claimed(draft):
    from app.composer.polish import describe

    assert "exactly as sung" in describe(draft, draft.melody, "ai", None)
    assert "AI composer offline" in describe(draft, draft.melody, "fallback", "AI composer offline")
