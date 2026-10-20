"""Data shapes passed between pipeline stages and returned by the API."""

from typing import Literal

from pydantic import BaseModel, Field


class SpeciesDetection(BaseModel):
    scientific_name: str
    common_name: str
    confidence: float = Field(ge=0, le=1)


class NoteEvent(BaseModel):
    """One note (e.g. "G5") or a rest, lasting `beats` beats."""

    pitch: str
    beats: float = Field(gt=0)


class Melody(BaseModel):
    notes: list[NoteEvent]
    tempo_bpm: int = Field(ge=40, le=200)
    title: str = "Forest Symphony"

    @property
    def duration_s(self) -> float:
        return sum(n.beats for n in self.notes) * 60.0 / self.tempo_bpm


class SpeciesFacts(BaseModel):
    habitat: str
    song: str
    fun_fact: str


class SymphonyResult(BaseModel):
    species: SpeciesDetection
    facts: SpeciesFacts | None
    melody: Melody
    composer: Literal["ai", "fallback"]
    composer_note: str
    audio_url: str
    download_name: str
    bird_pitch_range_hz: tuple[float, float]
    octave_shift: int
