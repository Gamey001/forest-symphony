"""Runs one recording through every stage: clean -> identify -> listen -> compose -> render."""

import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from app.audio.cleanup import clean_audio
from app.audio.loader import load_audio
from app.composer.agent import Composition
from app.composer.draft import Draft, build_draft
from app.composer.polish import describe, polish
from app.composer.tool import run_generate_melody
from app.config import Settings
from app.facts import FactBook
from app.listener.pitch import track_pitch
from app.listener.species import SpeciesDetector
from app.schemas import SpeciesDetection, SymphonyResult


class Composer(Protocol):
    def compose(self, draft: Draft, species: SpeciesDetection) -> Composition: ...


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "bird"


@dataclass
class SymphonyPipeline:
    settings: Settings
    detector: SpeciesDetector
    composer: Composer
    facts: FactBook

    def run(self, audio: bytes) -> SymphonyResult:
        s = self.settings
        clip = load_audio(audio, s.max_clip_seconds)
        cleaned = clean_audio(clip, s.highpass_hz, s.silence_threshold_db)  # raises if silent

        # BirdNET gets the original audio: it was trained on real, noisy field recordings,
        # and our filtering could remove cues it relies on.
        species = self.detector.detect(clip)

        draft = build_draft(track_pitch(cleaned), title=f"{species.common_name} Song")
        composition = self.composer.compose(draft, species)
        melody = polish(composition.melody, draft.tonic, draft.mode)

        filename = f"{_slug(species.common_name)}-{uuid.uuid4().hex[:8]}.wav"
        run_generate_melody(melody, s.output_dir / filename, s.synth_sample_rate)

        return SymphonyResult(
            species=species,
            facts=self.facts.lookup(species.scientific_name),
            melody=melody,
            composer=composition.source,
            composer_note=describe(
                draft, composition.melody, composition.source, composition.fallback_reason
            ),
            audio_url=f"/api/outputs/{filename}",
            download_name=f"{_slug(melody.title)}.wav",
            bird_pitch_range_hz=draft.pitch_range_hz,
            octave_shift=draft.octave_shift,
        )
