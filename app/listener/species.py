"""Identifies the bird with BirdNET, and refuses politely when the sound isn't a bird."""

import logging
import threading
from collections import defaultdict
from typing import Protocol

from app.audio.loader import AudioClip
from app.errors import NoBirdDetectedError
from app.schemas import SpeciesDetection

logger = logging.getLogger(__name__)

# BirdNET labels that are not birds (people, machines, background). Its label list also holds
# a few frogs, insects and mammals; those are treated as "not a bird" only if listed here.
NON_BIRD_LABELS = frozenset(
    {
        "Dog_Dog",
        "Engine_Engine",
        "Environmental_Environmental",
        "Fireworks_Fireworks",
        "Gun_Gun",
        "Human non-vocal_Human non-vocal",
        "Human vocal_Human vocal",
        "Human whistle_Human whistle",
        "Noise_Noise",
        "Power tools_Power tools",
        "Siren_Siren",
        "Gryllus assimilis_Gryllus assimilis",
        "Miogryllus saussurei_Miogryllus saussurei",
    }
)
# Below this score BirdNET is guessing; we don't even mention what it "heard".
HINT_CONFIDENCE = 0.25


class SpeciesDetector(Protocol):
    def detect(self, clip: AudioClip) -> SpeciesDetection: ...


def _split_label(label: str) -> tuple[str, str]:
    scientific, _, common = label.partition("_")
    return scientific, common or scientific


def pick_bird(scores: dict[str, float], min_confidence: float) -> SpeciesDetection:
    """Choose the most confident bird, or raise NoBirdDetectedError with a helpful hint."""
    birds = {label: s for label, s in scores.items() if label not in NON_BIRD_LABELS}
    if birds:
        label, confidence = max(birds.items(), key=lambda item: item[1])
        if confidence >= min_confidence:
            scientific, common = _split_label(label)
            return SpeciesDetection(
                scientific_name=scientific, common_name=common, confidence=round(confidence, 3)
            )

    others = {label: s for label, s in scores.items() if label in NON_BIRD_LABELS}
    heard = None
    if others:
        label, confidence = max(others.items(), key=lambda item: item[1])
        if confidence >= HINT_CONFIDENCE:
            heard = _split_label(label)[1].lower()

    message = "We couldn't hear a bird in this recording."
    message += f" It sounds more like {heard}." if heard else ""
    message += " Try again closer to the bird, or pick one of the sample recordings."
    raise NoBirdDetectedError(message, heard=heard)


class BirdNetDetector:
    """Wraps the offline BirdNET v2.4 model. Load once at startup; it takes a few seconds."""

    def __init__(self, min_confidence: float):
        import birdnet  # heavy import, kept local so tests without the model stay fast

        self._min_confidence = min_confidence
        self._model = birdnet.load("acoustic", "2.4", "tf", library="litert")
        self._lock = threading.Lock()

    def detect(self, clip: AudioClip) -> SpeciesDetection:
        with self._lock:
            result = self._model.predict_arrays(
                (clip.samples, clip.sample_rate), top_k=3, default_confidence_threshold=0.1
            )
        # BirdNET scores each 3-second window; keep each label's best score across windows.
        scores: dict[str, float] = defaultdict(float)
        for row in result.to_structured_array():
            label = str(row["species_name"])
            scores[label] = max(scores[label], float(row["confidence"]))
        logger.info("BirdNET scores: %s", dict(sorted(scores.items(), key=lambda i: -i[1])[:5]))
        return pick_bird(scores, self._min_confidence)
