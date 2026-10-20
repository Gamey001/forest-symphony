"""End-to-end checks on the seeded samples with the real BirdNET model (AI composer offline)."""

import pytest

from app.composer.agent import fallback_composition
from app.config import Settings
from app.errors import NoBirdDetectedError
from app.facts import FactBook
from app.pipeline import SymphonyPipeline

pytestmark = pytest.mark.integration


class OfflineComposer:
    def compose(self, draft, species):
        return fallback_composition(draft, "AI composer offline")


@pytest.fixture(scope="module")
def detector():
    from app.listener.species import BirdNetDetector

    try:
        return BirdNetDetector(min_confidence=0.5)
    except Exception as exc:  # model not downloaded and no internet
        pytest.skip(f"BirdNET unavailable: {exc}")


@pytest.fixture
def pipeline(detector, tmp_path):
    settings = Settings(output_dir=tmp_path)
    return SymphonyPipeline(settings, detector, OfflineComposer(), FactBook(settings.facts_file))


@pytest.mark.parametrize(
    ("sample", "species"),
    [("morning-robin", "American Robin"), ("deep-woods-owl", "Tawny Owl")],
)
def test_seeded_bird_samples_are_identified(pipeline, sample, species):
    result = pipeline.run(open(f"data/samples/{sample}.flac", "rb").read())
    assert result.species.common_name == species
    assert result.facts is not None


def test_wind_sample_is_rejected_as_not_a_bird(pipeline):
    with pytest.raises(NoBirdDetectedError):
        pipeline.run(open("data/samples/windy-ridge.flac", "rb").read())
