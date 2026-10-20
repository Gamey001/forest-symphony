"""API tests with a stand-in bird detector and composer, so they run fast and without models."""

import pytest
from fastapi.testclient import TestClient

from app.composer.agent import fallback_composition
from app.config import Settings
from app.errors import NoBirdDetectedError
from app.facts import FactBook
from app.main import Services, create_app
from app.pipeline import SymphonyPipeline
from app.samples import SampleLibrary
from app.schemas import SpeciesDetection


class StubDetector:
    """Says "robin" for anything except the wind sample's length (8 s)."""

    def detect(self, clip):
        if abs(clip.duration_s - 8.0) < 0.1:
            raise NoBirdDetectedError("We couldn't hear a bird in this recording.")
        return SpeciesDetection(scientific_name="Turdus migratorius", common_name="American Robin", confidence=0.79)


class OfflineComposer:
    def compose(self, draft, species):
        return fallback_composition(draft, "AI composer offline")


@pytest.fixture
def client(tmp_path):
    settings = Settings(output_dir=tmp_path / "outputs", ollama_host="http://127.0.0.1:9")
    services = Services(
        settings=settings,
        pipeline=SymphonyPipeline(settings, StubDetector(), OfflineComposer(), FactBook(settings.facts_file)),
        samples=SampleLibrary(settings.samples_dir),
    )
    with TestClient(create_app(services)) as test_client:
        yield test_client


def test_health_reports_composer_offline(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["composer"] == "offline"


def test_lists_samples_and_serves_their_audio(client):
    samples = client.get("/api/samples").json()
    assert "morning-robin" in [s["id"] for s in samples]
    assert client.get("/api/samples/morning-robin/audio").status_code == 200


def test_sample_produces_downloadable_melody(client):
    response = client.post("/api/symphony/sample/morning-robin")
    assert response.status_code == 200
    body = response.json()
    assert body["species"]["common_name"] == "American Robin"
    assert body["facts"]["habitat"]
    assert body["composer"] == "fallback"
    assert len(body["melody"]["notes"]) >= 3

    audio = client.get(body["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"


def test_upload_works_like_a_sample(client):
    data = open("data/samples/deep-woods-owl.flac", "rb").read()
    response = client.post("/api/symphony", files={"file": ("owl.flac", data, "audio/flac")})
    assert response.status_code == 200


def test_no_bird_returns_friendly_error(client):
    response = client.post("/api/symphony/sample/windy-ridge")
    assert response.status_code == 422
    assert response.json()["error"] == "no_bird_detected"


def test_garbage_upload_is_rejected(client):
    response = client.post("/api/symphony", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_audio"


def test_unknown_sample_is_404(client):
    assert client.post("/api/symphony/sample/nope").status_code == 404


@pytest.mark.parametrize("name", ["..%2F..%2Fapp%2Fconfig.py", "missing.wav", "UPPER.wav"])
def test_output_route_blocks_bad_names(client, name):
    assert client.get(f"/api/outputs/{name}").status_code == 404
