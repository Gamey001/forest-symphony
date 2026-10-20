"""Central settings. Every value can be overridden with a FOREST_* environment variable."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOREST_", env_file=".env", extra="ignore")

    data_dir: Path = ROOT_DIR / "data"
    samples_dir: Path = ROOT_DIR / "data" / "samples"
    facts_file: Path = ROOT_DIR / "data" / "species_facts.json"
    output_dir: Path = ROOT_DIR / "outputs"
    frontend_dir: Path = ROOT_DIR / "frontend"

    # Input limits keep processing fast on a laptop.
    max_upload_bytes: int = 15 * 1024 * 1024
    max_clip_seconds: float = 30.0

    # Cleanup: wind lives below ~1 kHz, birdsong mostly above it.
    highpass_hz: float = 250.0
    silence_threshold_db: float = -35.0

    # BirdNET: random noise can score ~0.45, so we require more than that.
    min_bird_confidence: float = 0.5

    # Composer (local Ollama).
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_s: float = 60.0
    composer_max_attempts: int = 2

    # Synthesizer.
    synth_sample_rate: int = 44_100


@lru_cache
def get_settings() -> Settings:
    return Settings()
