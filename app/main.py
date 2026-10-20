"""FastAPI app: serves the web page and the JSON API. Everything runs on this machine.

Run with: uvicorn app.main:app
"""

import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import ollama
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.composer.agent import OllamaComposer
from app.config import Settings, get_settings
from app.errors import InvalidAudioError, OutputNotFoundError, SymphonyError
from app.facts import FactBook
from app.listener.species import BirdNetDetector
from app.pipeline import SymphonyPipeline
from app.samples import Sample, SampleLibrary
from app.schemas import SymphonyResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_NAME_RE = re.compile(r"^[a-z0-9-]+\.wav$")  # blocks path tricks like ../../etc


@dataclass
class Services:
    settings: Settings
    pipeline: SymphonyPipeline
    samples: SampleLibrary


def build_services(settings: Settings) -> Services:
    logger.info("Loading BirdNET model...")
    pipeline = SymphonyPipeline(
        settings=settings,
        detector=BirdNetDetector(settings.min_bird_confidence),
        composer=OllamaComposer(
            ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_s),
            settings.ollama_model,
            settings.composer_max_attempts,
        ),
        facts=FactBook(settings.facts_file),
    )
    return Services(settings=settings, pipeline=pipeline, samples=SampleLibrary(settings.samples_dir))


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(get_settings())
        app.state.services.settings.output_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="The Forest Symphony", lifespan=lifespan)

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.exception_handler(SymphonyError)
    async def symphony_error_handler(_: Request, exc: SymphonyError) -> JSONResponse:
        body = {"error": exc.code, "message": exc.message}
        if getattr(exc, "heard", None):
            body["heard"] = exc.heard
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.get("/api/health")
    def health(request: Request) -> dict:
        settings = svc(request).settings
        try:
            tags = httpx.get(f"{settings.ollama_host}/api/tags", timeout=1.5).json()
            names = {m["name"].split(":")[0] for m in tags.get("models", [])}
            composer = "ready" if settings.ollama_model.split(":")[0] in names else "model_missing"
        except (httpx.HTTPError, ValueError):
            composer = "offline"
        return {"status": "ok", "composer": composer, "composer_model": settings.ollama_model}

    @app.get("/api/samples")
    def list_samples(request: Request) -> list[Sample]:
        return svc(request).samples.list()

    @app.get("/api/samples/{sample_id}/audio")
    def sample_audio(sample_id: str, request: Request) -> FileResponse:
        return FileResponse(svc(request).samples.path(sample_id))

    @app.post("/api/symphony/sample/{sample_id}")
    def symphony_from_sample(sample_id: str, request: Request) -> SymphonyResult:
        services = svc(request)
        return services.pipeline.run(services.samples.path(sample_id).read_bytes())

    @app.post("/api/symphony")
    def symphony_from_upload(file: UploadFile, request: Request) -> SymphonyResult:
        services = svc(request)
        limit = services.settings.max_upload_bytes
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise InvalidAudioError(f"That file is too large. Please keep it under {limit // 1_000_000} MB.")
        return services.pipeline.run(data)

    @app.get("/api/outputs/{filename}")
    def output_file(filename: str, request: Request) -> FileResponse:
        path = svc(request).settings.output_dir / filename
        if not OUTPUT_NAME_RE.match(filename) or not path.is_file():
            raise OutputNotFoundError("That melody file doesn't exist (it may have been cleaned up).")
        return FileResponse(path, media_type="audio/wav")

    frontend_dir = (services.settings if services else get_settings()).frontend_dir
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


app = create_app()
