"""The built-in demo recordings, so live demos never depend on a quiet room or a cooperative bird."""

import json
from pathlib import Path

from pydantic import BaseModel

from app.errors import SampleNotFoundError


class Sample(BaseModel):
    id: str
    title: str
    description: str
    file: str
    credit: str
    license: str
    source_page: str | None = None


class SampleLibrary:
    def __init__(self, samples_dir: Path):
        self._dir = samples_dir
        catalog = json.loads((samples_dir / "samples.json").read_text(encoding="utf-8"))
        self._samples = {entry["id"]: Sample(**entry) for entry in catalog}

    def list(self) -> list[Sample]:
        return list(self._samples.values())

    def get(self, sample_id: str) -> Sample:
        if sample_id not in self._samples:
            raise SampleNotFoundError(f"There is no sample called '{sample_id}'.")
        return self._samples[sample_id]

    def path(self, sample_id: str) -> Path:
        return self._dir / self.get(sample_id).file
