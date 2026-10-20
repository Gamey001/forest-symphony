"""Offline species facts. Facts come only from this curated file, never from the AI."""

import json
from pathlib import Path

from app.schemas import SpeciesFacts


class FactBook:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._facts = {name: SpeciesFacts(**entry) for name, entry in raw.items()}

    def lookup(self, scientific_name: str) -> SpeciesFacts | None:
        return self._facts.get(scientific_name)
