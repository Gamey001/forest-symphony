"""The AI composer: a local Ollama model arranges the draft melody by calling `generate_melody`.

The model never touches audio. It sees a short text summary (species, draft notes, rhythm) and
replies with a tool call. Its arguments are strictly checked; if they're wrong the model gets
the error and one more try. If Ollama is offline or keeps failing, the draft is used instead,
so the user always gets a melody.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.composer.draft import Draft
from app.composer.tool import TOOL_NAME, TOOL_SCHEMA, ToolArgumentError, parse_tool_arguments
from app.schemas import Melody, SpeciesDetection

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a gentle film-score composer who turns birdsong into short melodies.
You receive a draft transcribed from a real bird recording, already placed in a key.
Arrange it into a pleasant, memorable melody:
- Keep the bird's character: its contour (rises and falls) and rhythmic feel.
- Stay in the given key and in the same octaves as the draft. Do not jump between octaves.
- Smooth out jittery back-and-forth notes; you may repeat a short motif to give it shape.
- End on the key's home note with a long beat value.
- Use 8 to 32 notes, note names like "E5" or "rest", beats from the allowed list.
- Pick a tempo that suits the bird: lively for small songbirds, slow for owls and doves.
Always respond by calling the generate_melody tool exactly once. Do not invent facts about the bird."""


class ChatClient(Protocol):
    def chat(self, *, model: str, messages: list[dict], tools: list[dict], options: dict) -> Any: ...


@dataclass(frozen=True)
class Composition:
    melody: Melody
    source: Literal["ai", "fallback"]
    fallback_reason: str | None = None


def _draft_summary(draft: Draft, species: SpeciesDetection) -> str:
    return json.dumps(
        {
            "species": species.common_name,
            "key": draft.key,
            "draft_notes": [n.pitch for n in draft.melody.notes],
            "draft_beats": [n.beats for n in draft.melody.notes],
            "draft_tempo_bpm": draft.melody.tempo_bpm,
        }
    )


def fallback_composition(draft: Draft, reason: str) -> Composition:
    return Composition(melody=draft.melody, source="fallback", fallback_reason=reason)


class OllamaComposer:
    def __init__(self, client: ChatClient, model: str, max_attempts: int = 2):
        self._client = client
        self._model = model
        self._max_attempts = max_attempts

    def compose(self, draft: Draft, species: SpeciesDetection) -> Composition:
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _draft_summary(draft, species)},
        ]
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat(
                    model=self._model,
                    messages=messages,
                    tools=[TOOL_SCHEMA],
                    options={"temperature": 0.3},
                )
            except Exception as exc:  # connection refused, timeout, model missing...
                logger.warning("Composer unavailable, using draft: %s", exc)
                return fallback_composition(draft, "AI composer offline")

            message = response.message
            messages.append(message)
            calls = [c for c in (message.tool_calls or []) if c.function.name == TOOL_NAME]
            if not calls:
                messages.append({"role": "user", "content": f"Please respond by calling {TOOL_NAME}."})
                continue

            try:
                melody = parse_tool_arguments(dict(calls[0].function.arguments))
            except ToolArgumentError as exc:
                logger.info("Composer attempt %d rejected: %s", attempt, exc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": TOOL_NAME,
                        "content": f"Error: {exc}. Fix it and call {TOOL_NAME} again.",
                    }
                )
                continue
            return Composition(melody=melody, source="ai")

        return fallback_composition(draft, "AI composer gave an unusable answer")
