# The Forest Symphony

Record a bird, get a melody you can keep. The app cleans up the recording, identifies the bird,
follows the notes it sings, has a local AI arrange them, and renders a downloadable WAV file.
**Everything runs on your own computer, and no data leaves it.** After the one-time setup, it works with Wi-Fi off.

## Quick start

Requirements: Python 3.11–3.13 (3.12 recommended) and [Ollama](https://ollama.com).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# One-time downloads (need internet once, then everything is offline)
ollama pull llama3.1
.venv/bin/python -c "import birdnet; birdnet.load('acoustic', '2.4', 'tf', library='litert')"

# Run
ollama serve                        # in its own terminal
.venv/bin/uvicorn app.main:app --port 8000
```

Then open http://localhost:8000. The microphone works there because browsers trust `localhost`.

If Ollama isn't running, the app still works: it uses a direct transcription of the bird's song
and the page says so.

Run the tests with `.venv/bin/python -m pytest`. Add `-m "not integration"` to skip the tests
that run the real BirdNET model.

## How it works

```
Browser ──audio──► 1. Cleanup ──► 2. Identify (BirdNET) ──► 3. Pitch tracker ──► 4. Draft melody
                                        │                                               │
                                  not a bird? ──► friendly error                         ▼
Browser ◄──card + WAV── 7. Synthesizer ◄── 6. Polish ◄── 5. AI composer (llama3.1 via Ollama)
```

| Step | What happens | Where |
|---|---|---|
| 1. Cleanup | Cuts low wind rumble, turns down steady background hiss, and trims dead air. Rejects silent clips. | `app/audio/cleanup.py` |
| 2. Identify | BirdNET (a bird-specific model from Cornell) names the species. If the best match is below 50% confidence, or the sound is people, engines or noise, the app says no bird was heard. | `app/listener/species.py` |
| 3. Pitch | Finds the loudest frequency 100 times per second. It checks for a lower "true" note under loud overtones, which owls need. | `app/listener/pitch.py` |
| 4. Draft | Groups pitch readings into notes, moves them by whole octaves into a comfortable range, slows the rhythm down, and fits the notes into the best-matching key. This works with no AI at all. | `app/composer/draft.py` |
| 5. AI composer | The local model reads the draft as text and replies with a `generate_melody` tool call. Its answer is strictly checked: real note names, at most two octaves, valid rhythms. If the answer is wrong, the model is told why and gets one retry. If Ollama is down, the draft is used. | `app/composer/agent.py`, `tool.py` |
| 6. Polish | Keeps every note in the key and makes the tune end on its home note. The note shown to the user describes what the AI actually changed, measured in code rather than claimed by the model. | `app/composer/polish.py` |
| 7. Synthesizer | Plain sine-wave maths (numpy) with soft overtones and fades writes the WAV file. | `app/synth/synthesizer.py` |

Species facts on the result card come from a curated file (`data/species_facts.json`), never from the AI.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Is the AI composer ready, offline, or missing its model? |
| GET | `/api/samples` | Demo recordings |
| GET | `/api/samples/{id}/audio` | Play a demo recording |
| POST | `/api/symphony/sample/{id}` | Compose from a demo recording |
| POST | `/api/symphony` | Compose from an uploaded file (WAV, FLAC, OGG, MP3; max 15 MB, first 30 s used) |
| GET | `/api/outputs/{file}.wav` | Play or download a generated melody |

Errors come back as `{"error": "<code>", "message": "<friendly text>"}`. The codes are
`invalid_audio`, `audio_too_quiet`, `no_bird_detected`, `no_melody_found` and `sample_not_found`.

Settings (thresholds, model name, paths) live in `app/config.py` and can be overridden with
`FOREST_*` environment variables, for example `FOREST_OLLAMA_MODEL=qwen2.5:7b`.

## Demo recordings

| Sample | Expected result | Credit |
|---|---|---|
| Morning Robin Chirp | American Robin (~79%) | Jonathon Jongsma, [xeno-canto XC130246](https://commons.wikimedia.org/wiki/File:Turdus_migratorius_-_American_Robin_XC130246.ogg), CC BY-SA 3.0 |
| Deep Woods Owl | Tawny Owl (~93%) | Benoît Van Hecke, [xeno-canto XC603209](https://commons.wikimedia.org/wiki/File:Strix_aluco_-_Tawny_Owl_XC603209.mp3), CC BY-SA 4.0 |
| Windy Ridge (no bird) | "We couldn't hear a bird" | Generated, public domain |

The bird clips are 12-second excerpts of the originals and keep their original licences.
Rebuild them with `.venv/bin/python scripts/seed_samples.py`.

## Known limits

- **The AI composer is light-touch.** With llama3.1 (8B), the model usually keeps the draft notes
  and mainly picks the title. The draft and polish steps do most of the musical work. A larger
  local model, or a tighter prompt, would give the AI more to do.
- **"Not a bird" covers BirdNET's people, machine and noise labels.** BirdNET also knows a few
  frogs, insects and mammals; those can still pass as a "bird".
- **Busy recordings with several birds** produce one melody from whichever sound is loudest.
- **WAV only.** MP3 export would need `ffmpeg` installed.
- Generated files pile up in `outputs/`; delete the folder whenever you like.
- BirdNET's model is licensed CC BY-NC-SA 4.0 (non-commercial).
