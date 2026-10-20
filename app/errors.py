"""User-facing pipeline errors. Each carries a stable code and a friendly message for the UI."""


class SymphonyError(Exception):
    code = "symphony_error"
    status_code = 422

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidAudioError(SymphonyError):
    code = "invalid_audio"
    status_code = 400


class AudioTooQuietError(SymphonyError):
    code = "audio_too_quiet"


class NoBirdDetectedError(SymphonyError):
    code = "no_bird_detected"

    def __init__(self, message: str, heard: str | None = None):
        super().__init__(message)
        self.heard = heard


class NoMelodyFoundError(SymphonyError):
    code = "no_melody_found"


class SampleNotFoundError(SymphonyError):
    code = "sample_not_found"
    status_code = 404
