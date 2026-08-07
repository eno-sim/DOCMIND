"""Audio ingestion through a lazily loaded local Whisper model.

Whisper is loaded on first use rather than at API import time.  This keeps
``/health`` fast and avoids allocating a large model when a deployment only
handles text documents.  The model is cached per process after first load.
"""

from pathlib import Path
from threading import Lock
from typing import Any


_model: Any = None
_model_name: str | None = None
_model_lock = Lock()


def _get_model(model_name: str = "base") -> Any:
    """Load and cache a Whisper model in a thread-safe manner."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        with _model_lock:
            if _model is None or _model_name != model_name:
                try:
                    import whisper
                except ImportError as exc:  # pragma: no cover - environment dependent
                    raise RuntimeError("Install openai-whisper to transcribe audio") from exc
                _model = whisper.load_model(model_name)
                _model_name = model_name
    return _model


def transcribe_audio(
    path: str | Path,
    model_name: str = "base",
    language: str | None = None,
) -> list[dict[str, object]]:
    """Transcribe audio into timestamped, indexable segments.

    Whisper accepts common audio formats through its ffmpeg dependency.  The
    returned timestamps are seconds from the beginning of the source and are
    retained as metadata for future citations.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    model = _get_model(model_name)
    options: dict[str, object] = {"verbose": False}
    if language:
        options["language"] = language
    result = model.transcribe(str(file_path), **options)

    segments: list[dict[str, object]] = []
    for index, segment in enumerate(result.get("segments", [])):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segments.append({
            "text": text,
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "segment_index": index,
            "source": str(file_path),
            "file_type": "audio",
        })
    return segments
