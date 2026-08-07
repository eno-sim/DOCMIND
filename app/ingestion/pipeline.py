"""Unified ingestion entry point for all supported source modalities.

Keeping dispatch here means indexing does not need separate document, audio,
and video code paths.  Audio and video transcripts become ``TextChunk``
objects just like written documents; keyframe metadata is returned alongside
those chunks for a future multimodal/vector-image extension.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.audio import transcribe_audio
from app.ingestion.text import TextChunk, ingest_text_document
from app.ingestion.video import ingest_video


TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class IngestionResult:
    """Normalized output consumed by future indexing stages."""

    source: str
    modality: str
    chunks: list[TextChunk] = field(default_factory=list)
    assets: list[dict[str, Any]] = field(default_factory=list)


def _transcript_chunks(segments: list[dict[str, object]], source: str, modality: str) -> list[TextChunk]:
    """Convert timestamped Whisper segments to the common chunk model."""
    chunks = []
    for index, segment in enumerate(segments):
        text = str(segment["text"]).strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        chunks.append(TextChunk(
            text=text,
            source=source,
            chunk_id=f"{source}:segment-{index}",
            metadata={
                "source": source,
                "file_type": modality,
                "start_seconds": str(start),
                "end_seconds": str(end),
                "chunk_index": str(index),
            },
        ))
    return chunks


def ingest_file(
    path: str | Path,
    *,
    whisper_model: str = "base",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    keyframe_dir: str | Path | None = None,
    video_interval_seconds: float = 10.0,
) -> IngestionResult:
    """Ingest one supported file and normalize it for indexing.

    Unsupported extensions fail early with a useful message rather than
    silently producing an empty index.  Whisper and OpenCV remain lazy, so
    text ingestion does not require those runtime components to initialize.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Source does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return IngestionResult(
            source=str(file_path),
            modality="text",
            chunks=ingest_text_document(file_path, chunk_size, chunk_overlap),
        )
    if suffix in AUDIO_EXTENSIONS:
        segments = transcribe_audio(file_path, model_name=whisper_model)
        return IngestionResult(
            source=str(file_path),
            modality="audio",
            chunks=_transcript_chunks(segments, str(file_path), "audio"),
        )
    if suffix in VIDEO_EXTENSIONS:
        result = ingest_video(
            file_path,
            model_name=whisper_model,
            interval_seconds=video_interval_seconds,
            keyframe_dir=keyframe_dir,
        )
        transcript = result["transcript"]
        return IngestionResult(
            source=str(file_path),
            modality="video",
            chunks=_transcript_chunks(transcript, str(file_path), "video"),
            assets=result["keyframes"],
        )

    supported = sorted(TEXT_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
    raise ValueError(f"Unsupported file type {suffix!r}; supported types: {', '.join(supported)}")
