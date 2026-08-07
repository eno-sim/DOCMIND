"""Video ingestion using OpenCV keyframes and Whisper audio transcription.

The first implementation samples one representative frame at a fixed time
interval.  Fixed sampling is deterministic and inexpensive; scene-change
sampling can be added later and evaluated against this baseline.  Video audio
is sent through the same Whisper function used by standalone audio files.
"""

from pathlib import Path
from typing import Any

from app.ingestion.audio import transcribe_audio


def extract_keyframes(
    path: str | Path,
    interval_seconds: float = 10.0,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Extract representative frames with OpenCV.

    When ``output_dir`` is supplied, frames are persisted as JPEG files and
    each result includes ``path``.  Without it, only frame metadata is
    returned, avoiding unexpected disk writes during a library call.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {file_path}")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install opencv-python-headless to ingest video files") from exc

    destination = Path(output_dir) if output_dir is not None else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {file_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else None
    interval_frames = max(1, round(interval_seconds * fps)) if fps > 0 else 1
    frames: list[dict[str, Any]] = []

    try:
        frame_number = 0
        next_frame = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_number >= next_frame:
                timestamp = frame_number / fps if fps > 0 else float(frame_number)
                item: dict[str, Any] = {
                    "source": str(file_path),
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "file_type": "video",
                }
                if destination is not None:
                    frame_path = destination / f"{file_path.stem}-{frame_number:08d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise OSError(f"Could not write keyframe: {frame_path}")
                    item["path"] = str(frame_path)
                frames.append(item)
                next_frame += interval_frames
            frame_number += 1
    finally:
        capture.release()

    # Duration is useful to callers but is not repeated on every frame.
    for frame in frames:
        frame["duration"] = duration
    return frames


def ingest_video(
    path: str | Path,
    model_name: str = "base",
    interval_seconds: float = 10.0,
    keyframe_dir: str | Path | None = None,
) -> dict[str, object]:
    """Extract video keyframes and transcribe its audio track.

    Whisper uses ffmpeg internally and can read the video container directly,
    so no temporary audio file is needed here.  Both outputs retain source
    and timestamp metadata and can later be converted into normal TextChunks.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {file_path}")
    return {
        "source": str(file_path),
        "keyframes": extract_keyframes(file_path, interval_seconds, keyframe_dir),
        "transcript": transcribe_audio(file_path, model_name=model_name),
    }
