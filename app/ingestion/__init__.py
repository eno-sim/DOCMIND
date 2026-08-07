"""Document, audio, and video ingestion components.

``ingest_file`` is the stable, modality-independent entry point.  The
lower-level modules remain available for callers that need specialized
behavior, such as extracting only video keyframes.
"""

from app.ingestion.pipeline import IngestionResult, ingest_file

__all__ = ["IngestionResult", "ingest_file"]
