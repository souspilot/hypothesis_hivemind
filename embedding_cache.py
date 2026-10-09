"""Content fingerprints for per-paper embedding caches."""

import hashlib
import json
from pathlib import Path

from config import EMBEDDING_MODEL


def fingerprint(samples: list[str]) -> str:
    """Hash the embedding model and ordered response texts."""
    payload = json.dumps([EMBEDDING_MODEL, samples], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprints_path(embeddings_path: Path) -> Path:
    return embeddings_path.parent / ".fingerprints" / embeddings_path.name


def load_fingerprints(embeddings_path: Path) -> dict[str, str]:
    path = fingerprints_path(embeddings_path)
    return json.loads(path.read_text()) if path.exists() else {}
