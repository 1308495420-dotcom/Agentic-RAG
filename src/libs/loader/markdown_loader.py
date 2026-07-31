"""Markdown Loader for .md and .markdown files.

This module implements loading Markdown files directly, without any
external library — Markdown is plain text, just read and go.

Features:
- Reads .md / .markdown files via open().read()
- Parses YAML frontmatter (--- ... ---) into metadata
- Extracts title from first # heading
- No external dependencies (no MarkItDown needed)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Supported file extensions
MD_EXTENSIONS: tuple[str, ...] = (".md", ".markdown")


class MarkdownLoader(BaseLoader):
    """Load Markdown files directly as-is.

    Unlike PdfLoader, Markdown files are already plain text — no parsing
    needed.  The file content becomes the Document text directly.

    YAML frontmatter between ``---`` delimiters at the top of the file is
    automatically parsed into metadata (author, date, tags, etc.) and
    stripped from the body text so it doesn't pollute chunks.

    Configuration:
        encoding: Text encoding (default: "utf-8")
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)

        if path.suffix.lower() not in MD_EXTENSIONS:
            raise ValueError(
                f"File is not a Markdown file: {path}. "
                f"Expected extension: {', '.join(MD_EXTENSIONS)}"
            )

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            raw = path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as e:
            raise RuntimeError(
                f"Failed to decode Markdown file {path} with encoding "
                f"'{self.encoding}': {e}"
            ) from e

        if not raw.strip():
            logger.warning("Markdown file is empty: %s", path)

        # Parse frontmatter before building metadata
        fm, body = _parse_frontmatter(raw)

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
        }

        # Merge frontmatter into metadata (lower priority than explicit keys)
        metadata.update(fm)

        title = fm.get("title") or _extract_md_title(body)
        if title:
            metadata["title"] = title

        return Document(id=doc_id, text=body, metadata=metadata)

    def _compute_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from Markdown text.

    If the text starts with ``---``, everything between the first two
    ``---`` lines is parsed as YAML and returned as a dict.  The body
    is the remainder after the closing ``---``.

    Args:
        text: Raw Markdown file content.

    Returns:
        Tuple of (frontmatter_dict, body_text).  If no frontmatter is
        found, returns ``({}, text)``.
    """
    if not text.startswith("---\n"):
        return {}, text

    # Find closing delimiter
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    frontmatter_raw = text[4:end].strip()
    body = text[end + 4:].lstrip("\n")

    if not frontmatter_raw:
        return {}, body

    try:
        fm = yaml.safe_load(frontmatter_raw)
    except yaml.YAMLError as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
        return {}, body

    if not isinstance(fm, dict):
        return {}, body

    # Flatten list values to comma-separated strings for filter-friendliness
    result: Dict[str, Any] = {}
    for key, value in fm.items():
        if isinstance(value, list):
            result[key] = ", ".join(str(v) for v in value)
        else:
            result[key] = value

    return result, body


def _extract_md_title(text: str) -> Optional[str]:
    """Extract title from the first # heading in Markdown text."""
    for line in text.split("\n")[:30]:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
        if stripped and not stripped.startswith("#"):
            # First non-heading, non-empty line as fallback title
            return stripped
    return None
