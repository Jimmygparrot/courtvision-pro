"""Download and read the source game-log file."""

import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .models import SourceError
from .schema import MIN_SOURCE_ROWS


def download_source(url, destination, timeout=60, min_bytes=100_000):
    """Download ``url`` to ``destination`` and return request details."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "courtvision-pro-dry-run"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        raise SourceError(f"Download failed: HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise SourceError(
            f"Download failed: {type(error).__name__}: {error}"
        ) from error

    if status != 200:
        raise SourceError(f"Download failed: HTTP {status}")
    if len(body) < min_bytes:
        raise SourceError(
            f"Downloaded file is too small ({len(body)} bytes)."
        )
    destination.write_bytes(body)
    return {
        "url": url,
        "http_status": status,
        "bytes": len(body),
        "seconds": round(time.monotonic() - started, 2),
        "path": str(destination),
    }


def read_source_file(path):
    try:
        frame = pd.read_csv(path, dtype={"game_id": str, "player_id": str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise SourceError(f"Cannot read source file: {error}") from error
    if len(frame) < MIN_SOURCE_ROWS:
        raise SourceError(
            f"Source has only {len(frame)} rows; expected at least "
            f"{MIN_SOURCE_ROWS}."
        )
    return frame
