"""Back up, apply and roll back validated proposed cache files."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"


class ApplyError(Exception):
    """Applying or rolling back cache files failed."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_problems(report_context, min_coverage=0.0, max_corrected=None):
    """Return reasons that block applying; an empty list means safe to apply.

    ``min_coverage`` is the minimum share of cached players that must appear
    in the source, and ``max_corrected`` caps corrected records; both guard
    against a truncated or malformed download.
    """
    problems = []
    if report_context["failures"]:
        problems.append(
            f"{len(report_context['failures'])} validation failure(s)"
        )
    if report_context["output_problems"]:
        problems.append(
            f"{len(report_context['output_problems'])} proposed file problem(s)"
        )
    for plan in report_context["plans"]:
        for problem in plan.problems:
            problems.append(f"player {plan.player_id}: {problem}")

    source_latest = report_context.get("source_latest")
    cache_latest = report_context.get("cache_latest")
    if source_latest and cache_latest and source_latest < cache_latest:
        problems.append(
            f"source latest game date {source_latest} is older than the "
            f"cache latest {cache_latest}"
        )

    cache_players = report_context.get("cache_player_count", 0)
    if min_coverage and cache_players:
        absent = report_context["absent_from_source"]
        coverage = 1 - absent / cache_players
        if coverage < min_coverage:
            problems.append(
                f"source covers {coverage:.0%} of cached players; "
                f"minimum is {min_coverage:.0%}"
            )

    corrected = sum(len(plan.corrections) for plan in report_context["plans"])
    if max_corrected is not None and corrected > max_corrected:
        problems.append(
            f"{corrected} corrected records exceed the limit of {max_corrected}"
        )
    return problems


def apply_changes(targets, cache_dir, backup_dir):
    """Apply ``targets`` (player_id -> proposed file) to ``cache_dir``.

    Every existing file is copied to ``backup_dir`` and recorded in a manifest
    before anything is replaced. If any step fails, all changes are rolled back.
    """
    cache_dir = Path(cache_dir)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)
    (backup_dir / "files").mkdir()

    entries = []
    for player_id, proposed in sorted(targets.items()):
        proposed = Path(proposed)
        destination = cache_dir / f"{player_id}.csv"
        entry = {
            "player_id": player_id,
            "file": destination.name,
            "existed": destination.exists(),
            "original_sha256": None,
            "applied_sha256": sha256_file(proposed),
        }
        if destination.exists():
            shutil.copy2(destination, backup_dir / "files" / destination.name)
            entry["original_sha256"] = sha256_file(destination)
            if sha256_file(backup_dir / "files" / destination.name) != (
                entry["original_sha256"]
            ):
                raise ApplyError(f"Backup of {destination.name} is not exact.")
        entries.append(entry)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir.resolve()),
        "entries": entries,
    }
    manifest_path = backup_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for entry in entries:
            _replace_file(
                targets[entry["player_id"]],
                cache_dir / entry["file"],
                entry["applied_sha256"],
            )
    except Exception as error:
        rollback(backup_dir, cache_dir, force=True)
        raise ApplyError(
            f"Apply failed and was rolled back: {type(error).__name__}: {error}"
        ) from error
    return manifest


def _replace_file(proposed, destination, expected_sha256):
    proposed = Path(proposed)
    if sha256_file(proposed) != expected_sha256:
        raise ApplyError(f"{proposed.name} changed after validation.")
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        shutil.copyfile(proposed, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise ApplyError(f"Copy of {destination.name} is not exact.")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if sha256_file(destination) != expected_sha256:
        raise ApplyError(f"{destination.name} does not match the proposal.")


def rollback(backup_dir, cache_dir=None, force=False):
    """Restore the files recorded in ``backup_dir``'s manifest.

    Files created by the apply are deleted. A file changed since the apply is
    left alone unless ``force`` is set.
    """
    backup_dir = Path(backup_dir)
    manifest_path = backup_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ApplyError(f"No manifest found in {backup_dir}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir = Path(cache_dir or manifest["cache_dir"])

    restored, removed, skipped = 0, 0, []
    for entry in manifest["entries"]:
        destination = cache_dir / entry["file"]
        backup = backup_dir / "files" / entry["file"]
        current = sha256_file(destination) if destination.exists() else None
        untouched = current in (entry["applied_sha256"], entry["original_sha256"])
        if not force and current is not None and not untouched:
            skipped.append(entry["file"])
            continue
        if entry["existed"]:
            shutil.copy2(backup, destination)
            if sha256_file(destination) != entry["original_sha256"]:
                raise ApplyError(f"Restore of {entry['file']} is not exact.")
            restored += 1
        elif destination.exists():
            destination.unlink()
            removed += 1
    return {"restored": restored, "removed": removed, "skipped": skipped}
