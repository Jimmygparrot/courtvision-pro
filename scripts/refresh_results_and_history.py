"""Refresh completed NBA team results and reconcile saved predictions.

Uses the existing validated functions in nba_predictions. It never creates
predictions and never changes an already-resolved prediction.
"""
import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nba_predictions as nba  # noqa: E402

logger = logging.getLogger("courtvision.refresh")

ATTEMPTS = 3
BACKOFF_SECONDS = 30
STALE_LOCK = timedelta(hours=2)
BACKUPS_KEPT = 14

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_LOCKED = 2


class AlreadyRunning(Exception):
    pass


class RunLock:
    """Exclusive lock file; a lock older than STALE_LOCK is treated as abandoned."""

    def __init__(self, path):
        self.path = Path(path)
        self.held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = datetime.now() - datetime.fromtimestamp(
                    self.path.stat().st_mtime
                )
                if age < STALE_LOCK:
                    raise AlreadyRunning(f"Lock held: {self.path} (age {age}).")
                logger.warning("Removing stale lock %s (age %s).", self.path, age)
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as handle:
                handle.write(str(os.getpid()))
            self.held = True
            return self
        raise AlreadyRunning(f"Could not acquire lock: {self.path}")

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(
        log_dir / "refresh_results.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.setFormatter(fmt)
    logger.handlers = [file_handler, console]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def backup_database(db_path, backup_dir):
    """Consistent SQLite backup (not a file copy); prunes old backups."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"nba_prediction_history_{datetime.now():%Y%m%d_%H%M%S}.sqlite3"
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    for old in sorted(backup_dir.glob("nba_prediction_history_*.sqlite3"))[:-BACKUPS_KEPT]:
        old.unlink()
    return target


def refresh_with_retries(csv_path, sleep=time.sleep):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return nba.refresh_current_season_team_snapshot(csv_path)
        except Exception as error:
            logger.warning(
                "Refresh attempt %d/%d failed: %s: %s",
                attempt, ATTEMPTS, type(error).__name__, error,
            )
            if attempt == ATTEMPTS:
                raise
            sleep(BACKOFF_SECONDS * attempt)


def reconcile(csv_path, db_path=None):
    """Resolve saved predictions for completed games; returns rows updated.

    ``db_path=None`` uses the shared PostgreSQL history; a path selects SQLite.
    """
    # Read GAME_ID as text: the default integer parse drops leading zeros, so
    # results would never match the saved ten-digit prediction IDs.
    games = pd.read_csv(csv_path, dtype={"GAME_ID": str})
    results = nba.extract_completed_results(games)
    logger.info("Completed games in results file: %d", len(results))
    updated = nba.update_prediction_results(results, history_path=db_path)
    logger.info("Predictions newly resolved: %d", updated)
    return updated


def run(csv_path, db_path, backup_dir, sleep=time.sleep):
    csv_path = Path(csv_path)
    if db_path is None:
        logger.info("Using the shared PostgreSQL prediction history.")
    else:
        db_path = Path(db_path)
        if not db_path.exists():
            # Opening a missing path would silently create an empty database.
            raise FileNotFoundError(f"Prediction history not found: {db_path}")
        backup = backup_database(db_path, backup_dir)
        logger.info("Database backup written: %s", backup)

    latest = refresh_with_retries(csv_path, sleep)
    logger.info("Results refreshed through %s: %s", latest, csv_path)
    return reconcile(csv_path, db_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(
        ROOT / "data" / "historical" / f"nba_games_{nba.CURRENT_SEASON}.csv"))
    parser.add_argument("--db", default=None, help=(
        "SQLite file to reconcile instead of the shared PostgreSQL history "
        "(legacy/testing only)."))
    parser.add_argument("--log-dir", default=str(ROOT / "logs"))
    parser.add_argument("--backup-dir", default=str(ROOT / "backups"))
    parser.add_argument("--lock-file", default=str(ROOT / "logs" / "refresh_results.lock"))
    args = parser.parse_args(argv)

    setup_logging(args.log_dir)
    try:
        with RunLock(args.lock_file):
            logger.info("Starting results refresh.")
            run(args.csv, args.db, args.backup_dir)
            logger.info("Finished successfully.")
        return EXIT_OK
    except AlreadyRunning as error:
        logger.warning("Skipping: %s", error)
        return EXIT_LOCKED
    except Exception:
        logger.exception("Results refresh failed; existing files were left as they were.")
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
