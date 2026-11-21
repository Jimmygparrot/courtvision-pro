"""Check that only existing player cache CSVs are staged for commit.

Usage in CI: ``git status --porcelain=v1 --untracked-files=all |
python -m player_cache_updater.staging``. Exits non-zero when anything other
than a modified (not added, deleted or renamed) ``data/nba_cache/<season>/<id>.csv``
is present, whether staged or not.
"""

import re
import sys

from .schema import SEASON

ALLOWED_PATH = re.compile(rf"^data/nba_cache/{re.escape(SEASON)}/\d+\.csv$")


def unexpected_changes(porcelain_lines):
    """Return problems for any status line that is not a staged CSV edit."""
    problems = []
    for line in porcelain_lines:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        if status != "M ":
            problems.append(f"unexpected status {status!r}: {path}")
        elif not ALLOWED_PATH.match(path):
            problems.append(f"path outside the player cache: {path}")
    return problems


def main():
    problems = unexpected_changes(sys.stdin.read().splitlines())
    for problem in problems:
        print(f"REFUSING TO COMMIT: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
