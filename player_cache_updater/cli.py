"""Command line entry point for the player cache dry run."""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import compare, normalize, report, source
from .models import LocalFileError, SourceError
from .schema import SEASON, SEASON_START_YEAR, SOURCE_URL

ROOT = Path(__file__).resolve().parent.parent
STALE_AFTER_DAYS = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m player_cache_updater",
        description=(
            "Dry run: build proposed player cache CSVs from the "
            "sportsdataverse game logs. Never writes to data/nba_cache."
        ),
    )
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Use a local copy of the source CSV instead of downloading.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data" / "nba_cache" / SEASON
    )
    parser.add_argument(
        "--players-csv", type=Path, default=ROOT / "nba_players.csv"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Scratch directory (default: a new system temp directory).",
    )
    parser.add_argument("--timeout", type=float, default=60)
    return parser.parse_args(argv)


def load_roster(path):
    roster = pd.read_csv(path)
    return {int(row.id): str(row.full_name) for row in roster.itertuples()}


def run(args):
    protected = [ROOT / "data", args.cache_dir]
    out_dir = args.out_dir or Path(
        tempfile.mkdtemp(prefix="courtvision_player_cache_dryrun_")
    )
    out_dir = report.ensure_scratch_directory(out_dir, protected)
    today = datetime.now(ZoneInfo("America/New_York")).date()
    warnings = []

    if args.source_file:
        source_path = args.source_file
        source_info = {"path": str(source_path)}
    else:
        source_path = out_dir / "source" / "player_game_logs.csv"
        source_info = source.download_source(
            args.url, source_path, timeout=args.timeout
        )
    raw = source.read_source_file(source_path)
    normalized = normalize.normalize_source_frame(raw, SEASON_START_YEAR, today)
    rows = normalized.rows
    if rows.empty:
        raise SourceError("No valid source rows remain after validation.")

    latest = rows["_date"].max().date()
    source_info["latest_game_date"] = latest.isoformat()
    if (today - latest).days > STALE_AFTER_DAYS:
        warnings.append(
            f"Source latest game date {latest} is more than "
            f"{STALE_AFTER_DAYS} days old."
        )

    roster = load_roster(args.players_csv)
    cache_files = {
        int(path.stem): path
        for path in args.cache_dir.glob("*.csv")
        if path.stem.isdigit()
    }
    known = set(roster) | set(cache_files)

    failures = list(normalized.failures)
    plans = []
    unknown = {}
    source_player_ids = set()
    for player_id, group in rows.groupby("Player_ID", sort=True):
        player_id = int(player_id)
        source_player_ids.add(player_id)
        if player_id not in known:
            unknown[player_id] = int(len(group))
            continue

        local = None
        if player_id in cache_files:
            try:
                local = normalize.normalize_local_frame(
                    pd.read_csv(cache_files[player_id], dtype={"Game_ID": str}),
                    cache_files[player_id].name,
                )
            except (LocalFileError, OSError, pd.errors.ParserError,
                    pd.errors.EmptyDataError) as error:
                failures.append(
                    report.failure("cache", player_id, "", str(error))
                )
                continue
        plans.append(
            compare.plan_player(
                player_id, roster.get(player_id, str(player_id)), local, group
            )
        )

    written = []
    output_problems = []
    for plan in plans:
        if not plan.changed:
            continue
        path = report.write_proposed(plan, out_dir)
        written.append(path)
        output_problems.extend(
            f"{path.name}: {problem}"
            for problem in report.validate_proposed_file(path)
        )

    result = report.build_report({
        "plans": plans,
        "failures": failures,
        "source_rows": normalized.raw_rows,
        "valid_rows": len(rows),
        "source_duplicates_removed": normalized.duplicates_removed,
        "pct_filled": normalized.pct_filled,
        "players_in_source": len(source_player_ids),
        "unknown_players": [
            {"player_id": pid, "rows": count} for pid, count in unknown.items()
        ],
        "absent_from_source": len(set(cache_files) - source_player_ids),
        "written": written,
        "output_problems": output_problems,
        "source_info": source_info,
        "warnings": warnings,
    })
    report.write_report(result, out_dir)
    print(report.render_text(result, out_dir))
    return 0 if not output_problems else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (SourceError, ValueError) as error:
        print(f"DRY RUN FAILED: {error}")
        return 2
