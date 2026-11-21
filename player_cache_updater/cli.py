"""Command line entry point for the player cache updater."""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import apply, compare, normalize, report, source
from .models import LocalFileError, SourceError
from .schema import SEASON, SEASON_START_YEAR, SOURCE_URL

ROOT = Path(__file__).resolve().parent.parent
STALE_AFTER_DAYS = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m player_cache_updater",
        description=(
            "Build proposed player cache CSVs from the sportsdataverse game "
            "logs. Dry run by default: nothing is written to the cache "
            "unless --apply is given and every validation passes."
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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace cache files with the validated proposals (after backup).",
    )
    parser.add_argument(
        "--include-new-players",
        action="store_true",
        help="With --apply, also create cache files for players without one.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Backup directory for --apply (default: <out-dir>/backup).",
    )
    parser.add_argument(
        "--rollback",
        type=Path,
        metavar="BACKUP_DIR",
        help="Restore the cache from a backup made by --apply, then exit.",
    )
    parser.add_argument(
        "--force-rollback",
        action="store_true",
        help="Restore even files modified after the apply.",
    )
    parser.add_argument(
        "--min-player-coverage",
        type=float,
        default=0.0,
        help="Block applying unless this share of cached players is in the "
        "source (0-1).",
    )
    parser.add_argument(
        "--max-corrected-records",
        type=int,
        default=None,
        help="Block applying if more records than this would be corrected.",
    )
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
    proposed_files = {}
    output_problems = []
    for plan in plans:
        if not plan.changed:
            continue
        path = report.write_proposed(plan, out_dir)
        written.append(path)
        proposed_files[plan.player_id] = (plan, path)
        output_problems.extend(
            f"{path.name}: {problem}"
            for problem in report.validate_proposed_file(path)
        )

    context = {
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
        "cache_player_count": len(cache_files),
        "source_latest": latest,
        "cache_latest": max(
            (plan.local_latest.date() for plan in plans if plan.local_latest is not None),
            default=None,
        ),
        "written": written,
        "output_problems": output_problems,
        "source_info": source_info,
        "warnings": warnings,
    }
    result = report.build_report(context)

    targets = {
        player_id: path
        for player_id, (plan, path) in proposed_files.items()
        if plan.existing_file or args.include_new_players
    }
    held_back = len(proposed_files) - len(targets)
    blockers = apply.gate_problems(
        context, args.min_player_coverage, args.max_corrected_records
    )
    result["apply"] = {
        "mode": "apply" if args.apply else "dry-run",
        "gate_passed": not blockers,
        "blockers": blockers[:20],
        "files_eligible": len(targets),
        "new_player_files_held_back": held_back,
    }

    status = 0 if not output_problems else 1
    if args.apply:
        status = apply_validated(args, context, out_dir, targets, blockers, result)
    report.write_report(result, out_dir)
    print(report.render_text(result, out_dir))
    return status


def apply_validated(args, context, out_dir, targets, blockers, result):
    info = result["apply"]
    if blockers:
        info["applied"] = False
        return 3

    backup_dir = args.backup_dir or out_dir / "backup"
    report.ensure_scratch_directory(
        backup_dir, [ROOT / "data", args.cache_dir], create=False
    )
    try:
        manifest = apply.apply_changes(targets, args.cache_dir, backup_dir)
    except apply.ApplyError as error:
        info.update(applied=False, error=str(error))
        return 4

    problems = []
    for entry in manifest["entries"]:
        path = args.cache_dir / entry["file"]
        problems.extend(
            f"{path.name}: {problem}"
            for problem in report.validate_proposed_file(path)
        )
    info.update(
        applied=True,
        backup_dir=str(Path(backup_dir).resolve()),
        files_applied=len(manifest["entries"]),
        files_created=sum(not entry["existed"] for entry in manifest["entries"]),
        post_apply_problems=problems,
    )
    if problems:
        rolled_back = apply.rollback(backup_dir, args.cache_dir, force=True)
        info.update(applied=False, rolled_back=rolled_back)
        return 4
    return 0


def run_rollback(args):
    outcome = apply.rollback(
        args.rollback, args.cache_dir, force=args.force_rollback
    )
    print(
        f"Rollback complete: restored {outcome['restored']}, removed "
        f"{outcome['removed']} created file(s)."
    )
    if outcome["skipped"]:
        print(
            "Skipped (modified since the apply; use --force-rollback): "
            + ", ".join(outcome["skipped"])
        )
        return 1
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.rollback:
            return run_rollback(args)
        return run(args)
    except (SourceError, ValueError, apply.ApplyError) as error:
        print(f"UPDATER FAILED: {error}")
        return 2
