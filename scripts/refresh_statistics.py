"""Daily refresh of the statistics the deployed Streamlit app reads from the repo.

Two independent steps, each safe to repeat:

* Team results: the saved current-season snapshot is extended with ESPN finals
  that are confirmed against the saved schedule (see ``espn_results``). Only
  missing days are fetched. New rows are placed ahead of the existing rows,
  which are carried over byte for byte, and the result is validated before and
  after the atomic write.
* Player game logs: the existing ``player_cache_updater`` (sportsdataverse),
  which has its own coverage gates, backup and rollback.

A failed or incomplete step never replaces valid data. The outcome is recorded
in ``data/refresh_status.json``, which is rewritten only when its content
changes, so repeated no-op runs create no commits.

Exit codes: 0 complete, 1 error, 10 incomplete (retry later).
Dry run by default: nothing is written unless ``--apply`` is given.
"""
import argparse
import copy
import io
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import espn_results  # noqa: E402
import nba_predictions as nba  # noqa: E402
from player_cache_updater import cli as player_cli  # noqa: E402
from player_cache_updater.schema import SEASON as PLAYER_SEASON  # noqa: E402

TEAM_CSV = Path("data") / "historical" / f"nba_games_{nba.CURRENT_SEASON}.csv"
SCHEDULE_CSV = Path("data") / f"nba_schedule_{nba.CURRENT_SEASON}.csv"
STATUS_JSON = Path("data") / "refresh_status.json"
PLAYER_CACHE_DIR = Path("data") / "nba_cache" / PLAYER_SEASON

EXIT_COMPLETE = 0
EXIT_ERROR = 1
EXIT_INCOMPLETE = 10

MIN_PLAYER_COVERAGE = "0.9"
MAX_CORRECTED_RECORDS = "300"
STATUS_SCHEMA_VERSION = 1

OK_STATES = {"updated", "up_to_date", "pending"}
PLAYER_PATH = re.compile(
    rf"^data/nba_cache/{re.escape(PLAYER_SEASON)}/\d+\.csv$"
)


def _safe(error):
    return f"{type(error).__name__}: {str(error)[:200]}"


def _atomic_write(path, data):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# --- team results -------------------------------------------------------------

def refresh_team_results(root, today, apply, fetch=None, probe=False):
    """Extend the team snapshot with confirmed ESPN finals; never lose rows."""
    result = {
        "component": "team_results", "status": "error", "updated": False,
        "new_rows": 0, "current": False, "latest_game_date": None,
        "complete_through": None, "message": None,
    }
    path = Path(root) / TEAM_CSV
    try:
        original = path.read_bytes()
        existing_text = original.decode("utf-8")
        _, latest = nba._validate_complete_team_results(
            pd.read_csv(io.StringIO(existing_text))
        )
        result["latest_game_date"] = latest.isoformat()
        existing_pairs = set(zip(
            pd.read_csv(io.StringIO(existing_text), dtype=str)["GAME_ID"]
            .map(nba._normalize_game_id),
            pd.read_csv(io.StringIO(existing_text), dtype=str)["TEAM_ID"],
        ))
    except Exception as error:
        result["message"] = (
            "Existing team snapshot is unreadable or invalid and was not "
            f"modified ({_safe(error)})."
        )
        return result

    try:
        scheduled = nba._read_saved_schedule(Path(root) / SCHEDULE_CSV)
    except Exception as error:
        result["message"] = f"Saved schedule is unavailable ({_safe(error)})."
        return result

    try:
        if probe:
            payload = (fetch or espn_results.fetch_scoreboard)(today)
            result["espn_events_today"] = len(payload.get("events", []))
        new_rows, complete_through = espn_results.fetch_missing_team_results(
            scheduled, latest, today,
            season_id="2" + nba.CURRENT_SEASON[:4], fetch=fetch,
        )
    except Exception as error:
        result["message"] = f"ESPN results unavailable ({_safe(error)})."
        return result

    result["complete_through"] = complete_through.isoformat()
    result["current"] = complete_through >= today - timedelta(days=1)

    if not new_rows.empty:
        keys = list(zip(
            new_rows["GAME_ID"].map(nba._normalize_game_id),
            new_rows["TEAM_ID"].astype(str),
        ))
        fresh = [key not in existing_pairs for key in keys]
        new_rows = new_rows[fresh]

    if new_rows.empty:
        result.update(status="up_to_date", rows=_row_count(existing_text))
        return result

    header, newline, body = _split_header(existing_text)
    columns = header.strip().split(",")
    block = new_rows.astype(str).reindex(columns=columns, fill_value="")
    block = block.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ID"], ascending=[False, False, True]
    )
    eol = "\r\n" if "\r\n" in existing_text else "\n"
    proposed = header + block.to_csv(
        index=False, header=False, lineterminator=eol) + body

    try:
        _, new_latest = nba._validate_complete_team_results(
            pd.read_csv(io.StringIO(proposed))
        )
        if new_latest < latest:
            raise ValueError("Refreshed results are older than the snapshot.")
        if not proposed.endswith(body) or not proposed.startswith(header):
            raise ValueError("Existing rows would be altered.")
    except Exception as error:
        result["message"] = (
            f"Refreshed team results failed validation and were not saved "
            f"({_safe(error)})."
        )
        return result

    result.update(
        status="updated", new_rows=len(new_rows),
        latest_game_date=new_latest.isoformat(),
        rows=_row_count(proposed),
    )
    if not apply:
        return result

    try:
        _atomic_write(path, proposed.encode("utf-8"))
        nba._validate_complete_team_results(pd.read_csv(path))
    except Exception as error:
        _atomic_write(path, original)
        result.update(
            status="error", new_rows=0,
            latest_game_date=latest.isoformat(),
            message=f"Write check failed; the snapshot was restored ({_safe(error)}).",
        )
        return result
    result["updated"] = True
    return result


def _split_header(text):
    index = text.index("\n") + 1
    return text[:index], "\n", text[index:]


def _row_count(text):
    return max(0, len(text.splitlines()) - 1)


# --- player game logs ---------------------------------------------------------

def refresh_player_stats(root, apply, out_dir, backup_dir, reference_date,
                         runner=None):
    """Run the validated player cache updater; return a status summary."""
    runner = runner or player_cli.main
    out_dir = Path(out_dir)
    argv = [
        "--cache-dir", str(Path(root) / PLAYER_CACHE_DIR),
        "--players-csv", str(Path(root) / "nba_players.csv"),
        "--out-dir", str(out_dir),
        "--min-player-coverage", MIN_PLAYER_COVERAGE,
        "--max-corrected-records", MAX_CORRECTED_RECORDS,
    ]
    if apply:
        argv += ["--apply", "--backup-dir", str(backup_dir)]

    result = {
        "component": "player_stats", "status": "error", "updated": False,
        "current": False, "source_latest_game_date": None, "message": None,
    }
    try:
        code = runner(argv)
    except Exception as error:
        result["message"] = f"Player updater crashed ({_safe(error)})."
        return result

    report = {}
    report_path = out_dir / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
    info = report.get("apply", {})
    summary = report.get("summary", {})
    latest = report.get("source", {}).get("latest_game_date")
    result["source_latest_game_date"] = latest

    if code != 0:
        reasons = "; ".join(info.get("blockers", [])[:3]) or info.get("error")
        result["message"] = (
            f"Player updater exited with code {code}"
            + (f": {reasons}" if reasons else " (see the run log).")
        )
        return result

    result.update(
        new_games=summary.get("new_games", 0),
        files_eligible=info.get("files_eligible", 0),
        new_player_files_held_back=info.get("new_player_files_held_back", 0),
    )
    if apply:
        applied = int(info.get("files_applied", 0)) if info.get("applied") else 0
        result["files_applied"] = applied
        result["updated"] = applied > 0
        result["status"] = "updated" if applied else "up_to_date"
    else:
        result["status"] = "pending" if info.get("files_eligible") else "up_to_date"
    result["current"] = bool(
        latest and (reference_date is None
                    or date.fromisoformat(latest) >= reference_date)
    )
    return result


# --- status record ------------------------------------------------------------

def compute_status(old, team, player, complete, now, run_url):
    """Return the new status document (pure; the caller decides to write)."""
    status = copy.deepcopy(old) if old else {}
    status.setdefault("last_successful_update", None)
    status.setdefault("team_results", None)
    status.setdefault("player_stats", None)
    status["schema_version"] = STATUS_SCHEMA_VERSION

    if team["status"] in OK_STATES:
        status["team_results"] = {
            "latest_game_date": team["latest_game_date"],
            "rows": team.get("rows"),
            "source": "saved snapshot extended with ESPN scoreboard finals",
        }
    if player["status"] in OK_STATES and player["source_latest_game_date"]:
        status["player_stats"] = {
            "source_latest_game_date": player["source_latest_game_date"],
            "source": "sportsdataverse player_game_logs",
        }

    if team["updated"] or player["updated"]:
        status["last_successful_update"] = {
            "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "team_rows_added": team["new_rows"],
            "player_files_updated": player.get("files_applied", 0),
            "run_url": run_url,
        }

    previous = {
        (item["component"], item["message"]): item
        for item in (old or {}).get("last_errors", [])
    }
    errors = []
    for part in (team, player):
        if part["status"] == "error":
            key = (part["component"], part["message"])
            errors.append(previous.get(key) or {
                "component": part["component"],
                "message": part["message"],
                "first_seen_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
    status["last_errors"] = errors
    status["complete"] = complete
    return status


def status_text(status):
    return json.dumps(status, indent=2, sort_keys=True) + "\n"


def write_status_if_changed(root, new_status):
    path = Path(root) / STATUS_JSON
    text = status_text(new_status)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, text.encode("utf-8"))
    return True


def read_status(root):
    path = Path(root) / STATUS_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- orchestration ------------------------------------------------------------

def run(root, apply, probe=False, out_dir=None, today=None, fetch=None,
        player_runner=None, now=None, env=None):
    env = os.environ if env is None else env
    today = today or nba._today_eastern()
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="courtvision_refresh_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_url = None
    if env.get("GITHUB_RUN_ID"):
        run_url = (
            f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{env.get('GITHUB_REPOSITORY', '')}/actions/runs/{env['GITHUB_RUN_ID']}"
        )

    team = refresh_team_results(root, today, apply, fetch=fetch, probe=probe)
    reference = (
        date.fromisoformat(team["latest_game_date"])
        if team["latest_game_date"] else today - timedelta(days=1)
    )
    player = refresh_player_stats(
        root, apply, out_dir / "players", out_dir / "players_backup",
        reference, runner=player_runner,
    )

    failed = team["status"] == "error" or player["status"] == "error"
    complete = (
        not failed and team["current"] and player["current"]
    )
    old = read_status(root)
    new_status = compute_status(old, team, player, complete, now, run_url)
    status_changed = False
    if apply:
        status_changed = write_status_if_changed(root, new_status)

    exit_code = (
        EXIT_ERROR if failed else EXIT_COMPLETE if complete else EXIT_INCOMPLETE
    )
    outcome = {
        "mode": "apply" if apply else "dry-run",
        "exit_code": exit_code,
        "complete": complete,
        "team_results": team,
        "player_stats": player,
        "status_file_changed": status_changed,
        "proposed_status": new_status,
    }
    (out_dir / "refresh_result.json").write_text(
        json.dumps(outcome, indent=2, default=str), encoding="utf-8"
    )
    return exit_code, outcome


def render(outcome):
    team, player = outcome["team_results"], outcome["player_stats"]
    lines = [f"Statistics refresh ({outcome['mode']})"]
    lines.append(
        f"Team results: {team['status']}; new rows {team['new_rows']}; "
        f"latest game {team['latest_game_date']}; confirmed through "
        f"{team['complete_through']}; current={team['current']}"
    )
    if "espn_events_today" in team:
        lines.append(f"  ESPN reachable; {team['espn_events_today']} event(s) today.")
    if team["message"]:
        lines.append(f"  ERROR: {team['message']}")
    lines.append(
        f"Player stats: {player['status']}; source latest game "
        f"{player['source_latest_game_date']}; new games "
        f"{player.get('new_games')}; files eligible {player.get('files_eligible')}; "
        f"current={player['current']}"
    )
    if player["message"]:
        lines.append(f"  ERROR: {player['message']}")
    lines.append(
        f"Complete: {outcome['complete']} (exit {outcome['exit_code']}); "
        f"status file changed: {outcome['status_file_changed']}"
    )
    return "\n".join(lines)


# --- staging guard ------------------------------------------------------------

def unexpected_changes(porcelain_lines):
    """Problems for any change other than the allowed statistics files."""
    team_path = TEAM_CSV.as_posix()
    status_path = STATUS_JSON.as_posix()
    problems = []
    for line in porcelain_lines:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        if path == team_path and code == "M ":
            continue
        if path == status_path and code in ("M ", "A "):
            continue
        if PLAYER_PATH.match(path) and code == "M ":
            continue
        problems.append(f"not allowed ({code.strip() or '?'}): {path}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write validated updates (default: dry run).")
    parser.add_argument("--probe", action="store_true",
                        help="Also request today's ESPN scoreboard to prove access.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-staged", action="store_true",
                        help="Read `git status --porcelain=v1` from stdin and "
                             "refuse unexpected changes.")
    args = parser.parse_args(argv)

    if args.check_staged:
        problems = unexpected_changes(sys.stdin.read().splitlines())
        for problem in problems:
            print(f"REFUSING TO COMMIT: {problem}")
        return 1 if problems else 0

    code, outcome = run(args.root, args.apply, probe=args.probe,
                        out_dir=args.out_dir)
    print(render(outcome))
    return code


if __name__ == "__main__":
    sys.exit(main())
