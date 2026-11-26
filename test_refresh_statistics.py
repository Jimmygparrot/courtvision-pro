import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

import espn_results
import nba_predictions as nba
from scripts import refresh_statistics as rs
from test_espn_results import board, event, fetcher, scheduled, team

TODAY = date(2026, 11, 22)
NOW = datetime(2026, 11, 22, 13, 0, tzinfo=timezone.utc)
TEAMS = sorted(espn_results.teams.get_teams(), key=lambda t: t["id"])


def snapshot_csv(game_date="2026-11-20"):
    rows = []
    for index in range(15):
        home, away = TEAMS[index], TEAMS[index + 15]
        game_id = f"00226{index + 1:05d}"
        for side, other, venue, wl in ((home, away, "vs.", "W"), (away, home, "@", "L")):
            rows.append({
                "SEASON_ID": "22026", "TEAM_ID": side["id"],
                "TEAM_ABBREVIATION": side["abbreviation"],
                "TEAM_NAME": side["full_name"], "GAME_ID": game_id,
                "GAME_DATE": game_date,
                "MATCHUP": f"{side['abbreviation']} {venue} {other['abbreviation']}",
                "WL": wl, "MIN": 240, "PTS": 100,
            })
    return pd.DataFrame(rows).to_csv(index=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(nba, "_today_eastern", lambda: TODAY)
    (tmp_path / "data" / "historical").mkdir(parents=True)
    (tmp_path / rs.TEAM_CSV).write_text(snapshot_csv(), encoding="utf-8")
    sched = scheduled(("0022600055", "2026-11-21", "CHA", "CHI"))
    sched.to_csv(tmp_path / rs.SCHEDULE_CSV, index=False)
    return tmp_path


def player_runner(report=None, code=0):
    """Stand-in for the updater: writes the report the real one would."""
    def run(argv):
        out = argv[argv.index("--out-dir") + 1]
        if report is not None:
            path = rs.Path(out)
            path.mkdir(parents=True, exist_ok=True)
            (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return code
    return run


def player_report(latest="2026-11-21", eligible=3, applied=True, blockers=None):
    return {
        "source": {"latest_game_date": latest},
        "summary": {"new_games": 12},
        "apply": {
            "files_eligible": eligible, "applied": applied,
            "files_applied": eligible if applied else 0,
            "new_player_files_held_back": 25, "blockers": blockers or [],
        },
    }


GOOD_BOARD = {date(2026, 11, 21): board(event("CHA", "CHI", 105, 107))}


def do_run(repo, apply=True, boards=None, runner=None, **kwargs):
    return rs.run(
        repo, apply, out_dir=repo / "out", today=TODAY, now=NOW,
        fetch=fetcher(GOOD_BOARD if boards is None else boards),
        player_runner=runner or player_runner(player_report()), env={}, **kwargs,
    )


def team_bytes(repo):
    return (repo / rs.TEAM_CSV).read_bytes()


# --- team results -----------------------------------------------------------

def test_adds_only_missing_days_and_keeps_existing_bytes_exact(repo):
    before = team_bytes(repo)

    code, outcome = do_run(repo)

    after = team_bytes(repo)
    assert code == rs.EXIT_COMPLETE
    assert outcome["team_results"]["new_rows"] == 2
    header, _, body = before.decode().partition("\n")
    assert after.decode().startswith(header + "\n")
    assert after.decode().endswith(body)
    frame = pd.read_csv(repo / rs.TEAM_CSV, dtype=str)
    assert len(frame) == 32
    assert not frame.duplicated(["GAME_ID", "TEAM_ID"]).any()
    assert frame.iloc[0]["GAME_DATE"] == "2026-11-21"
    assert pd.isna(frame.iloc[0]["MIN"])
    assert frame.iloc[2]["MIN"] == "240"


def test_second_run_is_a_no_op_and_creates_no_changes(repo):
    do_run(repo)
    first_team, first_status = team_bytes(repo), (repo / rs.STATUS_JSON).read_bytes()

    code, outcome = do_run(
        repo, runner=player_runner(player_report(eligible=0, applied=False)))
    assert outcome["status_file_changed"] is False
    assert team_bytes(repo) == first_team
    assert (repo / rs.STATUS_JSON).read_bytes() == first_status


def test_dry_run_writes_nothing(repo):
    before = team_bytes(repo)

    code, outcome = do_run(repo, apply=False)

    assert outcome["mode"] == "dry-run"
    assert outcome["team_results"]["status"] == "updated"
    assert outcome["team_results"]["updated"] is False
    assert team_bytes(repo) == before
    assert not (repo / rs.STATUS_JSON).exists()


def test_espn_failure_keeps_existing_data_and_records_the_error(repo):
    before = team_bytes(repo)
    boards = {date(2026, 11, 21): espn_results.EspnUnavailable("Timeout")}

    code, outcome = do_run(repo, boards=boards)

    assert code == rs.EXIT_ERROR
    assert team_bytes(repo) == before
    status = rs.read_status(repo)
    assert status["last_errors"][0]["component"] == "team_results"
    assert "Timeout" in status["last_errors"][0]["message"]
    assert status["complete"] is False


def test_unfinished_game_is_incomplete_not_an_error(repo):
    boards = {date(2026, 11, 21): board(
        event("CHA", "CHI", 50, 48, status="STATUS_IN_PROGRESS"))}
    before = team_bytes(repo)

    code, outcome = do_run(repo, boards=boards)

    assert code == rs.EXIT_INCOMPLETE
    assert outcome["team_results"]["current"] is False
    assert team_bytes(repo) == before


def test_invalid_existing_snapshot_is_never_modified(repo):
    (repo / rs.TEAM_CSV).write_text("TEAM_ID,GAME_ID,GAME_DATE,WL\n1,1,2026-11-20,W\n")
    before = team_bytes(repo)

    code, outcome = do_run(repo)

    assert code == rs.EXIT_ERROR
    assert "not modified" in outcome["team_results"]["message"]
    assert team_bytes(repo) == before


def test_result_that_fails_validation_is_not_saved(repo, monkeypatch):
    before = team_bytes(repo)
    real = nba._validate_complete_team_results
    calls = []

    def validate(games):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("combined results look wrong")
        return real(games)

    monkeypatch.setattr(nba, "_validate_complete_team_results", validate)

    code, outcome = do_run(repo)

    assert code == rs.EXIT_ERROR
    assert "failed validation" in outcome["team_results"]["message"]
    assert team_bytes(repo) == before


def test_games_already_saved_are_not_duplicated(repo):
    # The snapshot already holds game 0022600055 on 11/21.
    text = snapshot_csv("2026-11-20").splitlines()
    extra = [
        f"22026,{team('CHA')['id']},CHA,Charlotte Hornets,0022600055,2026-11-21,CHA vs. CHI,L,240,105",
        f"22026,{team('CHI')['id']},CHI,Chicago Bulls,0022600055,2026-11-21,CHI @ CHA,W,240,107",
    ]
    (repo / rs.TEAM_CSV).write_text("\n".join([text[0], *extra, *text[1:]]) + "\n")
    before = team_bytes(repo)

    code, outcome = do_run(repo)

    assert outcome["team_results"]["new_rows"] == 0
    assert team_bytes(repo) == before


# --- player stats and the combined outcome ------------------------------------

def test_updater_failure_is_an_error_and_does_not_mark_success(repo):
    runner = player_runner(
        player_report(applied=False, blockers=["source covers 40% of cached players"]),
        code=3,
    )

    code, outcome = do_run(repo, runner=runner)

    assert code == rs.EXIT_ERROR
    assert outcome["player_stats"]["status"] == "error"
    assert "exited with code 3" in outcome["player_stats"]["message"]
    status = rs.read_status(repo)
    assert status["player_stats"] is None
    assert status["team_results"]["latest_game_date"] == "2026-11-21"


def test_lagging_player_source_is_incomplete_until_it_catches_up(repo):
    runner = player_runner(player_report(latest="2026-11-20", eligible=0, applied=False))

    code, outcome = do_run(repo, runner=runner)

    assert code == rs.EXIT_INCOMPLETE
    assert outcome["player_stats"]["current"] is False
    assert rs.read_status(repo)["complete"] is False

    code, _ = do_run(repo, runner=player_runner(player_report(latest="2026-11-21")))
    assert code == rs.EXIT_COMPLETE
    assert rs.read_status(repo)["complete"] is True


def test_player_apply_counts_and_up_to_date_detection(repo):
    _, outcome = do_run(repo, runner=player_runner(player_report(eligible=5)))
    assert outcome["player_stats"]["status"] == "updated"
    assert outcome["player_stats"]["files_applied"] == 5

    _, outcome = do_run(
        repo, runner=player_runner(player_report(eligible=0, applied=False)))
    assert outcome["player_stats"]["status"] == "up_to_date"
    assert outcome["player_stats"]["updated"] is False


def test_crashing_updater_does_not_touch_team_data_or_hide_the_error(repo):
    def crash(argv):
        raise RuntimeError("boom")

    code, outcome = do_run(repo, runner=crash)

    assert code == rs.EXIT_ERROR
    assert "crashed" in outcome["player_stats"]["message"]
    assert len(pd.read_csv(repo / rs.TEAM_CSV)) == 32


# --- status record --------------------------------------------------------------

def test_status_records_last_success_and_coverage(repo):
    do_run(repo)

    status = rs.read_status(repo)

    assert status["complete"] is True
    assert status["last_successful_update"]["utc"] == "2026-11-22T13:00:00Z"
    assert status["last_successful_update"]["team_rows_added"] == 2
    assert status["last_successful_update"]["player_files_updated"] == 3
    assert status["team_results"]["latest_game_date"] == "2026-11-21"
    assert status["player_stats"]["source_latest_game_date"] == "2026-11-21"
    assert status["last_errors"] == []


def test_repeated_identical_error_is_recorded_once_and_cleared_on_success(repo):
    down = {date(2026, 11, 21): espn_results.EspnUnavailable("Timeout")}
    do_run(repo, boards=down)
    first = (repo / rs.STATUS_JSON).read_bytes()

    _, outcome = rs.run(
        repo, True, out_dir=repo / "out2", today=TODAY,
        now=datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc),
        fetch=fetcher(down),
        player_runner=player_runner(player_report(eligible=0, applied=False)), env={},
    )

    assert outcome["status_file_changed"] is False
    assert (repo / rs.STATUS_JSON).read_bytes() == first
    do_run(repo)
    assert rs.read_status(repo)["last_errors"] == []


def test_failure_never_overwrites_previous_coverage(repo):
    do_run(repo)
    good = rs.read_status(repo)

    do_run(repo, boards={}, runner=player_runner(None, code=2))

    status = rs.read_status(repo)
    assert status["team_results"] == good["team_results"]
    assert status["last_successful_update"] == good["last_successful_update"]
    assert status["player_stats"] == good["player_stats"]


def test_probe_reports_espn_reachability(repo):
    boards = {**GOOD_BOARD, TODAY: board()}

    _, outcome = rs.run(
        repo, False, probe=True, out_dir=repo / "out", today=TODAY, now=NOW,
        fetch=fetcher(boards), player_runner=player_runner(player_report()), env={},
    )

    assert outcome["team_results"]["espn_events_today"] == 0
    assert "ESPN reachable" in rs.render(outcome)


# --- staging guard -------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    f"M  {rs.TEAM_CSV.as_posix()}",
    f"M  {rs.STATUS_JSON.as_posix()}",
    f"A  {rs.STATUS_JSON.as_posix()}",
    "M  data/nba_cache/2026-27/1626157.csv",
])
def test_guard_allows_only_statistics_files(line):
    assert rs.unexpected_changes([line]) == []


@pytest.mark.parametrize("line", [
    "M  app.py",
    "A  data/nba_cache/2026-27/1.csv",
    "D  data/nba_cache/2026-27/1.csv",
    " M data/historical/nba_games_2026-27.csv",
    "?? data/other.csv",
    f"D  {rs.TEAM_CSV.as_posix()}",
])
def test_guard_refuses_everything_else(line):
    assert rs.unexpected_changes([line])
