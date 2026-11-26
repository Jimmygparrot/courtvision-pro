import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import nba_predictions as nba
from scripts import refresh_results_and_history as job

ROOT = Path(__file__).resolve().parent


def _games(rows):
    # rows: (game_id, home_id, away_id, home_wins)
    out = []
    for gid, home, away, home_wins in rows:
        out.append({"GAME_ID": gid, "TEAM_ID": home, "WL": "W" if home_wins else "L",
                    "MATCHUP": "AAA vs. BBB", "GAME_DATE": "2026-11-20"})
        out.append({"GAME_ID": gid, "TEAM_ID": away, "WL": "L" if home_wins else "W",
                    "MATCHUP": "BBB @ AAA", "GAME_DATE": "2026-11-20"})
    return pd.DataFrame(out)


def _prediction(game_id, home, away, predicted):
    return pd.DataFrame([{
        "game_id": game_id, "game_date": "2026-11-20",
        "tipoff_utc": "2026-11-21T00:00:00+00:00",
        "home_team_id": home, "home_team": "Home", "away_team_id": away,
        "away_team": "Away", "predicted_winner": predicted,
        "home_win_probability": 0.6, "away_win_probability": 0.4,
        "win_probability": 0.6,
        "created_at": "2026-11-20T12:00:00+00:00",
        "model_version": "test-v1", "rating_source": "test", "status": "scheduled",
    }])


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "history.sqlite3"
    csv = tmp_path / "games.csv"
    nba.save_prediction_snapshots(_prediction("0022600001", "1", "2", "Home"), db)
    nba.save_prediction_snapshots(_prediction("0022600002", "3", "4", "Away"), db)
    _games([("0022600001", 1, 2, True)]).to_csv(csv, index=False)
    return db, csv, tmp_path


def test_leading_zero_game_ids_match(env):
    db, csv, _ = env
    assert csv.read_text().splitlines()[1].startswith("0022600001")
    assert job.reconcile(csv, db) == 1


def test_extract_completed_results_matches_ids_that_lost_leading_zeros(env):
    # Regression: a default pd.read_csv parses GAME_ID as int (22600001),
    # which previously never matched the saved ten-digit prediction ID.
    db, csv, _ = env
    games = pd.read_csv(csv)
    assert games["GAME_ID"].dtype.kind == "i"
    results = nba.extract_completed_results(games)
    assert results == {"0022600001": "1"}
    assert nba.update_prediction_results(results, history_path=db) == 1


def _rows(db):
    with sqlite3.connect(db) as c:
        return pd.read_sql_query("select * from predictions order by game_id", c)


ORIGINAL = ["game_id", "predicted_winner", "home_win_probability",
            "away_win_probability", "win_probability", "created_at", "model_version"]


def test_reconcile_resolves_completed_and_preserves_originals(env):
    db, csv, _ = env
    before = _rows(db)
    assert job.reconcile(csv, db) == 1
    after = _rows(db)
    assert before[ORIGINAL].equals(after[ORIGINAL])
    assert after.loc[0, "actual_winner"] == "1"
    assert pd.isna(after.loc[1, "actual_winner"])  # game not completed stays pending


def test_duplicate_run_is_idempotent_and_never_overwrites(env):
    db, csv, _ = env
    assert job.reconcile(csv, db) == 1
    first = _rows(db)
    assert job.reconcile(csv, db) == 0
    # a contradictory later result must not change a resolved row
    _games([("0022600001", 1, 2, False)]).to_csv(csv, index=False)
    assert job.reconcile(csv, db) == 0
    assert _rows(db).equals(first)
    assert len(first) == 2  # duplicate saves never add rows
    assert nba.save_prediction_snapshots(_prediction("0022600001", "1", "2", "Away"), db) == 0


def test_run_success_refreshes_and_reconciles(env, monkeypatch):
    db, csv, tmp = env
    monkeypatch.setattr(nba, "refresh_current_season_team_snapshot",
                        lambda path: "2026-11-20")
    assert job.run(csv, db, tmp / "bk") == 1
    assert len(list((tmp / "bk").glob("*.sqlite3"))) == 1


def test_api_failure_leaves_files_untouched_and_retries(env, monkeypatch):
    db, csv, tmp = env
    calls = []

    def boom(path):
        calls.append(1)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(nba, "refresh_current_season_team_snapshot", boom)
    csv_before, db_before = csv.read_bytes(), _rows(db)
    sleeps = []
    with pytest.raises(TimeoutError):
        job.run(csv, db, tmp / "bk", sleep=sleeps.append)
    assert len(calls) == job.ATTEMPTS and len(sleeps) == job.ATTEMPTS - 1
    assert csv.read_bytes() == csv_before
    assert _rows(db).equals(db_before)


def test_main_returns_failure_code_without_crashing(env, monkeypatch):
    db, csv, tmp = env
    monkeypatch.setattr(nba, "refresh_current_season_team_snapshot",
                        lambda p: (_ for _ in ()).throw(ValueError("bad data")))
    monkeypatch.setattr(job, "BACKOFF_SECONDS", 0)
    args = ["--csv", str(csv), "--db", str(db), "--log-dir", str(tmp / "logs"),
            "--backup-dir", str(tmp / "bk"), "--lock-file", str(tmp / "l.lock")]
    assert job.main(args) == job.EXIT_FAILED
    assert not (tmp / "l.lock").exists()  # lock released after failure


def test_overlapping_run_is_blocked(env):
    db, csv, tmp = env
    lock = tmp / "l.lock"
    with job.RunLock(lock):
        args = ["--csv", str(csv), "--db", str(db), "--log-dir", str(tmp / "logs"),
                "--backup-dir", str(tmp / "bk"), "--lock-file", str(lock)]
        assert job.main(args) == job.EXIT_LOCKED
    assert not lock.exists()


def test_missing_database_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        job.run(tmp_path / "g.csv", tmp_path / "none.sqlite3", tmp_path / "bk")
    assert not (tmp_path / "none.sqlite3").exists()
