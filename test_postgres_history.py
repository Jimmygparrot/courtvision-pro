"""PostgreSQL history integration tests on an isolated embedded server.

These never touch Supabase: they start a throwaway local PostgreSQL (pgserver),
install supabase/01_schema.sql, and connect as the restricted courtvision_app
role. Skipped when pgserver is not installed.
"""
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg
import pytest

import db
import nba_predictions as nba
from scripts import refresh_results_and_history as job

SCHEMA = Path(__file__).resolve().parent / "supabase" / "01_schema.sql"
REAL_CONNECT = psycopg.connect
SECRET = "s3cret-test-password"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="delete")
    parsed = urlparse(srv.get_uri())
    with REAL_CONNECT(srv.get_uri(), autocommit=True) as owner:
        for role in ("anon", "authenticated"):
            owner.execute(f"CREATE ROLE {role} NOLOGIN")
        owner.execute(SCHEMA.read_text(encoding="utf-8"))
        owner.execute(f"ALTER ROLE courtvision_app PASSWORD '{SECRET}'")
    yield srv, parsed.hostname, parsed.port
    srv.cleanup()


@pytest.fixture
def owner(server):
    srv, _, _ = server
    with REAL_CONNECT(srv.get_uri(), autocommit=True) as conn:
        conn.execute("TRUNCATE public.predictions, public.predictions_recovered")
        yield conn


@pytest.fixture
def app_env(server, monkeypatch, tmp_path):
    _, host, port = server
    cert = tmp_path / "ca.crt"
    cert.write_text("test certificate placeholder")
    for key, value in {
        "HOST": host, "PORT": str(port), "NAME": "postgres",
        "USER": "courtvision_app", "PASSWORD": SECRET, "SSLROOTCERT": str(cert),
    }.items():
        monkeypatch.setenv(f"COURTVISION_DB_{key}", value)
    captured = []

    def local_connect(*args, **kwargs):
        captured.append(dict(kwargs))
        # The embedded test server has no TLS; production keeps verify-full.
        kwargs.update(sslmode="disable", sslrootcert=None)
        return REAL_CONNECT(*args, **{k: v for k, v in kwargs.items() if v is not None})

    monkeypatch.setattr(db.psycopg, "connect", local_connect)
    return captured


def _prediction(game_id="0022600101", home="Home", away="Away", home_id="1",
                away_id="2", predicted="Home", created="2026-11-20T12:00:00+00:00",
                tipoff="2026-11-21T00:00:00+00:00"):
    return pd.DataFrame([{
        "game_id": game_id, "game_date": "2026-11-20", "tipoff_utc": tipoff,
        "home_team_id": home_id, "home_team": home, "away_team_id": away_id,
        "away_team": away, "predicted_winner": predicted,
        "home_win_probability": 0.6, "away_win_probability": 0.4,
        "win_probability": 0.6, "created_at": created,
        "model_version": "test-v1", "rating_source": "test", "status": "scheduled",
    }])


# --- configuration and connection safety -------------------------------------

def test_settings_require_restricted_role_and_hide_password(tmp_path):
    cert = tmp_path / "ca.crt"
    cert.write_text("x")
    base = {"COURTVISION_DB_HOST": "h", "COURTVISION_DB_PASSWORD": SECRET,
            "COURTVISION_DB_SSLROOTCERT": str(cert)}
    with pytest.raises(db.HistoryUnavailable, match="restricted"):
        db.load_settings({**base, "COURTVISION_DB_USER": "postgres"}, secrets={})
    settings = db.load_settings(
        {**base, "COURTVISION_DB_USER": "courtvision_app.projref"}, secrets={})
    assert SECRET not in repr(settings) and SECRET not in str(settings)


def test_missing_configuration_and_certificate_fail_without_secrets(tmp_path):
    with pytest.raises(db.HistoryUnavailable, match="not configured") as info:
        db.load_settings({"COURTVISION_DB_PASSWORD": SECRET}, secrets={})
    assert SECRET not in str(info.value)
    env = {"COURTVISION_DB_HOST": "h", "COURTVISION_DB_USER": "courtvision_app",
           "COURTVISION_DB_PASSWORD": SECRET}
    with pytest.raises(db.HistoryUnavailable, match="CA certificate"):
        db.load_settings(env, secrets={})
    with pytest.raises(db.HistoryUnavailable, match="not found"):
        db.load_settings({**env, "COURTVISION_DB_SSLROOTCERT": str(tmp_path / "no.crt")},
                         secrets={})


def test_secrets_are_used_when_environment_is_empty(tmp_path):
    cert = tmp_path / "ca.crt"
    cert.write_text("x")
    settings = db.load_settings({}, secrets={
        "host": "h", "user": "courtvision_app.p", "password": SECRET,
        "sslrootcert": str(cert), "port": 5432})
    assert (settings.host, settings.port, settings.dbname) == ("h", 5432, "postgres")


def test_connection_always_requests_verify_full(app_env):
    nba.read_prediction_history()
    assert app_env and all(c["sslmode"] == "verify-full" for c in app_env)
    assert all(c["user"] == "courtvision_app" and c["sslrootcert"] for c in app_env)


def test_unreachable_database_raises_and_never_falls_back_to_sqlite(
        app_env, monkeypatch, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]
    monkeypatch.setenv("COURTVISION_DB_PORT", str(closed_port))
    monkeypatch.setattr(db, "CONNECT_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(nba, "HISTORY_PATH", tmp_path / "must_not_exist.sqlite3")
    monkeypatch.chdir(tmp_path)
    for call in (lambda: nba.read_prediction_history(),
                 lambda: nba.save_prediction_snapshots(_prediction()),
                 lambda: nba.update_prediction_results({"0022600101": "1"})):
        with pytest.raises(db.HistoryUnavailable) as info:
            call()
        assert SECRET not in str(info.value)
    assert not list(tmp_path.glob("*.sqlite3")) and not (tmp_path / "data").exists()


# --- saving ---------------------------------------------------------------------

def test_save_is_idempotent_and_history_is_plain(app_env, owner):
    assert nba.save_prediction_snapshots(_prediction()) == 1
    assert nba.save_prediction_snapshots(_prediction()) == 0
    history = nba.read_prediction_history()
    row = history.iloc[0]
    assert row["game_date"] == "2026-11-20"
    assert row["created_at"] == "2026-11-20T12:00:00+00:00"
    assert row["home_win_probability"] == 0.6 and row["model_version"] == "test-v1"
    assert row["source"] == "pregame_saved"


def test_only_pregame_scheduled_predictions_are_saved(app_env, owner):
    late = _prediction(created="2026-11-21T00:00:01+00:00")
    started = _prediction(game_id="0022600102", away="Other", home_id="3", away_id="4")
    started["status"] = "live"
    assert nba.save_prediction_snapshots(pd.concat([late, started])) == 0
    assert nba.read_prediction_history().empty


def test_duplicate_of_recovered_game_is_skipped_without_blocking_others(app_env, owner):
    owner.execute("""
        INSERT INTO public.predictions_recovered
          (game_date, away_team, home_team, predicted_winner, home_win_probability,
           away_win_probability, created_at, created_at_displayed, tipoff_displayed_utc,
           tipoff_displayed, model_version, recovered_from)
        VALUES ('2026-11-20','Away','Home','Home',0.6,0.4,'2026-11-20 12:00+00','x',
                '2026-11-21 00:00+00','x','m','test')""")
    both = pd.concat([
        _prediction(),
        _prediction(game_id="0022600103", home="Third", away="Fourth", home_id="5",
                    away_id="6", predicted="Third"),
    ])
    assert nba.save_prediction_snapshots(both) == 1
    games = set(nba.read_prediction_history()["home_team"])
    assert games == {"Home", "Third"}  # one recovered row + one new row
    assert len(nba.read_prediction_history()) == 2


def test_original_fields_cannot_be_changed_or_deleted_by_app_role(app_env, owner):
    nba.save_prediction_snapshots(_prediction())
    with db.connect() as conn:
        for sql in ("UPDATE public.predictions SET home_win_probability = 0.9",
                    "DELETE FROM public.predictions",
                    "TRUNCATE public.predictions"):
            with pytest.raises(psycopg.errors.Error):
                with conn.transaction():
                    conn.execute(sql)


# --- grading --------------------------------------------------------------------

def test_result_is_recorded_once_and_originals_are_preserved(app_env, owner):
    nba.save_prediction_snapshots(_prediction())
    before = nba.read_prediction_history().iloc[0]
    assert nba.update_prediction_results({"0022600101": "1"}) == 1
    assert nba.update_prediction_results({"0022600101": "2"}) == 0  # already graded
    after = nba.read_prediction_history().iloc[0]
    assert after["actual_winner"] == "1" and bool(after["correct"]) is True
    for column in ("predicted_winner", "home_win_probability", "created_at",
                   "model_version", "tipoff_utc"):
        assert after[column] == before[column]


def test_result_for_unknown_game_or_non_participant_is_ignored(app_env, owner):
    nba.save_prediction_snapshots(_prediction())
    assert nba.update_prediction_results({"0022609999": "1"}) == 0
    assert nba.update_prediction_results({"0022600101": "99"}) == 0
    assert nba.read_prediction_history()["actual_winner"].isna().all()


def test_result_before_game_could_have_ended_is_rejected(app_env, owner):
    tipoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    owner.execute(
        "INSERT INTO public.predictions (game_id, game_date, tipoff_utc, home_team_id,"
        " home_team, away_team_id, away_team, predicted_winner, home_win_probability,"
        " away_win_probability, win_probability, created_at, model_version, rating_source)"
        " VALUES ('0022600104','2026-11-26',%s,'1','Home','2','Away','Home',0.6,0.4,0.6,%s,'m','r')",
        (tipoff, tipoff - timedelta(hours=1)))
    assert nba.update_prediction_results({"0022600104": "1"}) == 0
    assert nba.read_prediction_history()["actual_winner"].isna().all()


# --- unified history and accuracy ----------------------------------------------

def test_unified_accuracy_spans_both_tables(app_env, owner):
    nba.save_prediction_snapshots(_prediction())                              # correct
    nba.save_prediction_snapshots(_prediction(
        game_id="0022600105", home="H2", away="A2", home_id="3", away_id="4",
        predicted="H2"))                                                      # wrong
    nba.update_prediction_results({"0022600101": "1", "0022600105": "4"})
    owner.execute("""
        INSERT INTO public.predictions_recovered
          (game_date, away_team, home_team, predicted_winner, home_win_probability,
           away_win_probability, created_at, created_at_displayed, tipoff_displayed_utc,
           tipoff_displayed, model_version, recovered_from)
        VALUES ('2026-11-20','A3','H3','A3',0.4,0.6,'2026-11-20 12:00+00','x',
                '2026-11-21 00:00+00','x','m','test'),
               ('2026-11-20','A4','H4','H4',0.6,0.4,'2026-11-20 12:00+00','x',
                '2026-11-21 00:00+00','x','m','test')""")
    ids = [r[0] for r in owner.execute(
        "SELECT recovery_id FROM public.predictions_recovered ORDER BY recovery_id")]
    owner.execute("SELECT public.verify_recovered_game(%s,'0022600106','7','8','2026-11-21 00:00+00')", (ids[0],))
    owner.execute("SELECT public.grade_recovered_prediction(%s,'8',90,100,3,'test')", (ids[0],))  # A3 wins: correct
    history = nba.read_prediction_history()
    assert set(history["source"]) == {"pregame_saved", "recovered"}
    assert len(history) == 4

    completed, accuracy = nba.prediction_accuracy(history)
    totals = nba.read_prediction_accuracy()
    assert (completed, accuracy) == (3, pytest.approx(2 / 3))
    assert (totals["total_predictions"], totals["completed"], totals["pending"],
            totals["correct"], totals["incorrect"]) == (4, 3, 1, 2, 1)
    assert totals["accuracy"] == pytest.approx(0.6667)


def test_ungraded_recovered_prediction_does_not_count_toward_accuracy(app_env, owner):
    owner.execute("""
        INSERT INTO public.predictions_recovered
          (game_date, away_team, home_team, predicted_winner, home_win_probability,
           away_win_probability, created_at, created_at_displayed, tipoff_displayed_utc,
           tipoff_displayed, model_version, recovered_from)
        VALUES ('2026-11-20','A','H','H',0.6,0.4,'2026-11-20 12:00+00','x',
                '2026-11-21 00:00+00','x','m','test')""")
    assert nba.prediction_accuracy(nba.read_prediction_history()) == (0, None)
    assert nba.read_prediction_accuracy()["completed"] == 0


# --- results refresh job -----------------------------------------------------------

def test_refresh_job_reconciles_into_postgres_with_leading_zero_ids(app_env, owner, tmp_path):
    nba.save_prediction_snapshots(_prediction())
    csv = tmp_path / "games.csv"
    pd.DataFrame([
        {"GAME_ID": "0022600101", "TEAM_ID": 1, "WL": "W", "MATCHUP": "A vs. B"},
        {"GAME_ID": "0022600101", "TEAM_ID": 2, "WL": "L", "MATCHUP": "B @ A"},
    ]).to_csv(csv, index=False)
    assert job.reconcile(csv, None) == 1
    assert job.reconcile(csv, None) == 0
