"""PostgreSQL access for the cumulative prediction history.

Connects as the restricted ``courtvision_app`` role through the Supabase session
pooler with ``sslmode=verify-full``. Credentials come from environment variables
(local) or Streamlit secrets (cloud) and are never logged. There is no SQLite
fallback: any connection or database failure raises ``HistoryUnavailable``.

Environment variable (or ``[postgres]`` secret key):
    COURTVISION_DB_HOST          (host)         pooler host name
    COURTVISION_DB_PORT          (port)         default 5432 (session pooler)
    COURTVISION_DB_NAME          (dbname)       default postgres
    COURTVISION_DB_USER          (user)         courtvision_app[.<project-ref>]
    COURTVISION_DB_PASSWORD      (password)
    COURTVISION_DB_SSLROOTCERT   (sslrootcert)  path to the CA certificate, or "system"
    COURTVISION_DB_SSLROOTCERT_PEM (sslrootcert_pem)  CA certificate text (alternative)
"""
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import errors
from psycopg.rows import dict_row

logger = logging.getLogger("courtvision.db")

APP_ROLE = "courtvision_app"
ENV_PREFIX = "COURTVISION_DB_"
CONNECT_TIMEOUT_SECONDS = 10
DUPLICATE_MESSAGE = "duplicate game across predictions and predictions_recovered"


class HistoryUnavailable(RuntimeError):
    """The PostgreSQL prediction history cannot be used right now."""


@dataclass(frozen=True)
class DbSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str = field(repr=False)
    sslrootcert: str

    def __repr__(self):
        return (
            f"DbSettings(host={self.host!r}, port={self.port}, "
            f"dbname={self.dbname!r}, user={self.user!r}, password=<hidden>)"
        )


def _streamlit_secrets():
    try:
        import streamlit as st

        return dict(st.secrets.get("postgres", {}))
    except Exception:
        return {}


def _write_pem(pem):
    digest = hashlib.sha256(pem.encode("utf-8")).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"courtvision_pg_root_{digest}.crt"
    if not path.exists() or path.read_text(encoding="utf-8") != pem:
        path.write_text(pem, encoding="utf-8")
    return str(path)


def load_settings(environ=None, secrets=None):
    """Build validated settings; error messages never include secret values."""
    environ = os.environ if environ is None else environ
    secrets = _streamlit_secrets() if secrets is None else secrets

    def get(key):
        value = environ.get(ENV_PREFIX + key.upper())
        if value in (None, ""):
            value = secrets.get(key)
        return None if value in (None, "") else str(value).strip()

    missing = [k for k in ("host", "user", "password") if not get(k)]
    if missing:
        raise HistoryUnavailable(
            "Prediction history database is not configured "
            f"(missing: {', '.join(missing)})."
        )

    user = get("user")
    if user != APP_ROLE and not user.startswith(APP_ROLE + "."):
        raise HistoryUnavailable(
            f"Refusing to connect: the database user must be the restricted "
            f"'{APP_ROLE}' role."
        )

    try:
        port = int(get("port") or 5432)
    except ValueError:
        raise HistoryUnavailable("Database port must be a number.") from None

    rootcert = get("sslrootcert")
    pem = get("sslrootcert_pem")
    if not rootcert and pem:
        rootcert = _write_pem(pem)
    if not rootcert:
        raise HistoryUnavailable(
            "A CA certificate is required for verify-full TLS "
            "(set sslrootcert or sslrootcert_pem)."
        )
    if rootcert != "system" and not Path(rootcert).is_file():
        raise HistoryUnavailable("The configured CA certificate file was not found.")

    return DbSettings(
        host=get("host"),
        port=port,
        dbname=get("dbname") or "postgres",
        user=user,
        password=get("password"),
        sslrootcert=rootcert,
    )


def connect(settings=None):
    settings = settings or load_settings()
    try:
        return psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password,
            sslmode="verify-full",
            sslrootcert=settings.sslrootcert,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            application_name="courtvision-pro",
            prepare_threshold=None,
            row_factory=dict_row,
        )
    except psycopg.Error as error:
        logger.warning("Database connection failed: %s", type(error).__name__)
        raise HistoryUnavailable(
            "Could not connect to the prediction history database."
        ) from None


def _fail(error):
    logger.warning("Database operation failed: %s (%s)", type(error).__name__,
                   getattr(error, "sqlstate", None))
    raise HistoryUnavailable(
        "The prediction history database returned an error."
    ) from None


INSERT_SQL = """
    INSERT INTO public.predictions (
        game_id, game_date, tipoff_utc, home_team_id, home_team,
        away_team_id, away_team, predicted_winner,
        home_win_probability, away_win_probability,
        win_probability, created_at, model_version, rating_source
    ) VALUES (%s, %s::date, %s::timestamptz, %s, %s, %s, %s, %s, %s, %s, %s,
              %s::timestamptz, %s, %s)
    ON CONFLICT DO NOTHING
"""

RESULT_SQL = """
    UPDATE public.predictions
    SET actual_winner = %s, completed_at = COALESCE(%s::timestamptz, now())
    WHERE game_id = %s AND actual_winner IS NULL
      AND (home_team_id = %s OR away_team_id = %s)
"""


def save_predictions(rows):
    """Insert pregame predictions; returns the number newly saved.

    Each row commits on its own, so a duplicate game (same game ID, same
    matchup, or already present as a recovered prediction) is skipped without
    affecting the others.
    """
    saved = 0
    try:
        with connect() as connection:
            for params in rows:
                try:
                    with connection.transaction():
                        saved += connection.execute(INSERT_SQL, params).rowcount
                except errors.RaiseException as error:
                    if DUPLICATE_MESSAGE not in str(error):
                        raise
                    logger.info("Skipped duplicate game %s.", params[0])
    except psycopg.Error as error:
        _fail(error)
    return saved


def record_results(actual_winners, completed_at=None):
    """Record winners of confirmed-final games; returns rows updated.

    The database allows a result only on an unresolved row, only for one of its
    two teams, and only at least 90 minutes after tipoff.
    """
    updated = 0
    try:
        with connect() as connection:
            for game_id, winner_id in actual_winners.items():
                params = (str(winner_id), completed_at, str(game_id),
                          str(winner_id), str(winner_id))
                try:
                    with connection.transaction():
                        updated += connection.execute(RESULT_SQL, params).rowcount
                except errors.CheckViolation:
                    logger.warning("Result for game %s rejected as too early.", game_id)
    except psycopg.Error as error:
        _fail(error)
    return updated


def _plain(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def read_history():
    """Unified history (saved pregame + recovered), newest first."""
    try:
        with connect() as connection:
            rows = connection.execute(
                "SELECT * FROM public.prediction_history_all ORDER BY created_at DESC"
            ).fetchall()
    except psycopg.Error as error:
        _fail(error)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{k: _plain(v) for k, v in row.items()} for row in rows])


def read_accuracy():
    """Cumulative totals from the unified accuracy view."""
    try:
        with connect() as connection:
            row = connection.execute("SELECT * FROM public.prediction_accuracy").fetchone()
    except psycopg.Error as error:
        _fail(error)
    return {k: _plain(v) for k, v in row.items()}
