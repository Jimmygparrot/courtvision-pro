from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import math
import os
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CURRENT_SEASON = "2026-27"
PREVIOUS_SEASON = "2025-26"
NBA_TEAM_COUNT = 30
MODEL_VERSION = "pregame-win-rate-logistic-cross-season-v1"
MODEL_COEFFICIENT = 4.1196
MODEL_INTERCEPT = 0.1982
HISTORY_PATH = ROOT / "data" / "nba_prediction_history.sqlite3"
EASTERN = ZoneInfo("America/New_York")

PREDICTION_COLUMNS = [
    "game_id",
    "game_date",
    "tipoff_utc",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "predicted_winner",
    "home_win_probability",
    "away_win_probability",
    "win_probability",
    "created_at",
    "model_version",
    "rating_source",
    "status",
]


def _game_status(value):
    status = str(value).strip().lower()
    if status in {"final", "completed"}:
        return "final"
    if status in {"in progress", "live"}:
        return "in_progress"
    try:
        numeric_status = int(float(status))
    except ValueError:
        numeric_status = None
    if numeric_status == 3:
        return "final"
    if numeric_status == 2:
        return "in_progress"
    return "scheduled"


def _normalize_game_id(value):
    game_id = str(value).strip()
    return game_id.zfill(10) if game_id.isdigit() else game_id


def format_eastern_datetime(value):
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "Unavailable"

    local_time = timestamp.tz_convert(EASTERN)
    hour = local_time.strftime("%I").lstrip("0") or "0"
    return (
        f"{local_time.strftime('%b')} {local_time.day}, {local_time.year}, "
        f"{hour}:{local_time.strftime('%M %p %Z')}"
    )


def _local_schedule(schedule_path, game_date):
    path = Path(schedule_path)
    if not path.exists():
        return pd.DataFrame()

    schedule = pd.read_csv(path, dtype={
        "gameId": str,
        "homeTeam_teamId": str,
        "awayTeam_teamId": str,
    })
    schedule["tipoff_utc"] = pd.to_datetime(
        schedule["gameDateTimeUTC"], utc=True, errors="coerce"
    )
    schedule["game_date"] = (
        schedule["tipoff_utc"]
        .dt.tz_convert(EASTERN)
        .dt.date.astype(str)
    )
    schedule = schedule[schedule["game_date"] == str(game_date)].copy()
    if schedule.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "game_id": schedule["gameId"].map(_normalize_game_id),
        "game_date": schedule["game_date"],
        "tipoff_utc": schedule["tipoff_utc"],
        "home_team_id": schedule["homeTeam_teamId"].astype(str),
        "home_team": (
            schedule["homeTeam_teamCity"].astype(str)
            + " " + schedule["homeTeam_teamName"].astype(str)
        ).str.strip(),
        "away_team_id": schedule["awayTeam_teamId"].astype(str),
        "away_team": (
            schedule["awayTeam_teamCity"].astype(str)
            + " " + schedule["awayTeam_teamName"].astype(str)
        ).str.strip(),
        "status": schedule["gameStatus"].map(_game_status),
    })


def _scoreboard_schedule(game_date, schedule_path):
    from nba_api.stats.endpoints import scoreboardv3
    from nba_api.stats.static import teams

    scoreboard = scoreboardv3.ScoreboardV3(
        game_date=str(game_date),
        timeout=30,
    )
    game_headers = scoreboard.game_header.get_data_frame()
    if game_headers.empty:
        return pd.DataFrame()

    teams_by_abbreviation = {
        team["abbreviation"]: team for team in teams.get_teams()
    }
    local_games = _local_schedule(schedule_path, game_date)
    local_by_id = (
        local_games.set_index("game_id").to_dict("index")
        if not local_games.empty else {}
    )
    rows = []

    for _, game in game_headers.iterrows():
        game_id = _normalize_game_id(game["gameId"])
        local = local_by_id.get(game_id, {})
        game_code = str(game.get("gameCode", "")).rsplit("/", 1)[-1]
        away_abbreviation = game_code[:3]
        home_abbreviation = game_code[3:]
        away_team = teams_by_abbreviation.get(away_abbreviation)
        home_team = teams_by_abbreviation.get(home_abbreviation)

        home_id = (
            str(home_team["id"]) if home_team else local.get("home_team_id")
        )
        away_id = (
            str(away_team["id"]) if away_team else local.get("away_team_id")
        )
        if home_id is None or away_id is None:
            raise ValueError(
                f"Unable to map NBA scoreboard game {game_id} to its teams."
            )

        tipoff = pd.to_datetime(
            game.get("gameTimeUTC"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(tipoff):
            tipoff = pd.to_datetime(
                local.get("tipoff_utc"),
                utc=True,
                errors="coerce",
            )

        rows.append({
            "game_id": game_id,
            "game_date": str(game_date),
            "tipoff_utc": tipoff,
            "home_team_id": home_id,
            "home_team": (
                local.get("home_team")
                or (home_team["full_name"] if home_team else home_id)
            ),
            "away_team_id": away_id,
            "away_team": (
                local.get("away_team")
                or (away_team["full_name"] if away_team else away_id)
            ),
            "status": _game_status(game.get("gameStatus")),
        })

    return pd.DataFrame(rows)


def load_todays_matchups(game_date, schedule_path=None):
    """Prefer today's scoreboard; fall back to the saved schedule on API failure."""
    if schedule_path is None:
        schedule_path = ROOT / "data" / f"nba_schedule_{CURRENT_SEASON}.csv"

    local_games = _local_schedule(schedule_path, game_date)
    if not local_games.empty:
        return local_games, "saved schedule", None
    try:
        live_games = _scoreboard_schedule(game_date, schedule_path)
        if not live_games.empty:
            return live_games, "NBA scoreboard", None
        if local_games.empty:
            return live_games, "NBA scoreboard", None
        return (
            local_games,
            "saved schedule",
            "No games were returned by the NBA scoreboard; using the saved schedule.",
        )
    except Exception as error:
        if not local_games.empty:
            return (
                local_games,
                "saved schedule",
                "The NBA schedule refresh failed "
                f"({type(error).__name__}); using the saved schedule.",
            )
        return (
            pd.DataFrame(),
            "unavailable",
            "Today's NBA schedule could not be loaded "
            f"({type(error).__name__}).",
        )

def load_current_season_team_games():
    """Load fresh current-season results or return a dated stale snapshot."""
    snapshot = ROOT / "data" / "historical" / f"nba_games_{CURRENT_SEASON}.csv"
    snapshot_games = None
    snapshot_latest_date = None

    if snapshot.exists():
        try:
            snapshot_games, snapshot_latest_date = _prepare_team_results(
                pd.read_csv(snapshot)
            )
            logger.info(
                "Loaded current-season team snapshot: path=%s rows=%d latest=%s",
                snapshot,
                len(snapshot_games),
                snapshot_latest_date,
            )
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
        ) as error:
            logger.warning(
                "Rejected current-season team snapshot at %s (%s: %s)",
                snapshot,
                type(error).__name__,
                error,
                exc_info=True,
            )
            snapshot_games = None
            snapshot_latest_date = None
    else:
        logger.error(
            "Current-season team snapshot is missing: path=%s",
            snapshot,
        )

    if snapshot_games is not None:
        age = _today_eastern() - snapshot_latest_date
        if timedelta(0) <= age <= timedelta(days=1):
            logger.info(
                "Using fresh current-season team snapshot through %s",
                snapshot_latest_date,
            )
            return _set_team_results_metadata(
                snapshot_games,
                source="saved current-season snapshot",
                latest_game_date=snapshot_latest_date,
                is_stale=False,
            )

    logger.info(
        "Refreshing current-season team results from the NBA API; "
        "valid_snapshot=%s",
        snapshot_games is not None,
    )
    from nba_api.stats.endpoints import leaguegamelog

    try:
        api_games = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="T",
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            timeout=30,
        ).get_data_frames()[0]
        api_games, api_latest_date = _prepare_team_results(api_games)
        return _set_team_results_metadata(
            api_games,
            source="NBA API",
            latest_game_date=api_latest_date,
            is_stale=False,
        )
    except Exception as error:
        if snapshot_games is None:
            logger.error(
                "NBA API refresh failed and no valid current-season snapshot "
                "is available at %s (%s: %s)",
                snapshot,
                type(error).__name__,
                error,
                exc_info=True,
            )
            raise RuntimeError(
                "Current-season NBA team results are unavailable "
                f"({type(error).__name__}); no valid saved snapshot exists."
            ) from error

        latest_date = snapshot_latest_date.strftime("%b %d, %Y").replace(
            " 0", " "
        )
        logger.warning(
            "NBA API refresh failed (%s: %s); using current-season team "
            "snapshot through %s at %s",
            type(error).__name__,
            error,
            latest_date,
            snapshot,
            exc_info=True,
        )
        warning = (
            "The NBA API refresh failed "
            f"({type(error).__name__}); using the saved current-season "
            f"snapshot through {latest_date}. The snapshot may be outdated."
        )
        return _set_team_results_metadata(
            snapshot_games,
            source="saved current-season snapshot",
            latest_game_date=snapshot_latest_date,
            is_stale=True,
            warning=warning,
        )


def _prepare_team_results(games):
    required = {"TEAM_ID", "GAME_DATE", "WL"}
    if games is None or games.empty or not required.issubset(games.columns):
        raise ValueError(
            "NBA team results are empty or missing required columns."
        )

    results = games[games["WL"].isin(["W", "L"])].copy()
    dates = pd.to_datetime(
        results["GAME_DATE"],
        format="mixed",
        errors="coerce",
    )
    valid_dates = dates.notna()
    results = results.loc[valid_dates].copy()
    if results.empty:
        raise ValueError("NBA team results contain no valid completed games.")

    latest_game_date = dates.loc[valid_dates].max().date()
    return results, latest_game_date


def _validate_complete_team_results(games):
    required = {"GAME_ID", "TEAM_ID", "GAME_DATE", "WL"}
    if games is None or games.empty or not required.issubset(games.columns):
        raise ValueError(
            "Team results are empty or missing required game identifiers."
        )

    win_loss = games["WL"].fillna("").astype(str).str.strip().str.upper()
    invalid_outcomes = ~win_loss.isin(["", "W", "L"])
    if invalid_outcomes.any():
        raise ValueError("Team results contain invalid win/loss outcomes.")

    completed = games.loc[win_loss.isin(["W", "L"])].copy()
    if completed.empty:
        raise ValueError("Team results contain no completed games.")

    results, latest_game_date = _prepare_team_results(completed)
    if len(results) != len(completed):
        raise ValueError("Completed team results contain invalid game dates.")

    dates = pd.to_datetime(
        results["GAME_DATE"],
        format="mixed",
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("Completed team results contain invalid game dates.")
    if latest_game_date > _today_eastern():
        raise ValueError("Team results contain a game date in the future.")

    team_ids = pd.to_numeric(results["TEAM_ID"], errors="coerce")
    if (
        team_ids.isna().any()
        or (team_ids <= 0).any()
        or (team_ids % 1 != 0).any()
    ):
        raise ValueError("Team results contain invalid team IDs.")

    game_ids = pd.to_numeric(results["GAME_ID"], errors="coerce")
    if (
        game_ids.isna().any()
        or (game_ids <= 0).any()
        or (game_ids % 1 != 0).any()
    ):
        raise ValueError("Team results contain invalid game IDs.")

    results["TEAM_ID"] = team_ids.astype("int64")
    results["GAME_ID"] = game_ids.astype("int64").map(
        lambda game_id: f"{game_id:010d}"
    )
    results["GAME_DATE"] = dates.dt.strftime("%Y-%m-%d")
    results["WL"] = win_loss.loc[results.index]

    if results.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Team results contain duplicate team-game rows.")
    if results["TEAM_ID"].nunique() != NBA_TEAM_COUNT:
        raise ValueError("Team results do not cover all 30 NBA teams.")

    for game_id, rows in results.groupby("GAME_ID", sort=False):
        if (
            len(rows) != 2
            or rows["TEAM_ID"].nunique() != 2
            or sorted(rows["WL"].tolist()) != ["L", "W"]
            or rows["GAME_DATE"].nunique() != 1
        ):
            raise ValueError(
                f"Completed game {game_id} has incomplete or inconsistent "
                "team results."
            )

    return results, latest_game_date


def refresh_current_season_team_snapshot(snapshot_path=None):
    """Fetch and atomically replace the current-season results snapshot."""
    if snapshot_path is None:
        snapshot_path = (
            ROOT
            / "data"
            / "historical"
            / f"nba_games_{CURRENT_SEASON}.csv"
        )
    snapshot_path = Path(snapshot_path)

    from nba_api.stats.endpoints import leaguegamelog

    logger.info("Refreshing current-season team snapshot at %s", snapshot_path)
    try:
        api_games = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="T",
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            timeout=60,
        ).get_data_frames()[0]
        refreshed, latest_game_date = _validate_complete_team_results(
            api_games
        )
    except Exception:
        logger.exception(
            "Current-season refresh failed; leaving the existing snapshot "
            "untouched at %s",
            snapshot_path,
        )
        raise

    if snapshot_path.exists():
        try:
            existing, existing_latest_date = _validate_complete_team_results(
                pd.read_csv(snapshot_path)
            )
        except (
            OSError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
        ) as error:
            logger.warning(
                "Existing team snapshot is invalid and will only be replaced "
                "after a validated API refresh (%s: %s)",
                type(error).__name__,
                error,
            )
        else:
            if latest_game_date < existing_latest_date:
                raise ValueError(
                    "Refreshed team results are older than the current "
                    "snapshot."
                )

            old_pairs = set(zip(existing["GAME_ID"], existing["TEAM_ID"]))
            new_pairs = set(zip(refreshed["GAME_ID"], refreshed["TEAM_ID"]))
            if not old_pairs.issubset(new_pairs):
                raise ValueError(
                    "Refreshed team results omit previously saved "
                    "team-game records."
                )

    refreshed = refreshed.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ID"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=snapshot_path.parent,
            prefix=f".{snapshot_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            refreshed.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, snapshot_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Updated current-season team snapshot: path=%s rows=%d latest=%s",
        snapshot_path,
        len(refreshed),
        latest_game_date,
    )
    return latest_game_date


def _set_team_results_metadata(
    games,
    source,
    latest_game_date,
    is_stale,
    warning=None,
):
    games.attrs["source"] = source
    games.attrs["latest_game_date"] = latest_game_date.isoformat()
    games.attrs["is_stale"] = is_stale
    games.attrs["warning"] = warning
    return games


def _today_eastern():
    return datetime.now(EASTERN).date()



def load_previous_season_team_games(path=None):
    if path is None:
        path = ROOT / "data" / "historical" / f"nba_games_{PREVIOUS_SEASON}.csv"
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["TEAM_ID", "GAME_DATE", "WL"])
    return pd.read_csv(path, usecols=["TEAM_ID", "GAME_DATE", "WL"])


def team_strengths_for_date(current_games, previous_games, target_date):
    """Return smoothed team win rates using only games before target_date."""
    target_date = pd.Timestamp(target_date).date()

    def rates(games):
        if games is None or games.empty:
            return {}
        required = {"TEAM_ID", "GAME_DATE", "WL"}
        if not required.issubset(games.columns):
            return {}

        data = games[list(required)].copy()
        data["TEAM_ID"] = data["TEAM_ID"].astype(str)
        data["GAME_DATE"] = pd.to_datetime(
            data["GAME_DATE"],
            format="mixed",
            errors="coerce",
        ).dt.date
        data = data[
            data["GAME_DATE"].notna()
            & (data["GAME_DATE"] < target_date)
            & data["WL"].isin(["W", "L"])
        ]
        if data.empty:
            return {}

        summary = data.assign(WIN=data["WL"].eq("W")).groupby("TEAM_ID")[
            "WIN"
        ].agg(["sum", "count"])
        return {
            team_id: (int(row["sum"]) + 5) / (int(row["count"]) + 10)
            for team_id, row in summary.iterrows()
        }

    current_rates = rates(current_games)
    previous_rates = rates(previous_games)
    current_is_stale = bool(
        current_games is not None
        and current_games.attrs.get("is_stale", False)
    )
    current_latest_date = (
        current_games.attrs.get("latest_game_date")
        if current_games is not None else None
    )
    current_source = "current-season results through prior day"
    if current_is_stale:
        latest_date = (
            pd.Timestamp(current_latest_date).strftime("%b %d, %Y").replace(
                " 0", " "
            )
            if current_latest_date else "an unknown date"
        )
        current_source = (
            f"outdated current-season snapshot through {latest_date}"
        )

    strengths = {}
    sources = {}
    for team_id in set(current_rates) | set(previous_rates):
        if team_id in current_rates:
            strengths[team_id] = current_rates[team_id]
            sources[team_id] = current_source
        else:
            strengths[team_id] = previous_rates[team_id]
            sources[team_id] = "previous-season baseline"
    return strengths, sources


def _probability(home_strength, away_strength):
    z = MODEL_INTERCEPT + MODEL_COEFFICIENT * (
        home_strength - away_strength
    )
    return 1.0 / (1.0 + math.exp(-z))


def build_pregame_predictions(
    games,
    current_team_games,
    previous_team_games,
    now=None,
):
    """Predict only games that are still scheduled, from prior-date records."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    rows = []

    if games is None or games.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    for _, game in games.iterrows():
        if _game_status(game.get("status", "scheduled")) != "scheduled":
            continue
        tipoff = pd.to_datetime(game.get("tipoff_utc"), utc=True, errors="coerce")
        if pd.isna(tipoff) or tipoff <= pd.Timestamp(now):
            continue

        game_date = str(game["game_date"])
        strengths, sources = team_strengths_for_date(
            current_team_games,
            previous_team_games,
            game_date,
        )
        home_id = str(game["home_team_id"])
        away_id = str(game["away_team_id"])
        home_strength = strengths.get(home_id, 0.5)
        away_strength = strengths.get(away_id, 0.5)
        home_probability = _probability(home_strength, away_strength)
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_source = sources.get(
            home_id,
            "neutral early-season baseline",
        )
        away_source = sources.get(
            away_id,
            "neutral early-season baseline",
        )
        if home_source == away_source:
            rating_source = home_source
        else:
            stale_sources = sorted({
                source
                for source in (home_source, away_source)
                if source.startswith("outdated current-season snapshot")
            })
            rating_source = "mixed team-record sources"
            if stale_sources:
                rating_source += "; includes " + ", ".join(stale_sources)

        rows.append({
            "game_id": str(game["game_id"]),
            "game_date": game_date,
            "tipoff_utc": (
                tipoff.isoformat() if pd.notna(tipoff) else None
            ),
            "home_team_id": home_id,
            "home_team": home_team,
            "away_team_id": away_id,
            "away_team": away_team,
            "predicted_winner": (
                home_team if home_probability >= 0.5 else away_team
            ),
            "home_win_probability": home_probability,
            "away_win_probability": 1.0 - home_probability,
            "win_probability": max(home_probability, 1.0 - home_probability),
            "created_at": now.isoformat(),
            "model_version": MODEL_VERSION,
            "rating_source": rating_source,
            "status": "scheduled",
        })

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _connect(history_path):
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            game_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            tipoff_utc TEXT,
            home_team_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            away_team TEXT NOT NULL,
            predicted_winner TEXT NOT NULL,
            home_win_probability REAL NOT NULL,
            away_win_probability REAL NOT NULL,
            win_probability REAL NOT NULL,
            created_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            rating_source TEXT NOT NULL,
            actual_winner TEXT,
            completed_at TEXT
        )
        """
    )
    return connection


def _pregame_rows(predictions):
    """Rows eligible to be saved: scheduled, with a tipoff after creation."""
    for row in predictions.to_dict("records"):
        created = pd.to_datetime(row["created_at"], utc=True)
        tipoff = pd.to_datetime(row.get("tipoff_utc"), utc=True, errors="coerce")
        if row.get("status") != "scheduled" or pd.isna(tipoff):
            continue
        if created >= tipoff:
            continue
        yield row


def save_prediction_snapshots(predictions, history_path=None):
    """Save pregame predictions.

    With no ``history_path`` the shared PostgreSQL history is used and a
    failure raises ``db.HistoryUnavailable``. An explicit ``history_path``
    selects a SQLite file (tests and migration tools only).
    """
    if predictions is None or predictions.empty:
        return 0

    if history_path is None:
        import db

        return db.save_predictions([
            (
                row["game_id"], row["game_date"], row.get("tipoff_utc"),
                row["home_team_id"], row["home_team"],
                row["away_team_id"], row["away_team"], row["predicted_winner"],
                float(row["home_win_probability"]),
                float(row["away_win_probability"]),
                float(row["win_probability"]),
                row["created_at"], row["model_version"], row["rating_source"],
            )
            for row in _pregame_rows(predictions)
        ])

    saved = 0
    with _connect(history_path) as connection:
        for row in _pregame_rows(predictions):
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO predictions (
                    game_id, game_date, tipoff_utc, home_team_id, home_team,
                    away_team_id, away_team, predicted_winner,
                    home_win_probability, away_win_probability,
                    win_probability, created_at, model_version, rating_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["game_id"],
                    row["game_date"],
                    row.get("tipoff_utc"),
                    row["home_team_id"],
                    row["home_team"],
                    row["away_team_id"],
                    row["away_team"],
                    row["predicted_winner"],
                    float(row["home_win_probability"]),
                    float(row["away_win_probability"]),
                    float(row["win_probability"]),
                    row["created_at"],
                    row["model_version"],
                    row["rating_source"],
                ),
            )
            saved += cursor.rowcount
    return saved


def extract_completed_results(team_games):
    """Map completed games to winner IDs; incomplete games are ignored."""
    required = {"GAME_ID", "TEAM_ID", "WL", "MATCHUP"}
    if team_games is None or team_games.empty or not required.issubset(
        team_games.columns
    ):
        return {}

    results = {}
    # The saved snapshot is read from CSV, where GAME_ID loses its leading zeros.
    game_ids = team_games["GAME_ID"].map(_normalize_game_id)
    for game_id, rows in team_games.groupby(game_ids):
        if (
            len(rows) != 2
            or rows["TEAM_ID"].nunique() != 2
            or not rows["WL"].isin(["W", "L"]).all()
        ):
            continue
        home_rows = rows[rows["MATCHUP"].astype(str).str.contains(r"\bvs\.", regex=True)]
        winners = rows[rows["WL"] == "W"]
        if len(home_rows) == 1 and len(winners) == 1:
            results[str(game_id)] = str(winners.iloc[0]["TEAM_ID"])
    return results


def update_prediction_results(actual_winners, history_path=None, now=None):
    """Record winners of confirmed-final games (see ``extract_completed_results``)."""
    if not actual_winners:
        return 0
    if history_path is None:
        import db

        return db.record_results(actual_winners, completed_at=now)
    if now is None:
        now = datetime.now(timezone.utc).isoformat()
    updated = 0
    with _connect(history_path) as connection:
        for game_id, winner_id in actual_winners.items():
            cursor = connection.execute(
                """
                UPDATE predictions
                SET actual_winner = ?, completed_at = ?
                WHERE game_id = ? AND actual_winner IS NULL
                  AND (home_team_id = ? OR away_team_id = ?)
                """,
                (str(winner_id), now, str(game_id), str(winner_id), str(winner_id)),
            )
            updated += cursor.rowcount
    return updated


def read_prediction_accuracy():
    """Cumulative totals from the unified PostgreSQL accuracy view."""
    import db

    return db.read_accuracy()


def read_prediction_history(history_path=None):
    if history_path is None:
        import db

        return db.read_history()
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame()
    with _connect(path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY created_at DESC", connection
        )


def prediction_accuracy(history):
    completed = history[
        history["actual_winner"].notna()
    ] if not history.empty else history
    if completed.empty:
        return 0, None
    correct = (
        completed["predicted_winner"]
        == completed.apply(
            lambda row: (
                row["home_team"]
                if str(row["actual_winner"]) == str(row["home_team_id"])
                else row["away_team"]
            ),
            axis=1,
        )
    )
    return len(completed), float(correct.mean())
