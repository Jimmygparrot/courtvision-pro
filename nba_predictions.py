from datetime import datetime, timezone
from pathlib import Path
import math
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
CURRENT_SEASON = "2026-27"
PREVIOUS_SEASON = "2025-26"
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
        "game_id": schedule["gameId"].astype(str),
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
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams

    scoreboard = scoreboardv2.ScoreboardV2(
        game_date=str(game_date),
        timeout=10,
    ).get_data_frames()[0]
    if scoreboard.empty:
        return pd.DataFrame()

    names = {str(team["id"]): team["full_name"] for team in teams.get_teams()}
    local_games = _local_schedule(schedule_path, game_date)
    local_by_id = (
        local_games.set_index("game_id").to_dict("index")
        if not local_games.empty else {}
    )
    rows = []

    for _, game in scoreboard.iterrows():
        game_id = str(game["GAME_ID"])
        home_id = str(game.get("HOME_TEAM_ID", game.get("TEAM_ID_HOME")))
        away_id = str(game.get("VISITOR_TEAM_ID", game.get("TEAM_ID_AWAY")))
        local = local_by_id.get(game_id, {})
        rows.append({
            "game_id": game_id,
            "game_date": str(game_date),
            "tipoff_utc": local.get("tipoff_utc", pd.NaT),
            "home_team_id": home_id,
            "home_team": local.get("home_team", names.get(home_id, home_id)),
            "away_team_id": away_id,
            "away_team": local.get("away_team", names.get(away_id, away_id)),
            "status": _game_status(game.get("GAME_STATUS_ID")),
        })

    return pd.DataFrame(rows)


def load_todays_matchups(game_date, schedule_path=None):
    """Prefer today's scoreboard; fall back to the saved schedule on API failure."""
    if schedule_path is None:
        schedule_path = ROOT / "data" / f"nba_schedule_{CURRENT_SEASON}.csv"

    local_games = _local_schedule(schedule_path, game_date)
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
    from nba_api.stats.endpoints import leaguegamelog

    return leaguegamelog.LeagueGameLog(
        player_or_team_abbreviation="T",
        season=CURRENT_SEASON,
        season_type_all_star="Regular Season",
        timeout=10,
    ).get_data_frames()[0]


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
    strengths = {}
    sources = {}
    for team_id in set(current_rates) | set(previous_rates):
        if team_id in current_rates:
            strengths[team_id] = current_rates[team_id]
            sources[team_id] = "current-season results through prior day"
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
            "rating_source": (
                sources.get(home_id, "neutral early-season baseline")
                if sources.get(home_id) == sources.get(away_id)
                else "mixed team-record sources"
            ),
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


def save_prediction_snapshots(predictions, history_path=HISTORY_PATH):
    if predictions is None or predictions.empty:
        return 0

    saved = 0
    with _connect(history_path) as connection:
        for row in predictions.to_dict("records"):
            created = pd.to_datetime(row["created_at"], utc=True)
            tipoff = pd.to_datetime(row.get("tipoff_utc"), utc=True, errors="coerce")
            if row.get("status") != "scheduled" or pd.isna(tipoff):
                continue
            if created >= tipoff:
                continue
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
    for game_id, rows in team_games.groupby("GAME_ID"):
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


def update_prediction_results(actual_winners, history_path=HISTORY_PATH, now=None):
    if not actual_winners:
        return 0
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


def read_prediction_history(history_path=HISTORY_PATH):
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
