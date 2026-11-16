from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

from nba_api.stats.endpoints import playergamelog


CURRENT_SEASON = "2026-27"
REFRESH_HOURS = 1


def cache_path(player_id, season):
    return (
        Path(__file__).resolve().parent
        / "data"
        / "nba_cache"
        / season
        / f"{player_id}.csv"
    )


def read_cached_games(player_id, season):
    path = cache_path(player_id, season)

    if path.exists():
        return pd.read_csv(path)

    return None


def get_player_games(player_id, season="2025-26"):
    path = cache_path(player_id, season)
    cached_games = read_cached_games(player_id, season)

    # Historical seasons do not need refreshing.
    if season != CURRENT_SEASON and cached_games is not None:
        return cached_games

    # Reuse recent current-season data.
    if season == CURRENT_SEASON and cached_games is not None:
        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc
        )

        age = datetime.now(timezone.utc) - modified

        if age < timedelta(hours=REFRESH_HOURS):
            return cached_games

    # Download updated statistics from the NBA API.
    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=10
        )

        games = game_log.get_data_frames()[0]

        # Save successfully downloaded results.
        path.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(path, index=False)

        return games

    except Exception as error:
        print(f"NBA API refresh failed for player {player_id}, season {season}: {error}", flush=True)    
        # Keep CourtVision running if the NBA API fails.
        if cached_games is not None:
            return cached_games

        raise RuntimeError(
            "NBA statistics could not be downloaded. Please retry."
        ) from error


def get_player_id(player_name):
    """
    Find an NBA player ID, ignoring accents and capitalization.
    """
    import unicodedata

    def normalize(name):
        name = unicodedata.normalize("NFKD", str(name))
        return "".join(
            char for char in name
            if not unicodedata.combining(char)
        ).casefold().strip()

    players_df = pd.read_csv("nba_players.csv")
    target = normalize(player_name)

    matches = players_df[
        players_df["full_name"].apply(normalize) == target
    ]

    if matches.empty:
        return None

    return int(matches.iloc[0]["id"])
