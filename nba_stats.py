import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelog


CURRENT_SEASON = "2026-27"
REFRESH_HOURS = 6
RETRY_MINUTES = 5

_refresh_lock = threading.Lock()
_failed_refresh_until = {}
_empty_refresh_until = {}


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
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None

    return None


def _cache_is_fresh(path):
    if not path.exists():
        return False

    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - modified < timedelta(hours=REFRESH_HOURS)


def get_player_games(player_id, season="2025-26", force_refresh=False):
    path = cache_path(player_id, season)
    cached_games = read_cached_games(player_id, season)

    if season != CURRENT_SEASON and cached_games is not None:
        return cached_games

    if not force_refresh and cached_games is not None and _cache_is_fresh(path):
        return cached_games

    cache_key = (str(player_id), season)
    now = datetime.now(timezone.utc)
    if not force_refresh and _empty_refresh_until.get(cache_key, now) > now:
        return cached_games if cached_games is not None else pd.DataFrame()
    if not force_refresh and _failed_refresh_until.get(cache_key, now) > now:
        if cached_games is not None:
            cached_games.attrs["refresh_error"] = (
                "The NBA statistics refresh recently failed; showing the "
                "last successfully saved game log."
            )
            return cached_games
        raise RuntimeError(
            "NBA statistics could not be downloaded. Please retry."
        )

    # Re-check freshness after acquiring the lock so concurrent sessions don't
    # make duplicate requests for the same player and season.
    with _refresh_lock:
        cached_games = read_cached_games(player_id, season)
        if (
            not force_refresh
            and cached_games is not None
            and _cache_is_fresh(path)
        ):
            return cached_games
        now = datetime.now(timezone.utc)
        if not force_refresh and _empty_refresh_until.get(cache_key, now) > now:
            return cached_games if cached_games is not None else pd.DataFrame()
        if not force_refresh and _failed_refresh_until.get(cache_key, now) > now:
            if cached_games is not None:
                cached_games.attrs["refresh_error"] = (
                    "The NBA statistics refresh recently failed; showing the "
                    "last successfully saved game log."
                )
                return cached_games
            raise RuntimeError(
                "NBA statistics could not be downloaded. Please retry."
            )

        try:
            game_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=10,
            )
            games = game_log.get_data_frames()[0]

            if games.empty:
                _empty_refresh_until[cache_key] = (
                    datetime.now(timezone.utc)
                    + timedelta(hours=REFRESH_HOURS)
                )
                if cached_games is not None and not cached_games.empty:
                    cached_games.attrs["refresh_error"] = None
                    return cached_games
                games.attrs["refresh_error"] = None
                return games

            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.tmp")
            try:
                games.to_csv(temporary_path, index=False)
                os.replace(temporary_path, path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            _failed_refresh_until.pop(cache_key, None)
            _empty_refresh_until.pop(cache_key, None)
            games.attrs["refresh_error"] = None
            return games

        except Exception as error:
            _empty_refresh_until.pop(cache_key, None)
            _failed_refresh_until[cache_key] = (
                datetime.now(timezone.utc) + timedelta(minutes=RETRY_MINUTES)
            )
            print(
                f"NBA API refresh failed for player {player_id}, "
                f"season {season}: {error}",
                flush=True,
            )
            if cached_games is not None:
                cached_games.attrs["refresh_error"] = (
                    "NBA statistics could not be refreshed. Showing the "
                    "last successfully saved game log."
                )
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
