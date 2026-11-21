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
_validated_at = {}


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


def _validation_is_fresh(cache_key, now):
    validated_at = _validated_at.get(cache_key)
    return (
        validated_at is not None
        and now - validated_at < timedelta(hours=REFRESH_HOURS)
    )


def _game_id_column(games):
    for column in ("Game_ID", "GAME_ID"):
        if column in games.columns:
            return column
    return None


def _latest_game_marker(games):
    if games is None or games.empty or "GAME_DATE" not in games.columns:
        return None

    dates = pd.to_datetime(
        games["GAME_DATE"],
        format="mixed",
        errors="coerce",
    )
    valid_dates = dates.notna()
    if not valid_dates.any():
        return None

    latest_date = dates[valid_dates].max().date()
    latest_rows = games.loc[valid_dates & dates.dt.date.eq(latest_date)]
    game_id_column = _game_id_column(games)
    latest_game_ids = (
        tuple(sorted(latest_rows[game_id_column].dropna().astype(str).unique()))
        if game_id_column else ()
    )
    return latest_date, latest_game_ids


def _merge_game_logs(cached_games, refreshed_games):
    cached = cached_games.copy()
    refreshed = refreshed_games.copy()
    cached_id_column = _game_id_column(cached)
    refreshed_id_column = _game_id_column(refreshed)

    if cached_id_column and refreshed_id_column:
        if cached_id_column != refreshed_id_column:
            refreshed = refreshed.rename(
                columns={refreshed_id_column: cached_id_column}
            )
        game_id_column = cached_id_column
    else:
        game_id_column = cached_id_column or refreshed_id_column

    combined = pd.concat(
        [cached, refreshed],
        ignore_index=True,
        sort=False,
    )

    if game_id_column:
        has_game_id = combined[game_id_column].notna()
        identified = combined.loc[has_game_id].drop_duplicates(
            subset=[game_id_column],
            keep="last",
        )
        unidentified = combined.loc[~has_game_id]
        combined = pd.concat(
            [identified, unidentified],
            ignore_index=True,
            sort=False,
        )
    elif {"GAME_DATE", "MATCHUP"}.issubset(combined.columns):
        has_game_key = combined[["GAME_DATE", "MATCHUP"]].notna().all(axis=1)
        identified = combined.loc[has_game_key].drop_duplicates(
            subset=["GAME_DATE", "MATCHUP"],
            keep="last",
        )
        unidentified = combined.loc[~has_game_key]
        combined = pd.concat(
            [identified, unidentified],
            ignore_index=True,
            sort=False,
        )

    if "GAME_DATE" in combined.columns:
        sort_order = pd.to_datetime(
            combined["GAME_DATE"],
            format="mixed",
            errors="coerce",
        ).sort_values(
            ascending=False,
            na_position="last",
            kind="stable",
        ).index
        combined = combined.iloc[sort_order]

    return combined.reset_index(drop=True)


def _cache_with_warning(games, message):
    if games is not None:
        games.attrs["refresh_error"] = message
    return games


def get_player_games(player_id, season="2025-26", force_refresh=False):
    path = cache_path(player_id, season)
    cached_games = read_cached_games(player_id, season)

    if season != CURRENT_SEASON and cached_games is not None:
        return cached_games

    cache_key = (str(player_id), season)
    now = datetime.now(timezone.utc)
    if (
        not force_refresh
        and cached_games is not None
        and _validation_is_fresh(cache_key, now)
    ):
        return cached_games

    if not force_refresh and _empty_refresh_until.get(cache_key, now) > now:
        return _cache_with_warning(
            cached_games if cached_games is not None else pd.DataFrame(),
            "The NBA statistics source returned no current-season games; "
            "showing the last saved game log.",
        )
    if not force_refresh and _failed_refresh_until.get(cache_key, now) > now:
        if cached_games is not None:
            return _cache_with_warning(
                cached_games,
                "The NBA statistics refresh recently failed; showing the "
                "last successfully saved game log.",
            )
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
            and _validation_is_fresh(
                cache_key,
                datetime.now(timezone.utc),
            )
        ):
            return cached_games
        now = datetime.now(timezone.utc)
        if not force_refresh and _empty_refresh_until.get(cache_key, now) > now:
            return _cache_with_warning(
                cached_games if cached_games is not None else pd.DataFrame(),
                "The NBA statistics source returned no current-season games; "
                "showing the last saved game log.",
            )
        if not force_refresh and _failed_refresh_until.get(cache_key, now) > now:
            if cached_games is not None:
                return _cache_with_warning(
                    cached_games,
                    "The NBA statistics refresh recently failed; showing the "
                    "last successfully saved game log.",
                )
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
                _validated_at.pop(cache_key, None)
                _empty_refresh_until[cache_key] = (
                    datetime.now(timezone.utc)
                    + timedelta(hours=REFRESH_HOURS)
                )
                if cached_games is not None and not cached_games.empty:
                    return _cache_with_warning(
                        cached_games,
                        "The NBA statistics source returned no current-season "
                        "games; showing the last saved game log.",
                    )
                games.attrs["refresh_error"] = None
                return games

            refreshed_marker = _latest_game_marker(games)
            if refreshed_marker is None:
                raise ValueError(
                    "NBA statistics response did not contain valid game dates."
                )

            cached_marker = _latest_game_marker(cached_games)
            if cached_games is not None and cached_marker == refreshed_marker:
                _validated_at[cache_key] = datetime.now(timezone.utc)
                _failed_refresh_until.pop(cache_key, None)
                _empty_refresh_until.pop(cache_key, None)
                cached_games.attrs["refresh_error"] = None
                return cached_games

            if (
                cached_games is not None
                and cached_marker is not None
                and refreshed_marker < cached_marker
            ):
                _failed_refresh_until[cache_key] = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=RETRY_MINUTES)
                )
                return _cache_with_warning(
                    cached_games,
                    "The NBA statistics source returned older games than the "
                    "saved log; keeping the last saved statistics.",
                )

            merged_games = (
                _merge_game_logs(cached_games, games)
                if cached_games is not None and not cached_games.empty
                else games.copy()
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.tmp")
            try:
                merged_games.to_csv(temporary_path, index=False)
                os.replace(temporary_path, path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            _validated_at[cache_key] = datetime.now(timezone.utc)
            _failed_refresh_until.pop(cache_key, None)
            _empty_refresh_until.pop(cache_key, None)
            merged_games.attrs["refresh_error"] = None
            return merged_games

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
                _validated_at.pop(cache_key, None)
                return _cache_with_warning(
                    cached_games,
                    "NBA statistics could not be refreshed. Showing the "
                    "last successfully saved game log.",
                )

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
