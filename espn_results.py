"""Completed NBA team results from ESPN's public scoreboard.

stats.nba.com and cdn.nba.com refuse requests from cloud hosts, so the saved
current-season snapshot is extended with ESPN final scores instead. Every day
is checked against the saved NBA schedule: a day is accepted only when each
scheduled regular-season game is reported final exactly once, with a clear
winner, and nothing unscheduled appears. Accepted days are contiguous, so a
day that is still in progress (or is malformed) is left for the next refresh.
"""
import logging
from datetime import timedelta

import pandas as pd
import requests
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
REQUEST_TIMEOUT_SECONDS = 8
MAX_DAYS = 45
REGULAR_SEASON_PREFIX = "00226"
FINAL_STATUS = "STATUS_FINAL"
ESPN_TEAM_ALIASES = {"LA Clippers": "Los Angeles Clippers"}

RESULT_COLUMNS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID",
    "GAME_DATE", "MATCHUP", "WL", "PTS",
]


class EspnUnavailable(RuntimeError):
    """ESPN could not be reached or returned unusable data."""


def fetch_scoreboard(day):
    try:
        response = requests.get(
            SCOREBOARD_URL,
            params={"dates": f"{day:%Y%m%d}"},
            headers={"User-Agent": "CourtVisionPro/1.0"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise EspnUnavailable(type(error).__name__) from error


def _team_lookup():
    return {team["full_name"]: team for team in teams.get_teams()}


def _normalize_game_id(value):
    game_id = str(value).strip()
    return game_id.zfill(10) if game_id.isdigit() else game_id


def _final_games(payload, team_by_name):
    """Return {(home_id, away_id): game} for the final games in a payload."""
    games = {}
    for event in payload.get("events", []):
        competition = event["competitions"][0]
        if competition["status"]["type"]["name"] != FINAL_STATUS:
            continue
        sides = {}
        for competitor in competition["competitors"]:
            name = competitor["team"]["displayName"]
            team = team_by_name[ESPN_TEAM_ALIASES.get(name, name)]
            sides[competitor["homeAway"]] = {
                "team": team,
                "score": int(competitor["score"]),
                "winner": bool(competitor.get("winner")),
            }
        home, away = sides["home"], sides["away"]
        key = (str(home["team"]["id"]), str(away["team"]["id"]))
        if key in games:
            raise ValueError(f"Duplicate final game {key} in ESPN scoreboard.")
        if home["score"] == away["score"]:
            raise ValueError(f"Tied final score for {key}.")
        if home["winner"] == away["winner"] or (
            home["winner"] != (home["score"] > away["score"])
        ):
            raise ValueError(f"Winner flag disagrees with the score for {key}.")
        games[key] = {"home": home, "away": away}
    return games


def _result_rows(game_id, day, game, season_id):
    rows = []
    for side, other, venue in (("home", "away", "vs."), ("away", "home", "@")):
        team, opponent = game[side]["team"], game[other]["team"]
        rows.append({
            "SEASON_ID": season_id,
            "TEAM_ID": int(team["id"]),
            "TEAM_ABBREVIATION": team["abbreviation"],
            "TEAM_NAME": team["full_name"],
            "GAME_ID": game_id,
            "GAME_DATE": day.isoformat(),
            "MATCHUP": f"{team['abbreviation']} {venue} {opponent['abbreviation']}",
            "WL": "W" if game[side]["winner"] else "L",
            "PTS": game[side]["score"],
        })
    return rows


def fetch_missing_team_results(
    scheduled, after_date, through_date, season_id="22026", fetch=None
):
    """Return (rows, complete_through) for the days after ``after_date``.

    ``scheduled`` is the saved schedule with ``gameId``, ``game_date`` (ET,
    text) and ``homeTeam_teamId`` / ``awayTeam_teamId`` columns. Days up to
    ``through_date`` are examined in order; ``complete_through`` is the last
    day whose scheduled games are all confirmed. Days without scheduled games
    need no request. Raises EspnUnavailable if nothing could be confirmed
    because ESPN failed.
    """
    fetch = fetch or fetch_scoreboard
    team_by_name = _team_lookup()
    rows = []
    complete_through = after_date
    confirmed_with_espn = False
    day = after_date + timedelta(days=1)

    while day <= through_date and (day - after_date).days <= MAX_DAYS:
        day_games = scheduled[scheduled["game_date"] == day.isoformat()]
        if day_games.empty:
            complete_through = day
            day += timedelta(days=1)
            continue

        game_ids = day_games["gameId"].map(_normalize_game_id)
        regular = day_games[game_ids.str.startswith(REGULAR_SEASON_PREFIX)]
        expected = {
            (str(game["homeTeam_teamId"]), str(game["awayTeam_teamId"])):
                _normalize_game_id(game["gameId"])
            for _, game in regular.iterrows()
        }
        other = {
            (str(game["homeTeam_teamId"]), str(game["awayTeam_teamId"]))
            for _, game in day_games.drop(regular.index).iterrows()
        }

        try:
            found = _final_games(fetch(day), team_by_name)
        except EspnUnavailable:
            if not confirmed_with_espn and not rows:
                raise
            logger.warning("ESPN became unavailable at %s; keeping earlier days.", day)
            break
        except (KeyError, TypeError, ValueError, IndexError) as error:
            logger.warning("Rejected ESPN results for %s (%s: %s).",
                           day, type(error).__name__, error)
            break

        if set(expected) - set(found):
            logger.info("ESPN has not finalized every game on %s yet.", day)
            break
        if set(found) - set(expected) - other:
            logger.warning("ESPN reported unscheduled games on %s; day rejected.", day)
            break

        confirmed_with_espn = True
        for key, game_id in expected.items():
            rows.extend(_result_rows(game_id, day, found[key], season_id))
        complete_through = day
        day += timedelta(days=1)

    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if frame.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("ESPN results contain duplicate team-game rows.")
    return frame, complete_through
