import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from nba_api.stats.endpoints import commonplayerinfo

CACHE_FILE = Path(__file__).with_name("player_teams.json")
CACHE_TTL = timedelta(hours=24)
TIMESTAMP_FILE = Path(__file__).with_name("player_team_timestamps.json")


def get_player_team(player_id):
    if player_id is None:
        return None

    cache = {}

    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except (ValueError, OSError):
            cache = {}

    key = str(player_id)

    timestamps = {}

    if TIMESTAMP_FILE.exists():
        try:
            timestamps = json.loads(TIMESTAMP_FILE.read_text())
        except (ValueError, OSError):
            timestamps = {}

    if key in cache and key in timestamps:
        try:
            last_updated = datetime.fromisoformat(timestamps[key])
            if datetime.now(timezone.utc) - last_updated < CACHE_TTL:
                return cache[key]
        except (ValueError, TypeError):
            pass

    try:
        info = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            timeout=10
        ).get_data_frames()[0]

        row = info.iloc[0]
        city = str(row["TEAM_CITY"]).strip()
        name = str(row["TEAM_NAME"]).strip()

        if not city or not name:
            return None

        team = f"{city} {name}"
        cache[key] = team
        CACHE_FILE.write_text(json.dumps(cache, indent=2))

        timestamps[key] = datetime.now(timezone.utc).isoformat()
        TIMESTAMP_FILE.write_text(json.dumps(timestamps, indent=2))

        return team

    except Exception as error:
        print(f"NBA team lookup error: {error}")
        return cache.get(key)
